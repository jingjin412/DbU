#!/usr/bin/env python3
"""Build a portable, offline reference-review UI for the >=3-person, >2s subset."""
import array
import base64
import hashlib
import json
import sys
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOICE = ROOT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def waveform(path, bins=240):
    with wave.open(str(path), 'rb') as stream:
        if stream.getsampwidth() != 2:
            raise ValueError('Expected PCM16: ' + str(path))
        samples = array.array('h', stream.readframes(stream.getnframes()))
        if sys.byteorder != 'little':
            samples.byteswap()
    step = max(1, len(samples) // bins)
    return [round(max(abs(x) for x in samples[i:i+step]) / 32768, 4)
            for i in range(0, len(samples), step)]


def main():
    voice_path = ROOT / 'manifests/bazinga_voices.json'
    data_path = VOICE / 'data/bazinga/clean/dialogue.json'
    voices = json.loads(voice_path.read_text())
    data = json.loads(data_path.read_text())
    lookup = {(v['story_id'], name): v for v in voices for name in v['source_labels']}
    dialogues = []
    needed = set()
    for did, turns in data.items():
        names = {t['speaker'] for t in turns} | {l['name'] for t in turns for l in t['listener']}
        if len(names) < 3:
            continue
        participants = [lookup.get((did.split('.')[0], name)) for name in sorted(names)]
        if any(not v or not v.get('reference_wav') or v['duration_seconds'] <= 2
               or not (ROOT / v['reference_wav']).is_file() for v in participants):
            continue
        ids = [v['character_id'] for v in participants]
        needed.update(ids)
        dialogues.append({'dialogue_id': did, 'utterances': len(turns), 'characters': ids})
    counts = Counter(cid for d in dialogues for cid in d['characters'])
    records = []
    for voice in voices:
        if voice['character_id'] not in needed:
            continue
        path = ROOT / voice['reference_wav']
        if digest(path) != voice['reference_sha256']:
            raise ValueError('Reference hash changed: ' + str(path))
        records.append({**{k: voice[k] for k in [
            'character_id', 'character', 'story_id', 'reference_wav', 'reference_sha256',
            'duration_seconds', 'transcript', 'episode_id', 'start_time', 'end_time']},
            'dialogue_count': counts[voice['character_id']], 'waveform': waveform(path),
            'audio': 'data:audio/wav;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')})
    payload = {'schema': 'bazinga_reference_review_v1', 'voices_sha256': digest(voice_path),
               'dataset_sha256': digest(data_path), 'min_participants': 3,
               'reference_min_seconds_exclusive': 2, 'voices': records, 'dialogues': dialogues}
    template = (ROOT / 'bazinga_review.template.html').read_text()
    assert template.count('__REVIEW_DATA__') == 1
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    output = ROOT / 'bazinga_review.html'
    output.write_text(template.replace('__REVIEW_DATA__', encoded))
    print(json.dumps({'output': str(output), 'references': len(records), 'dialogues': len(dialogues),
                      'utterances': sum(d['utterances'] for d in dialogues),
                      'file_megabytes': round(output.stat().st_size / 1e6, 2)}))


if __name__ == '__main__':
    main()
