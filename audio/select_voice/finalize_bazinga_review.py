#!/usr/bin/env python3
"""Apply both confirmed human review rounds without overwriting original data/references."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import wave

ROOT = Path(__file__).resolve().parent
VOICE = ROOT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def checked_reference(path, expected):
    if digest(path) != expected:
        raise ValueError('Reference hash mismatch: ' + str(path))
    with wave.open(str(path), 'rb') as stream:
        if stream.getsampwidth() != 2 or stream.getnframes() == 0:
            raise ValueError('Expected nonempty PCM16 WAV: ' + str(path))
        return stream.getnframes()/stream.getframerate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round1', type=Path, default=ROOT/'bazinga_reference_review.json')
    parser.add_argument('--round2', type=Path, default=ROOT/'bazinga_reference_review_round2.json')
    args = parser.parse_args()
    voices_path = ROOT/'manifests/bazinga_voices.json'
    data_path = VOICE/'data/bazinga/clean/dialogue.json'
    metadata_path = data_path.with_name('metadata.json')
    candidates_path = ROOT/'manifests/bazinga_round2_candidates.json'
    original_voices = json.loads(voices_path.read_text())
    originals = {v['character_id']: v for v in original_voices}
    data = json.loads(data_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    first = json.loads(args.round1.read_text())
    second = json.loads(args.round2.read_text())
    candidates = json.loads(candidates_path.read_text())
    if first['schema'] != 'bazinga_reference_review_v1' or second['schema'] != 'bazinga_reference_review_round2_v1':
        raise ValueError('Incorrect review schema/round')
    for review in [first, second, candidates]:
        if review['voices_sha256'] != digest(voices_path) or review['dataset_sha256'] != digest(data_path):
            raise ValueError('Source data/casting version changed')
    if second['candidates_sha256'] != digest(candidates_path):
        raise ValueError('Second-round candidate version changed')
    if second['round1_review_sha256'] != digest(args.round1) or candidates['round1_review_sha256'] != digest(args.round1):
        raise ValueError('First-round review version changed')
    target_ids = {cid for cid, r in first['records'].items() if r['decision'] in ('replace', 'exclude')}
    alternative = {v['character_id']: v for v in candidates['voices']}
    if set(second['records']) != target_ids or set(alternative) != target_ids:
        raise ValueError('Incomplete second-round role coverage')
    accepted_first = {cid: r for cid, r in first['records'].items() if r['decision'] in ('keep', 'trim')}
    if candidates['accepted_round1'] != accepted_first:
        raise ValueError('Accepted first-round ranges changed')
    lookup = {(v['story_id'], name): v for v in original_voices for name in v['source_labels']}
    base = {}
    original_three_person_count = 0
    original_three_person_utterances = 0
    for did, turns in data.items():
        names = {t['speaker'] for t in turns} | {l['name'] for t in turns for l in t['listener']}
        if len(names) < 3:
            continue
        original_three_person_count += 1
        original_three_person_utterances += len(turns)
        participants = [lookup.get((did.split('.')[0], name)) for name in names]
        if any(not v or not v.get('reference_wav') or v['duration_seconds'] <= 2 for v in participants):
            continue
        base[did] = {v['character_id'] for v in participants}
    scope = set().union(*base.values())
    if set(first['records']) != scope:
        raise ValueError('Incomplete first-round role coverage')
    decisions = {}
    crop_payloads = {}
    excluded = {}
    for cid, first_record in first['records'].items():
        original = originals[cid]
        if first_record['reviewed'] is not True:
            raise ValueError('Unconfirmed first-round role: ' + cid)
        checked_reference(ROOT/original['reference_wav'], first_record['reference_sha256'])
        if first_record['reference_sha256'] != original['reference_sha256']:
            raise ValueError('Original casting mismatch: ' + cid)
        round_number = 1
        record = first_record
        source = original
        action = first_record['decision']
        if cid in target_ids:
            round_number = 2
            record = second['records'][cid]
            source = alternative[cid]
            if record['reviewed'] is not True or record['decision'] not in ('keep', 'exclude'):
                raise ValueError('Unconfirmed or invalid second-round decision: ' + cid)
            if record['reference_sha256'] != source['reference_sha256'] or record['reference_wav'] != source['reference_wav']:
                raise ValueError('Second-round reference mismatch: ' + cid)
            if source['no_candidate'] and record['decision'] != 'exclude':
                raise ValueError('No-candidate role cannot pass: ' + cid)
            if record['decision'] == 'keep' and record['issues']:
                raise ValueError('Second-round role has unresolved issues: ' + cid)
            if not source['no_candidate']:
                checked_reference(ROOT/source['reference_wav'], source['reference_sha256'])
            if record['decision'] == 'exclude':
                excluded[cid] = {'review_round': 2, 'issues': record['issues'], 'notes': record['notes'],
                                 'no_candidate': source['no_candidate']}
                continue
            if record['trim_start_seconds'] != 0 or abs(record['trim_end_seconds']-source['duration_seconds']) > .001:
                raise ValueError('Second round does not allow unreviewed crops: ' + cid)
            action = 'replace'
        elif action not in ('keep', 'trim'):
            raise ValueError('Unresolved first-round role: ' + cid)
        voice = {**original}
        if action == 'replace':
            voice.update({k: v for k, v in source.items() if k not in {
                'previous_review', 'previous_reference_sha256', 'no_candidate', 'candidate_count'}})
        elif action == 'trim':
            start, end = record['trim_start_seconds'], record['trim_end_seconds']
            duration = original['duration_seconds']
            if not 0 <= start < end <= duration+.001:
                raise ValueError('Invalid crop: ' + cid)
            with wave.open(str(ROOT/original['reference_wav']), 'rb') as stream:
                params = stream.getparams()
                start_frame, end_frame = round(start*params.framerate), round(end*params.framerate)
                stream.setpos(start_frame)
                raw = stream.readframes(end_frame-start_frame)
            actual_duration = len(raw)/params.nchannels/params.sampwidth/params.framerate
            if actual_duration <= 2:
                raise ValueError('Accepted crop is <=2 seconds: ' + cid)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as stream:
                stream.setparams(params)
                stream.writeframes(raw)
            content = buffer.getvalue()
            sha = hashlib.sha256(content).hexdigest()
            path = ROOT/'references/bazinga_reviewed'/original['story_id']/(sha[:24]+'.wav')
            crop_payloads[path] = content
            voice.update(reference_wav=str(path.relative_to(ROOT)), reference_sha256=sha,
                         duration_seconds=actual_duration, start_time=original['start_time']+start_frame/params.framerate,
                         end_time=original['start_time']+end_frame/params.framerate,
                         transcript_scope='original_reference_before_crop; not retranscribed',
                         crop={'parent_reference_wav': original['reference_wav'],
                               'parent_reference_sha256': original['reference_sha256'],
                               'start_seconds': start, 'end_seconds': end,
                               'start_frame': start_frame, 'end_frame': end_frame})
            # Full-clip statistics no longer describe the reviewed crop.
            for field in ['rms', 'peak', 'clipped_fraction', 'quiet_fraction', 'score']:
                voice.pop(field, None)
        voice.update(review_status='human_review_accepted', short_reference=voice['duration_seconds']<3,
                     human_review={'round': round_number, 'action': action,
                                   'issues': record['issues'], 'notes': record['notes'],
                                   'updated_at': record['updated_at']})
        decisions[cid] = voice
    removed = {did: sorted(ids & set(excluded)) for did, ids in base.items() if ids & set(excluded)}
    kept = {did: data[did] for did in base if did not in removed}
    used = set().union(*(base[did] for did in kept))
    if not used <= set(decisions):
        raise ValueError('Retained dialogue has unresolved speaker/listener')
    if any(decisions[cid]['duration_seconds'] <= 2 for cid in used):
        raise ValueError('Retained role reference is <=2 seconds')
    for did, turns in kept.items():
        if did not in metadata or len(metadata[did]['utterances']) != len(turns):
            raise ValueError('Metadata alignment mismatch: ' + did)
    # All source/review/range checks finish before writing any outputs.
    for path, content in crop_payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Content-addressed crop changed: ' + str(path))
        if not path.exists():
            path.write_bytes(content)
    subset = VOICE/'data/bazinga/reference_qc_v1'
    write_json(subset/'dialogue.json', kept)
    write_json(subset/'metadata.json', {did: metadata[did] for did in kept})
    reviewed_voices = ROOT/'manifests/bazinga_voices_reviewed.json'
    write_json(reviewed_voices, [decisions[cid] for cid in sorted(decisions)])
    report = {'schema': 'bazinga_reference_final_review_v1',
              'source_data_sha256': digest(data_path), 'source_metadata_sha256': digest(metadata_path),
              'source_voices_sha256': digest(voices_path), 'round1_review_sha256': digest(args.round1),
              'round2_review_sha256': digest(args.round2), 'candidates_sha256': digest(candidates_path),
              'output_data_sha256': digest(subset/'dialogue.json'),
              'output_metadata_sha256': digest(subset/'metadata.json'),
              'output_voices_sha256': digest(reviewed_voices),
              'original_three_person_dialogues': original_three_person_count,
              'original_three_person_utterances': original_three_person_utterances,
              'review_base_dialogues': len(base), 'review_base_utterances': sum(len(data[did]) for did in base),
              'remaining_dialogues': len(kept), 'remaining_utterances': sum(map(len, kept.values())),
              'accepted_voice_slots': len(decisions), 'used_voice_slots': len(used),
              'actions': dict(Counter(v['human_review']['action'] for v in decisions.values())),
              'excluded_characters': excluded, 'excluded_dialogue_reasons': removed,
              'retained_dialogue_ids': list(kept), 'used_character_ids': sorted(used),
              'by_series': {series: {'dialogues': sum(did.startswith(series+'.') for did in kept),
                                    'utterances': sum(len(turns) for did, turns in kept.items()
                                                      if did.startswith(series+'.'))}
                            for series in ['GameOfThrones', 'SixFeetUnder', 'TheOffice']},
              'policy': '>=3 participants; speaker/listener whole-dialogue exclusion; reviewed reference >2 seconds'}
    write_json(ROOT/'manifests/bazinga_final_review.json', report)
    write_json(subset/'reference_review.json', report)
    print(json.dumps({k: report[k] for k in ['remaining_dialogues', 'remaining_utterances',
                                            'accepted_voice_slots', 'used_voice_slots', 'actions',
                                            'excluded_characters', 'by_series']}, ensure_ascii=False))
    print('Subset: ' + str(subset/'dialogue.json'))
    print('Reviewed voices: ' + str(reviewed_voices))


if __name__ == '__main__':
    main()
