#!/usr/bin/env python3
"""Select a disjoint alternative per failed round-one role; build an offline final review."""
import argparse
import array
import base64
import bisect
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import wave

from build_bazinga_review import digest, waveform

ROOT = Path(__file__).resolve().parent
VOICE = ROOT.parents[1]


def read_rows(path, duration):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) < 6:
            continue
        start, end = float(parts[2]), float(parts[3])
        if 0 <= start <= end <= duration and end - start < 10:
            rows.append({'speaker': parts[1], 'start': start, 'end': end,
                         'text': parts[4], 'confidence': float(parts[5]), 'line': line_number})
    return rows


def candidate_windows(rows, series, episode, targets, originals):
    ordered = sorted(rows, key=lambda r: r['start'])
    starts = [r['start'] for r in ordered]
    groups = []
    for row in rows:
        if (not groups or row['speaker'] != groups[-1][-1]['speaker']
                or row['start'] < groups[-1][-1]['start'] - .05
                or row['start'] - groups[-1][-1]['end'] > .65):
            groups.append([])
        groups[-1].append(row)
    for group in groups:
        cid = series + ':' + group[0]['speaker']
        if cid not in targets:
            continue
        old = originals[cid]
        for a in range(len(group)):
            if (a and group[a]['start'] - group[a-1]['end'] < .15
                    and group[a-1]['text'] not in ['.', '!', '?']):
                continue
            for b in range(a, len(group)):
                window = group[a:b+1]
                start = window[0]['start']
                end = max(r['end'] for r in window)
                duration = end - start
                if duration > 9:
                    break
                if duration <= 2 or len(window) < 4:
                    continue
                if (b+1 < len(group) and group[b+1]['start'] - group[b]['end'] < .15
                        and group[b]['text'] not in ['.', '!', '?']):
                    continue
                # A fresh region, including a two-second buffer around the rejected original.
                if (episode == old['episode_id']
                        and min(end, old['end_time']+2) > max(start, old['start_time']-2)):
                    continue
                nearby = ordered[max(0, bisect.bisect_left(starts, start-10)):
                                 bisect.bisect_left(starts, end+.15)]
                # Require a 150ms boundary guard from differently annotated speakers.
                if any(r['speaker'] != group[0]['speaker']
                       and min(end+.15, r['end']) > max(start-.15, r['start']) for r in nearby):
                    continue
                confidence = sum(r['confidence'] for r in window) / len(window)
                if confidence < .3:
                    continue
                # Prefer >=3s, high-confidence alignment, then lengths near six seconds.
                tier = 0 if duration >= 3 and confidence >= .65 else 1 if confidence >= .65 else 2
                yield cid, {'episode_id': episode, 'start_time': start, 'end_time': end,
                            'duration_seconds': duration, 'alignment_confidence_mean': confidence,
                            'source_line_start': window[0]['line'], 'source_line_end': window[-1]['line'],
                            'transcript': ' '.join(r['text'] for r in window), 'selection_tier': tier,
                            'score': abs(duration-6) + (1-confidence)*8}


def extract(candidate, folder):
    source = ROOT / 'sources/bazinga' / (candidate['episode_id'] + '.en.wav')
    with wave.open(str(source), 'rb') as stream:
        if stream.getsampwidth() != 2:
            raise ValueError('Expected PCM16')
        rate, channels = stream.getframerate(), stream.getnchannels()
        start_frame = round(candidate['start_time'] * rate)
        end_frame = round(candidate['end_time'] * rate)
        stream.setpos(start_frame)
        raw = stream.readframes(end_frame-start_frame)
    samples = array.array('h', raw)
    if sys.byteorder != 'little':
        samples.byteswap()
    rms = (sum(x*x for x in samples)/len(samples)) ** .5 / 32768
    clipped = sum(abs(x) >= 32735 for x in samples)/len(samples)
    if rms < .005 or clipped > .01:
        return None
    sha = hashlib.sha256(raw).hexdigest()
    target = folder / (sha[:24] + '.wav')
    if not target.exists():
        with wave.open(str(target), 'wb') as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(raw)
    return {**candidate, 'source_path': str(source.relative_to(ROOT)),
            'reference_wav': str(target.relative_to(ROOT)), 'reference_sha256': digest(target),
            'duration_seconds': len(raw)/2/channels/rate, 'sample_rate': rate,
            'rms': rms, 'clipped_fraction': clipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, default=ROOT/'bazinga_reference_review.json')
    args = parser.parse_args()
    review = json.loads(args.review.read_text())
    voice_path = ROOT/'manifests/bazinga_voices.json'
    data_path = VOICE/'data/bazinga/clean/dialogue.json'
    voices = json.loads(voice_path.read_text())
    originals = {v['character_id']: v for v in voices}
    if review['schema'] != 'bazinga_reference_review_v1':
        raise ValueError('Need round-one export')
    if review['voices_sha256'] != digest(voice_path) or review['dataset_sha256'] != digest(data_path):
        raise ValueError('Round-one review source version changed')
    targets = set()
    for cid, record in review['records'].items():
        old = originals[cid]
        if not record['reviewed'] or record['reference_sha256'] != digest(ROOT/old['reference_wav']):
            raise ValueError('Unconfirmed or changed first-round reference: ' + cid)
        # A reviewed trim is accepted; only replacement/exclusion roles need a new clip.
        if record['decision'] in ('replace', 'exclude'):
            targets.add(cid)
    if not targets:
        raise ValueError('No failed roles to review')
    candidates = defaultdict(list)
    episodes = json.loads((VOICE/'data/bazinga/metadata.json').read_text())
    for episode, meta in episodes.items():
        if not any(originals[cid]['story_id'] == meta['series'] for cid in targets):
            continue
        source = ROOT/'sources/bazinga'/(episode+'.en.wav')
        if not source.is_file():
            continue
        with wave.open(str(source), 'rb') as stream:
            duration = stream.getnframes()/stream.getframerate()
        annotation = VOICE/'datagen/raw'/meta['source_path']
        if digest(annotation) != meta['source_sha256']:
            raise ValueError('Annotation changed: ' + episode)
        for cid, candidate in candidate_windows(read_rows(annotation, duration), meta['series'],
                                                 episode, targets, originals):
            candidate['annotation_sha256'] = meta['source_sha256']
            candidates[cid].append(candidate)
    lookup = {(v['story_id'], n): v for v in voices for n in v['source_labels']}
    dialogues = []
    for did, turns in json.loads(data_path.read_text()).items():
        names = {t['speaker'] for t in turns} | {l['name'] for t in turns for l in t['listener']}
        if len(names) < 3:
            continue
        participants = [lookup.get((did.split('.')[0], n)) for n in sorted(names)]
        if any(not v or not v.get('reference_wav') or v['duration_seconds'] <= 2 for v in participants):
            continue
        dialogues.append({'dialogue_id': did, 'utterances': len(turns),
                          'characters': [v['character_id'] for v in participants]})
    expected = {cid for dialogue in dialogues for cid in dialogue['characters']}
    if expected != set(review['records']):
        raise ValueError('Round-one review coverage changed')
    counts = Counter(cid for dialogue in dialogues for cid in dialogue['characters'])
    folder = ROOT/'references/bazinga_round2'
    folder.mkdir(parents=True, exist_ok=True)
    selected = []
    for cid in sorted(targets):
        old = originals[cid]
        replacement = None
        for candidate in sorted(candidates[cid], key=lambda c: (c['selection_tier'], c['score'],
                                                               c['episode_id'], c['start_time'])):
            replacement = extract(candidate, folder)
            if replacement:
                break
        shared = {'character_id': cid, 'character': old['character'], 'story_id': old['story_id'],
                  'dialogue_count': counts[cid], 'previous_review': review['records'][cid],
                  'previous_reference_sha256': old['reference_sha256'],
                  'candidate_count': len(candidates[cid])}
        if replacement:
            if replacement['reference_sha256'] == old['reference_sha256']:
                raise ValueError('Alternative equals rejected original')
            selected.append({**shared, **replacement, 'no_candidate': False})
        else:
            # Kept in the UI explicitly, with a default exclusion and no fake audio.
            selected.append({**shared, 'no_candidate': True, 'reference_wav': None,
                             'reference_sha256': 'no_candidate:'+old['reference_sha256'],
                             'duration_seconds': 0, 'episode_id': '', 'start_time': 0, 'end_time': 0,
                             'transcript': '沒有找到符合 >2 秒、不同原區段、無異人標註重疊條件的新候選。'})
    manifest = {'schema': 'bazinga_reference_candidates_round2_v1',
                'round1_review_sha256': digest(args.review), 'voices_sha256': digest(voice_path),
                'dataset_sha256': digest(data_path), 'review_round': 2,
                'accepted_round1': {cid: record for cid, record in review['records'].items()
                                    if record['decision'] in ('keep', 'trim')},
                'policy': {'target': 'round-one replace or exclude roles only; trim is accepted',
                           'accepted_trim_policy': 'use the reviewed round-one start/end range',
                           'min_seconds_exclusive': 2,
                           'original_region_guard_seconds': 2, 'other_speaker_guard_seconds': .15,
                           'preferred_alignment_confidence': .65, 'minimum_alignment_confidence': .3,
                           'second_round_failure': 'exclude whole dialogue including listeners',
                           'human_identity_and_background_review_required': True},
                'voices': selected, 'dialogues': dialogues}
    manifest_path = ROOT/'manifests/bazinga_round2_candidates.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    page_records = []
    for item in selected:
        if item['no_candidate']:
            page_records.append({**item, 'audio': '', 'waveform': []})
        else:
            path = ROOT/item['reference_wav']
            page_records.append({**item, 'audio': 'data:audio/wav;base64,'+
                                 base64.b64encode(path.read_bytes()).decode('ascii'),
                                 'waveform': waveform(path)})
    payload = {**manifest, 'schema': 'bazinga_reference_review_round2_v1',
               'candidates_sha256': digest(manifest_path), 'voices': page_records}
    template = (ROOT/'bazinga_review.template.html').read_text()
    template = template.replace('<title>Bazinga Reference 檢查工作台</title>',
                                '<title>Bazinga 第二輪 Reference 篩選</title>')
    template = template.replace('<h1>角色原音檢查工作台</h1>', '<h1>第二輪 · 替代原音篩選</h1>')
    template = template.replace('119 個 reference · 至少 3 位參與者 · 原始 reference 大於 2 秒。預設「沒問題」；聽完按「確認並下一個」，有問題再勾選。',
                                f'{len(selected)} 位第一輪標為更換／排除的角色，每人最多一個全新片段。第一輪保留與裁切者已通過，不需重聽。預設「通過」；有任何問題就排除角色所在整段對話。沒有新候選者預設排除。')
    template = template.replace('max="119"', f'max="{len(selected)}"')
    template = template.replace('裁切選項只記錄時間與試聽，不會修改 WAV。換 reference／排除角色，或裁切後剩 ≤2 秒，會在上方估算中移除相關整段 dialogue。',
                                '這是最後一輪：有任何問題即排除，不再裁切或更換。第一輪保留者沿用原音，裁切者沿用你確認的裁切範圍；未確認角色仍納入上方估算。')
    template = template.replace('沒問題（預設）</label>', '通過，沒問題（預設）</label>')
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    output = ROOT/'bazinga_review_round2.html'
    output.write_text(template.replace('__REVIEW_DATA__', encoded))
    print(json.dumps({'page': str(output), 'target_roles': len(selected),
                      'new_candidates': sum(not r['no_candidate'] for r in selected),
                      'no_candidate': [r['character_id'] for r in selected if r['no_candidate']],
                      'fallback_candidates': [r['character_id'] for r in selected
                                              if not r['no_candidate'] and r['selection_tier'] != 0],
                      'megabytes': round(output.stat().st_size/1e6, 2)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
