#!/usr/bin/env python3
"""Convert pinned Bazinga word transcripts to MPDD shape using the standard library."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
VOICE = ROOT.parent
SHOWS = ('GameOfThrones', 'SixFeetUnder', 'TheOffice')
SPECIAL = {'all', 'multiple_persons', 'unknown', 'not_available', '?'}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 6 or fields[0] != path.stem:
            raise ValueError(f'Invalid source row: {path}:{number}')
        rows.append({'line': number, 'speaker': fields[1], 'start': float(fields[2]),
                     'end': float(fields[3]), 'token': fields[4],
                     'addressee': fields[7] if len(fields) > 7 else '?'})
    return rows


def detokenize(tokens):
    text = ' '.join(tokens)
    text = re.sub(r"\b(\w+)\s+(['’])\s+(s|re|ve|ll|d|m)\b", r'\1\2\3', text)
    text = re.sub(r"\b(\w+n)\s+(['’])\s+t\b", r'\1\2t', text)
    text = re.sub(r"\s+([.,!?;:%\)\]\}])", r'\1', text)
    text = re.sub(r"([\(\[\{])\s+", r'\1', text)
    text = re.sub(r"\s+(['’](?:s|re|ve|ll|d|m)|n['’]t)\b", r'\1', text)
    return text


def listeners(raw):
    if raw in {'_', '?'}:
        return []
    return [{'name': name} for name in dict.fromkeys(raw.split('@'))]


def convert(rows, episode):
    groups = []
    for row in rows:
        if (not groups
                or (groups[-1][-1]['speaker'], groups[-1][-1]['addressee']) != (row['speaker'], row['addressee'])
                or (row['speaker'] in {'not_available', '?'} and groups[-1][-1]['token'] in {'.', '!', '?'})):
            groups.append([])
        groups[-1].append(row)
    turns, metadata = [], []
    for i, group in enumerate(groups):
        first = group[0]
        raw = first['addressee']
        names = [x['name'] for x in listeners(raw)]
        status = ('missing' if raw == '?' else 'no_specific_addressee' if raw == '_' else
                  'unspecified_group_or_unknown' if any(n in SPECIAL or 'unknown' in n for n in names) else 'explicit')
        turns.append({'speaker': first['speaker'], 'utterance': detokenize([r['token'] for r in group]),
                      'listener': listeners(raw)})
        speaker_status = 'missing' if first['speaker'] in {'not_available', '?'} else 'group' if first['speaker'] in {'all', 'multiple_persons'} else 'unknown_identity' if 'unknown' in first['speaker'] else 'named'
        metadata.append({'speaker_status': speaker_status, 'utterance_id': f'{episode}:u{i+1:05d}', 'turn_index': i,
                         'source_line_start': first['line'], 'source_line_end': group[-1]['line'],
                         'token_count': len(group), 'start_time': min(r['start'] for r in group),
                         'end_time': max(r['end'] for r in group), 'raw_addressee': raw,
                         'addressee_status': status,
                         'explicit_addressee_target': status == 'explicit'})
    return turns, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpdd-source', type=Path, required=True,
                        help='Directory containing dialogue_chinese.json, metadata.json and readme.txt')
    args = parser.parse_args()
    mpdd_source = args.mpdd_source.resolve()
    for name in ('dialogue_chinese.json', 'metadata.json', 'readme.txt'):
        if not (mpdd_source/name).is_file():
            parser.error(f'Missing MPDD input: {mpdd_source/name}')
    (VOICE/'data/mpdd').mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT/'raw/source_manifest.json').read_text())
    for record in source['files']:
        if digest(ROOT/'raw'/record['path']) != record['sha256']:
            raise ValueError(f'Source checksum mismatch: {record["path"]}')
    combined, metadata, audit, stats = {}, {}, [], {}
    for show in SHOWS:
        folder = ROOT/'raw/data'/show
        ids = [line.split(',', 1)[0] for line in (folder/'episodes.txt').read_text().splitlines() if line.strip()]
        characters = {row[0]: {'name': row[2], 'actor': row[3]} for row in csv.reader((folder/'characters.txt').open()) if row}
        show_data, counter = {}, Counter()
        for episode in ids:
            path = folder/f'{episode}.txt'
            fields = [line.split() for line in path.read_text().splitlines() if line.strip()]
            counts = Counter(r[7] if len(r) > 7 else '?' for r in fields)
            selected = any(label not in {'?', '_'} for label in counts)
            audit.append({'episode_id': episode, 'series': show, 'selected': selected,
                          'reason': 'has_addressee_annotations' if selected else 'no_explicit_addressee_annotations',
                          'tokens': len(fields), 'malformed_rows': sum(len(r) < 6 or r[0] != episode for r in fields),
                          'missing_addressee_tokens': counts['?'],
                          'no_specific_addressee_tokens': counts['_']})
            if not selected:
                continue
            rows = read_rows(path)
            turns, turn_meta = convert(rows, episode)
            show_data[episode] = turns
            participants = sorted({r['speaker'] for r in rows} | {l['name'] for t in turns for l in t['listener']})
            metadata[episode] = {'dataset': 'bazinga', 'series': show, 'episode_id': episode,
                                 'dialogue_unit': 'episode', 'source_path': f'data/{show}/{episode}.txt',
                                 'source_sha256': digest(path), 'official_split': 'test' if '.Season01.' in episode else 'validation' if any(s in episode for s in ('.Season02.', '.Season03.')) else 'train',
                                 'observed_participant_ids': participants,
                                 'participant_scope': 'full_episode_speakers_and_addressees_including_special_values; not frozen task candidates',
                                 'audio_downloaded': False,
                                 'audio_url': f'{source["repository"]}/resolve/{source["revision"]}/data/{show}/{episode}.en.wav',
                                 'utterances': turn_meta}
            counter.update({'episodes': 1, 'utterances': len(turns), 'tokens': len(rows)})
            counter.update(m['addressee_status'] for m in turn_meta)
            counter['missing_speaker_utterances'] += sum(m['speaker_status'] == 'missing' for m in turn_meta)
            counter['missing_speaker_tokens'] += sum(m['token_count'] for m in turn_meta if m['speaker_status'] == 'missing')
            counter['multi_named_addressee_utterances'] += sum(m['addressee_status'] == 'explicit' and '@' in m['raw_addressee'] for m in turn_meta)
        combined.update(show_data)
        stats[show] = {'scanned_episodes': len(ids), **dict(counter)}
        write_json(VOICE/f'data/bazinga/{show}/dialogue.json', show_data)
        write_json(VOICE/f'data/bazinga/{show}/characters.json', characters)
    write_json(VOICE/'data/bazinga/dialogue.json', combined)
    write_json(VOICE/'data/bazinga/metadata.json', metadata)
    write_json(VOICE/'data/bazinga/episode_inventory.json', audit)
    write_json(VOICE/'data/bazinga/statistics.json', stats)
    write_json(VOICE/'data/bazinga/source_manifest.json', source)
    for name in ('dialogue_chinese.json', 'metadata.json', 'readme.txt'):
        shutil.copyfile(mpdd_source/name, VOICE/'data/mpdd'/name)
    mpdd = json.loads((mpdd_source/'dialogue_chinese.json').read_text())
    stripped = {key: [{'speaker': t['speaker'], 'utterance': t['utterance'],
                       'listener': [{'name': l['name']} for l in t['listener']]} for t in turns]
                for key, turns in mpdd.items()}
    write_json(VOICE/'data/mpdd/dialogue.json', stripped)
    write_json(VOICE/'data/mpdd/provenance.json', {
        'source': str(mpdd_source/'dialogue_chinese.json'), 'sha256': digest(mpdd_source/'dialogue_chinese.json'),
        'original_copy': 'dialogue_chinese.json', 'normalized_copy': 'dialogue.json',
        'normalization': 'remove emotion and listener.relation only; preserve all dialogue IDs and utterances',
        'dialogues': len(mpdd), 'utterances': sum(map(len, mpdd.values()))})
    print(json.dumps(stats, indent=2))
    print('MPDD:', len(mpdd), 'dialogues,', sum(map(len, mpdd.values())), 'utterances')


if __name__ == '__main__':
    main()
