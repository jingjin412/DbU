#!/usr/bin/env python3
"""Remove invalid annotations and split at every removed span, without scene heuristics."""
import json
from collections import Counter
from pathlib import Path
from statistics import median

from prepare_data import VOICE, SHOWS, digest, write_json

BASE = VOICE/'data/bazinga'
OUT = BASE/'clean'
SPECIAL = {'', '_', '?', 'all', 'multiple_persons', 'unknown', 'not_available', 'none', 'null'}


def invalid_id(value):
    return value.lower() in SPECIAL or 'unknown' in value.lower()


def reasons(turn):
    result = []
    if invalid_id(turn['speaker']):
        result.append('invalid_speaker')
    if not turn['listener']:
        result.append('no_addressee')
    elif any(invalid_id(item['name']) for item in turn['listener']):
        result.append('invalid_addressee')
    return result


def runs(turns):
    """Return maximal contiguous valid source-index runs; never bridge a removed turn."""
    result, current = [], []
    for i, turn in enumerate(turns):
        if reasons(turn):
            if current:
                result.append(current)
                current = []
        else:
            current.append(i)
    if current:
        result.append(current)
    return result


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    def percentile(p):
        index = (len(values)-1)*p
        lo = int(index)
        return round(values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(index-lo), 3)
    return {'count': len(values), 'min': round(values[0], 3), 'median': round(median(values), 3),
            'p90': percentile(.9), 'p95': percentile(.95), 'p99': percentile(.99), 'max': round(values[-1], 3)}


def summarize(data, metadata, source_count, removed):
    lengths = [len(t) for t in data.values()]
    speakers = Counter(len({t['speaker'] for t in turns}) for turns in data.values())
    participants = Counter(len(m['observed_participant_ids']) for m in metadata.values())
    retained = sum(lengths)
    return {'source_utterances': source_count, 'retained_utterances': retained,
            'removed_utterances': len(removed), 'retained_fraction': round(retained/source_count, 6) if source_count else 0,
            'dialogues': len(data), 'length_distribution': distribution(lengths),
            'length_histogram': dict(sorted(Counter(lengths).items())),
            'speaker_count_histogram': dict(sorted(speakers.items())),
            'participant_count_histogram': dict(sorted(participants.items())),
            'single_utterance_dialogues': lengths.count(1),
            'multi_label_utterances': sum(len(t['listener']) > 1 for turns in data.values() for t in turns),
            'removal_reasons_nonexclusive': dict(Counter(r for x in removed for r in x['reasons']))}


def main():
    source = json.loads((BASE/'dialogue.json').read_text())
    source_meta = json.loads((BASE/'metadata.json').read_text())
    data, metadata, removed, gaps = {}, {}, [], []
    for episode, turns in source.items():
        em = source_meta[episode]
        for i, turn in enumerate(turns):
            why = reasons(turn)
            if why:
                removed.append({'episode_id': episode, 'source_turn_index': i,
                                'utterance_id': em['utterances'][i]['utterance_id'],
                                'speaker': turn['speaker'], 'listener': turn['listener'], 'reasons': why})
        for indices in runs(turns):
            start, end = indices[0], indices[-1]
            key = f'{episode}:d{start+1:05d}-{end+1:05d}'
            data[key] = [turns[i] for i in indices]
            participants = sorted({t['speaker'] for t in data[key]} | {x['name'] for t in data[key] for x in t['listener']})
            metadata[key] = {'episode_id': episode, 'series': em['series'], 'official_split': em['official_split'],
                             'dialogue_unit': 'contiguous_valid_annotation_run',
                             'source_turn_start': start, 'source_turn_end': end,
                             'boundary_before': 'episode_start' if start == 0 else 'removed_utterance',
                             'boundary_after': 'episode_end' if end == len(turns)-1 else 'removed_utterance',
                             'observed_participant_ids': participants,
                             'participant_scope': 'full_segment; not frozen inference candidates',
                             'utterances': [{**em['utterances'][i], 'source_turn_index': i, 'turn_index': j} for j, i in enumerate(indices)]}
            for left, right in zip(indices, indices[1:]):
                a, b = turns[left], turns[right]
                ma, mb = em['utterances'][left], em['utterances'][right]
                roles_a = {a['speaker']} | {x['name'] for x in a['listener']}
                roles_b = {b['speaker']} | {x['name'] for x in b['listener']}
                gaps.append({'dialogue_id': key, 'episode_id': episode,
                             'left_utterance_id': ma['utterance_id'], 'right_utterance_id': mb['utterance_id'],
                             'gap_seconds': round(mb['start_time'] - ma['end_time'], 3),
                             'disjoint_adjacent_participants': roles_a.isdisjoint(roles_b),
                             'left': a, 'right': b})
    stats = {'all': summarize(data, metadata, sum(map(len, source.values())), removed), 'by_series': {}}
    for show in SHOWS:
        sd = {k: v for k, v in data.items() if metadata[k]['series'] == show}
        sm = {k: metadata[k] for k in sd}
        sr = [x for x in removed if source_meta[x['episode_id']]['series'] == show]
        count = sum(len(v) for k, v in source.items() if source_meta[k]['series'] == show)
        stats['by_series'][show] = summarize(sd, sm, count, sr)
        write_json(OUT/show/'dialogue.json', sd)
    gap_values = [g['gap_seconds'] for g in gaps]
    gap_report = {'scope': 'adjacent retained source utterances within cleaned dialogues only',
                  'distribution_seconds': distribution(gap_values),
                  'negative_gap_count': sum(v < 0 for v in gap_values),
                  'threshold_diagnostics_only': {str(t): {'gaps_at_least_threshold': sum(v >= t for v in gap_values),
                      'also_disjoint_adjacent_participants': sum(g['gap_seconds'] >= t and g['disjoint_adjacent_participants'] for g in gaps)} for t in (2, 5, 10, 20, 30, 60)},
                  'longest_gap_examples': sorted(gaps, key=lambda x: x['gap_seconds'], reverse=True)[:20],
                  'largest_overlap_examples': sorted((g for g in gaps if g['gap_seconds'] < 0), key=lambda x: x['gap_seconds'])[:10],
                  'disjoint_participant_examples': sorted((g for g in gaps if g['disjoint_adjacent_participants']), key=lambda x: x['gap_seconds'], reverse=True)[:10]}
    write_json(OUT/'dialogue.json', data)
    write_json(OUT/'metadata.json', metadata)
    write_json(OUT/'removed_utterances.json', removed)
    write_json(OUT/'statistics.json', stats)
    write_json(OUT/'gap_analysis.json', gap_report)
    write_json(OUT/'config.json', {'version': 1, 'source_dialogue_sha256': digest(BASE/'dialogue.json'),
                                 'source_metadata_sha256': digest(BASE/'metadata.json'),
                                 'invalid_exact_ids_case_insensitive': sorted(SPECIAL), 'invalid_id_substring': 'unknown',
                                 'mixed_valid_invalid_addressees': 'remove_entire_utterance',
                                 'split_at_every_removed_span': True, 'time_gap_threshold': None,
                                 'participant_change_splitting': False, 'minimum_utterances': 1,
                                 'minimum_speakers': 1, 'multi_label': 'preserve_all_named_addressees'})
    print(json.dumps({'all': stats['all'], 'gaps': gap_report['distribution_seconds'],
                      'thresholds': gap_report['threshold_diagnostics_only']}, indent=2))


if __name__ == '__main__':
    main()
