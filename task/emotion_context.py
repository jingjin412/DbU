"""Attach verified per-turn emotion labels to visible text history only."""
import json
from collections import Counter
from types import SimpleNamespace

from data import samples
from tasks import LABELS, prompt

VERSION = 'voice-text-history-emotion-v1'


def load(args):
    predicted = getattr(args, 'history_emotions', None)
    gold_path = getattr(args, 'history_emotion_gold', None)
    if not predicted and not gold_path:
        return None
    if predicted and gold_path:
        raise ValueError('Choose predicted or gold history emotions')
    if args.condition not in ('text', 'neutral_audio') or args.task not in ('downstream', 'addressee_recognition', 'next_speaker_ranking'):
        raise ValueError('History emotions require text or neutral_audio downstream tasks')
    # Emotion source must use identical visible cutoffs, role names and text.
    builder = SimpleNamespace(**vars(args))
    builder.max_samples = 0
    builder.gold_emotions = None
    expected = samples(builder, 'emotion_recognition')
    entries = {}
    if predicted:
        rows = [json.loads(line) for line in predicted.read_text().splitlines() if line.strip()]
        indexed = {row['sample_id']: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('Duplicate emotion sample IDs')
        for sample in expected:
            row = indexed.get(sample['sample_id'])
            if row is None:
                raise ValueError('Missing emotion prediction: ' + sample['sample_id'])
            for field in ('dataset', 'dialogue_id', 'task', 'naming', 'context_start', 'context_end',
                          'target_utterance_id', 'candidate_mapping', 'candidates'):
                if row.get(field) != sample[field]:
                    raise ValueError('Emotion source mismatch: ' + field + ': ' + sample['sample_id'])
            if row['condition'] != 'text' or row['prompt'] != prompt(sample):
                raise ValueError('Emotion source history/prompt differs: ' + sample['sample_id'])
            if row['status'] not in ('ok', 'invalid', 'error'):
                raise ValueError('Unknown emotion response status')
            if row['status'] == 'ok' and row['prediction'] not in LABELS:
                raise ValueError('Unknown predicted emotion')
            entries[sample['target_utterance_id']] = {
                'emotion': row['prediction'] if row['status'] == 'ok' else 'unavailable',
                'source_status': row['status'], 'source_sample_id': row['sample_id']}
    else:
        gold = json.loads(gold_path.read_text())
        dialogue = json.loads(args.data.read_text())
        for sample in expected:
            did, index = sample['dialogue_id'], sample['context_end']
            source_turn = gold[did][index]
            turn = dialogue[did][index]
            if (source_turn['speaker'], source_turn['utterance']) != (turn['speaker'], turn['utterance']):
                raise ValueError(f'Gold emotion text/speaker differs: {did}:{index}')
            if source_turn.get('emotion') not in LABELS:
                raise ValueError(f'Missing/unknown gold emotion: {did}:{index}')
            entries[sample['target_utterance_id']] = {
                'emotion': source_turn['emotion'], 'source_status': 'gold'}
    return entries


def attach(selected, entries):
    usage = {}
    for sample in selected:
        sample['history_emotion_usage'] = []
        for turn in sample['history']:
            uid = turn['utterance_id']
            entry = entries[uid]
            turn['provided_emotion'] = entry['emotion']
            detail = {'utterance_id': uid, **entry}
            sample['history_emotion_usage'].append(detail)
            usage[uid] = detail
        sample['history_emotion_unavailable_count'] = sum(
            entry['emotion'] == 'unavailable' for entry in sample['history_emotion_usage'])
    return {'visible_unique_utterances': len(usage),
            'source_status_counts': dict(Counter(entry['source_status'] for entry in usage.values())),
            'unavailable_utterances': sum(entry['emotion'] == 'unavailable' for entry in usage.values()),
            'samples_with_unavailable_emotion': sum(s['history_emotion_unavailable_count'] > 0 for s in selected)}
