#!/usr/bin/env python3
"""Join frozen neutral casting with one model's text emotion predictions (CPU only)."""
import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOICE = ROOT.parents[1]
from emotion_settings import STRENGTHS, validate_jobs
MODELS = {'qwen3omni': 'qwen3-instruct', 'gemma4e4b': 'gemma4-e4b',
          'voxtral': 'voxtral-small-24b'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(dataset, model, naming, predictions=None, invalid_policy='error'):
    kind = MODELS[model]
    base_path = ROOT/'plans'/('mpdd_neutral.json' if dataset == 'mpdd'
                              else 'bazinga_reviewed_neutral.json')
    base = json.loads(base_path.read_text())
    if base['blocked']:
        raise ValueError('Resolve blocked neutral casting first')
    validate_jobs(base['jobs'], base['versions']['emotion_vector_order'])
    if base['versions'] != json.loads((ROOT/'versions.json').read_text()):
        raise ValueError('Casting/runtime versions differ')
    if predictions is None:
        raise ValueError('Provide --predictions from your own text emotion run')
    predictions = predictions.resolve()
    cfg_path = predictions.with_name('config.json')
    cfg = json.loads(cfg_path.read_text())
    if (cfg['task'], cfg['dataset'], cfg['condition'], cfg['model_kind']) != (
            'emotion_recognition', dataset, 'text', kind):
        raise ValueError('Wrong prediction task/dataset/condition/model')
    source_path = Path(cfg['data'])
    observed_source_sha256 = digest(source_path)
    sys.path.insert(0, str(VOICE/'task'))
    from tasks import prompt
    actual_naming = cfg.get('naming', 'anonymous')
    if actual_naming != naming:
        raise ValueError('Prediction naming does not match requested naming')
    source = json.loads(source_path.read_text())
    # Existing Bazinga predictions used the full clean data; casting uses the reviewed
    # subset. Verify every retained source turn, without claiming subset re-inference.
    source_turns = {}
    if dataset == 'bazinga':
        metadata_path = Path(cfg.get('metadata') or source_path.with_name('metadata.json'))
        if cfg.get('metadata_sha256') and digest(metadata_path) != cfg['metadata_sha256']:
            raise ValueError('Prediction metadata changed')
        metadata = json.loads(metadata_path.read_text())
    for did, turns in source.items():
        for i, turn in enumerate(turns):
            uid = (f'mpdd:{did}:u{i+1:05d}' if dataset == 'mpdd'
                   else metadata[did]['utterances'][i]['utterance_id'])
            if uid in source_turns:
                raise ValueError('Duplicate source utterance ID')
            source_turns[uid] = (did, turn)
    rows = {}
    for line in predictions.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = row['target_utterance_id']
        if uid in rows:
            raise ValueError('Duplicate prediction utterance ID')
        rows[uid] = row
    if model == 'voxtral':
        summary_path = predictions.with_name('metrics.json')
        if not summary_path.is_file():
            raise ValueError('Voxtral predictions are not finalized: metrics.json missing')
        summary = json.loads(summary_path.read_text())
        if summary['samples'] != len(rows) or cfg.get('max_samples', 0) or cfg.get('max_dialogues', 0):
            raise ValueError('Expected completed full-dataset Voxtral emotion predictions')
    jobs, skipped = [], []
    for original in base['jobs']:
        uid = original['utterance_id']
        source_did, turn = source_turns[uid]
        if (source_did, turn['speaker'], turn['utterance']) != (
                original['dialogue_id'], original['speaker'], original['text']):
            raise ValueError('Casting and prediction source turn differ: ' + uid)
        row = rows.get(uid)
        if row and (row['dataset'], row['dialogue_id'], row['task'], row['condition']) != (
                dataset, source_did, 'emotion_recognition', 'text'):
            raise ValueError('Prediction identity mismatch: ' + uid)
        if row and model == 'voxtral' and (row.get('model') != cfg['model'] or row.get('naming') != naming):
            raise ValueError('Voxtral prediction checkpoint/naming mismatch: '+uid)
        if row:
            turns = source[source_did]
            names = sorted({t['speaker'] for t in turns} |
                           {l['name'] for t in turns for l in t['listener']})
            mapping = {name: (name if naming == 'original' else f'P{k+1:02d}')
                       for k, name in enumerate(names)}
            start = row['context_start']
            end = row['context_end']
            if end != original['turn_index'] or not 0 <= start <= end < len(turns):
                raise ValueError('Prediction context indices mismatch: ' + uid)
            sample = {**row, 'naming': naming,
                      'history': [{'index': k, 'speaker': mapping[turns[k]['speaker']],
                                   'utterance': turns[k]['utterance']}
                                  for k in range(start, end+1)]}
            if (row['prompt'] != prompt(sample) or row['candidate_mapping'] != mapping
                    or row['candidates'] != list(mapping.values())):
                skipped.append({'utterance_id': uid, 'dialogue_id': source_did,
                                'reason': 'source_context_mismatch',
                                'status': row['status']})
                continue
        if not row or row['status'] != 'ok' or row['prediction'] not in STRENGTHS:
            skipped.append({'utterance_id': uid, 'dialogue_id': original['dialogue_id'],
                            'reason': 'missing_or_invalid_emotion_prediction',
                            'status': row.get('status') if row else 'missing',
                            'raw_output': row.get('raw_output') if row else None,
                            'error': row.get('error') if row else None})
            continue
        label = row['prediction']
        vector = [0.0] * len(base['versions']['emotion_vector_order'])
        vector[base['versions']['emotion_vector_order'].index(label)] = STRENGTHS[label]
        jobs.append({**original, 'emotion': label, 'emotion_vector': vector,
                     'emo_alpha': 1.0, 'emotion_model': model,
                     'prediction_sample_id': row['sample_id']})
    if not jobs:
        raise ValueError('No valid predicted utterances')
    return {**base, 'condition': 'predicted_audio', 'jobs': jobs,
            'blocked': skipped if invalid_policy == 'error' else [],
            'skipped': skipped, 'invalid_policy': invalid_policy,
            'emotion_strengths': STRENGTHS, 'emotion_model': model,
            'prediction_run': {'source': str(predictions), 'sha256': digest(predictions),
                               'config': cfg, 'config_sha256': digest(cfg_path),
                               'requested_naming': naming},
            'casting_plan_sha256': digest(base_path),
            'observed_prediction_source_sha256': observed_source_sha256,
            'prediction_source_hash_matches': observed_source_sha256 == cfg['data_sha256'],
            'source_verification': 'exact full prompt, candidates, mapping and target turn for every included prediction',
            'prompt_implementation_sha256': digest(VOICE/'task/tasks.py'),
            'source_prediction_scope': 'full source dataset; retained reviewed casting turns',
            'neutral_policy': 'explicit neutral vector at 0.6; emo_alpha=1.0',
            'counts': {'casting_turns': len(base['jobs']), 'valid': len(jobs),
                       'skipped': len(skipped), 'labels': dict(Counter(j['emotion'] for j in jobs))}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['mpdd', 'bazinga'], required=True)
    p.add_argument('--model', choices=list(MODELS), required=True)
    p.add_argument('--naming', choices=['anonymous', 'original'], default='original')
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--invalid-policy', choices=['skip', 'error'], default='error')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    plan = prepare(args.dataset, args.model, args.naming, args.predictions, args.invalid_policy)
    output = args.output or VOICE/'audio/emotion'/(
        'MPDD' if args.dataset == 'mpdd' else 'bazinga')/args.model/'plans'/(
        args.naming + '.json')
    write(output, plan)
    write(output.with_suffix('.skipped.json'), plan['skipped'])
    print(json.dumps({'output': str(output), **plan['counts']}, ensure_ascii=False))
    if plan['blocked']:
        raise SystemExit('Invalid predictions recorded; --invalid-policy error blocks synthesis')


if __name__ == '__main__':
    main()
