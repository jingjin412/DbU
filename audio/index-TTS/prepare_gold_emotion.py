#!/usr/bin/env python3
"""Build MPDD gold-emotion TTS using frozen neutral casting (CPU only)."""
import argparse
from collections import Counter
import json
from pathlib import Path

from prepare_model_emotion import STRENGTHS, ROOT, VOICE, digest, write
from emotion_settings import validate_jobs


def prepare(base_path, data_path, gold_path, versions_path):
    base = json.loads(base_path.read_text())
    validate_jobs(base['jobs'], base['versions']['emotion_vector_order'])
    data = json.loads(data_path.read_text())
    gold = json.loads(gold_path.read_text())
    if base['dataset'] != 'mpdd' or base['condition'] != 'neutral_audio' or base['blocked']:
        raise ValueError('Requires unblocked MPDD neutral casting')
    if base['data_sha256'] != digest(data_path):
        raise ValueError('Casting source data changed; rebuild casting first')
    if base['versions'] != json.loads(versions_path.read_text()):
        raise ValueError('Casting/runtime versions differ')
    order = base['versions']['emotion_vector_order']
    if len(order) != len(set(order)) or not set(STRENGTHS) <= set(order):
        raise ValueError('Invalid emotion vector order')
    jobs = []
    seen = set()
    for original in base['jobs']:
        did, index, uid = original['dialogue_id'], original['turn_index'], original['utterance_id']
        if uid in seen or uid != f'mpdd:{did}:u{index+1:05d}':
            raise ValueError('Duplicate/misaligned utterance ID: ' + uid)
        seen.add(uid)
        if len(gold[did]) != len(data[did]):
            raise ValueError('Gold/source dialogue length mismatch: ' + did)
        turn, label_turn = data[did][index], gold[did][index]
        expected = (original['speaker'], original['text'])
        if any((t['speaker'], t['utterance']) != expected for t in (turn, label_turn)):
            raise ValueError('Gold/source/casting turn mismatch: ' + uid)
        label = label_turn.get('emotion')
        if label not in STRENGTHS:
            raise ValueError('Missing/unknown gold emotion: ' + uid)
        vector = [0.0] * len(order)
        vector[order.index(label)] = STRENGTHS[label]
        jobs.append({**original, 'emotion': label, 'emotion_vector': vector,
                     'emo_alpha': 1.0, 'seed': 42, 'use_random': False,
                     'emotion_source': 'mpdd_original_annotation'})
    if not jobs:
        raise ValueError('No casting jobs')
    return {**base, 'condition': 'gold_audio', 'jobs': jobs, 'prediction_run': None,
            'emotion_source': 'mpdd_original_annotation',
            'gold_emotions': {'source': str(gold_path.resolve()), 'sha256': digest(gold_path),
                              'verification': 'exact dialogue length, speaker and text at each retained turn'},
            'casting_plan_sha256': digest(base_path), 'emotion_strengths': STRENGTHS,
            'neutral_policy': 'explicit neutral vector at 0.6; emo_alpha=1.0',
            'counts': {'casting_turns': len(jobs), 'valid': len(jobs), 'skipped': 0,
                       'labels': dict(Counter(j['emotion'] for j in jobs))}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=VOICE/'audio/emotion/MPDD/gold/plans/original.json')
    args = p.parse_args()
    plan = prepare(ROOT/'plans/mpdd_neutral.json', VOICE/'data/mpdd/dialogue.json',
                   VOICE/'data/mpdd/dialogue_chinese.json', ROOT/'versions.json')
    write(args.output, plan)
    print(json.dumps({'output': str(args.output), **plan['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
