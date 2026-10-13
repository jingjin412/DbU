"""Build paired samples with explicit cutoffs; labels never enter prompt history."""
import json
from pathlib import Path
from tasks import LABELS


def samples(args, task):
    dialogues = json.loads(args.data.read_text())
    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    gold_data = json.loads(args.gold_emotions.read_text()) if args.gold_emotions else {}
    result, selected = [], []
    for dialogue_id, turns in dialogues.items():
        names = sorted({t['speaker'] for t in turns} | {l['name'] for t in turns for l in t['listener']})
        if len(names) < args.min_participants:
            continue
        if args.max_dialogues and len(selected) >= args.max_dialogues:
            break
        selected.append(dialogue_id)
        naming = getattr(args, 'naming', 'anonymous')
        mapping = {name: (f'P{i+1:02d}' if naming == 'anonymous' else name) for i, name in enumerate(names)}
        candidates = list(mapping.values())
        for i, turn in enumerate(turns):
            if task == 'next_speaker_ranking' and i == len(turns)-1:
                continue
            start = max(0, i+1-args.context_turns) if args.context_turns else 0
            history = []
            for j in range(start, i+1):
                sm = metadata.get(dialogue_id, {}).get('utterances', [])
                uid = sm[j]['utterance_id'] if sm else f'{args.dataset}:{dialogue_id}:u{j+1:05d}'
                history.append({'index': j, 'speaker': mapping[turns[j]['speaker']], 'utterance': turns[j]['utterance'], 'utterance_id': uid})
            if task == 'addressee_recognition':
                gold = list(dict.fromkeys(mapping[l['name']] for l in turn['listener']))
                if not gold:
                    raise ValueError(f'Empty addressee: {dialogue_id}:{i}; clean input first')
            elif task == 'next_speaker_ranking':
                gold = mapping[turns[i+1]['speaker']]
            else:
                gold = turn.get('emotion')
                if dialogue_id in gold_data:
                    external = gold_data[dialogue_id][i]
                    if (external['speaker'], external['utterance']) != (turn['speaker'], turn['utterance']):
                        raise ValueError('Gold emotion data does not align')
                    gold = external.get('emotion')
                if gold is not None and gold not in LABELS:
                    raise ValueError(f'Unknown gold emotion: {gold}')
            result.append({'sample_id': f'{args.dataset}:{dialogue_id}:{i}:{task}', 'dataset': args.dataset,
                           'dialogue_id': dialogue_id, 'task': task, 'naming': naming, 'context_start': start, 'context_end': i,
                           'target_utterance_id': history[-1]['utterance_id'], 'history': history,
                           'candidates': candidates, 'candidate_mapping': mapping, 'gold': gold})
            if args.max_samples and len(result) >= args.max_samples:
                return result
    return result
