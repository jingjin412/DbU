#!/usr/bin/env python3
"""Run/dry-run any DbU task; supports resumable, auditable JSONL output."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from collections import Counter
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import threading
import time

from data import samples
from model_io import Client, VOXTRAL_AUDIO_LAYOUT, FLAMINGO_AUDIO_LAYOUT, PHI4_AUDIO_LAYOUT, audio_groups
from tasks import TASKS, VERSION, RANKING_VERSION, PARSER_VERSION, PLAIN_PROMPT_VERSION, PLAIN_PARSER_VERSION, parse, metrics, prompt

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path else None


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def duration(seconds):
    if seconds is None:
        return '--'
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def main(default_task=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=(*TASKS, 'all', 'downstream'), default=default_task or 'all')
    p.add_argument('--dataset', choices=('both', 'bazinga', 'mpdd'), default='both')
    p.add_argument('--data', type=Path, help='Custom input; requires a single --dataset')
    p.add_argument('--metadata', type=Path, help='Optional utterance provenance; auto-detected beside data')
    p.add_argument('--gold-emotions', type=Path, help='Optional aligned original MPDD JSON; scoring only')
    emotions = p.add_mutually_exclusive_group()
    emotions.add_argument('--history-emotions', type=Path, help='Saved text emotion predictions to provide for each visible turn')
    emotions.add_argument('--history-emotion-gold', type=Path, help='Aligned gold emotion JSON to provide for each visible turn')
    p.add_argument('--naming', choices=('anonymous', 'original'), default='anonymous',
                   help='anonymous: speaker/candidate P IDs; original: exact source character names')
    p.add_argument('--min-participants', type=int, default=3)
    p.add_argument('--context-turns', type=int, default=0, help='0 = all history within dialogue')
    p.add_argument('--max-dialogues', type=int, default=0)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--model-kind', choices=('gemma4-e4b', 'qwen3-instruct', 'voxtral-small-24b', 'kimi-audio-7b-instruct', 'audio-flamingo-3', 'phi-4-multimodal-instruct', 'openai-compatible'), default='gemma4-e4b')
    p.add_argument('--backend', choices=('http', 'local'), default='http')
    p.add_argument('--output-format', choices=('unconstrained', 'json-schema', 'plain'), default='unconstrained',
                   help='JSON prompt/parser, constrained JSON, or one-line plain prompt/parser')
    p.add_argument('--max-model-len', type=int, default=0, help='Local backend context limit; 0 uses model default')
    p.add_argument('--runtime-metadata', type=Path, help='Slurm server runtime/provenance JSON')
    p.add_argument('--model', help='Checkpoint path recorded and checked against server root')
    p.add_argument('--served-model', default='voice-model')
    p.add_argument('--endpoint', default='http://127.0.0.1:8901/v1')
    p.add_argument('--condition', choices=('text', 'neutral_audio', 'predicted_audio', 'gold_audio', 'real_audio'), default='text')
    p.add_argument('--sample-selection', type=Path, help='Frozen per-task sample IDs for matched audio comparisons')
    p.add_argument('--audio-manifest', type=Path)
    p.add_argument('--audio-scope', choices=('history', 'current'), default='history')
    p.add_argument('--audio-layout', choices=('utterance', 'dialogue_prefix'), default='utterance')
    p.add_argument('--max-audio-clips', type=int, default=64)
    p.add_argument('--concurrency', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--output-dir', type=Path, default=ROOT.parent/'results')
    p.add_argument('--run-id', default=os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--preview-manifest', type=Path, help='Write selected preview paths during dry-run')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--fail-on-all-invalid', action='store_true', help='Stop local inference if the first up to 10 samples have no valid output')
    p.add_argument('--fail-fast-requests', action='store_true', help='Stop and cancel queued requests on the first HTTP/transport error')
    p.add_argument('--smoke-audio-coverage', action='store_true', help='Small smoke covers initial, >5-turn and longest visible history')
    args = p.parse_args()
    if args.data is not None and not args.data.is_file():
        p.error(f'--data must point to an existing file (received {str(args.data)!r}); check unset shell variables')
    local_client = None
    if args.backend == 'local':
        if args.output_format != 'unconstrained':
            p.error('--output-format plain/json-schema requires --backend http')
        if args.model_kind not in ('voxtral-small-24b', 'kimi-audio-7b-instruct') or args.condition != 'text':
            p.error('Local backend supports only Voxtral/Kimi text condition')
        if args.concurrency != 1 or not args.model or args.temperature != 0:
            p.error('Local backend requires --concurrency 1, --model and --temperature 0')
        if args.max_model_len < 0 or args.max_new_tokens > 4096:
            p.error('Local backend requires nonnegative context and at most 4096 output tokens')
        from local_models import LocalClient
        local_client = LocalClient(args)
    if min(args.context_turns, args.max_dialogues, args.max_samples) < 0 or min(args.min_participants, args.concurrency, args.max_new_tokens, args.max_audio_clips) < 1:
        p.error('Invalid non-positive option')
    if args.condition != 'text' and not args.audio_manifest:
        p.error('Audio conditions require --audio-manifest')
    if (args.history_emotions or args.history_emotion_gold) and (args.condition not in ('text', 'neutral_audio') or args.task not in ('downstream', 'addressee_recognition', 'next_speaker_ranking')):
        p.error('History emotions require text or neutral_audio downstream tasks')
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        p.error('--run-id must be a single folder name')
    if args.dataset == 'both' and any((args.data, args.metadata, args.gold_emotions, args.audio_manifest, args.history_emotions, args.history_emotion_gold)):
        p.error('Custom data/metadata/gold/audio manifest requires --dataset bazinga or mpdd')
    datasets = ('bazinga', 'mpdd') if args.dataset == 'both' else (args.dataset,)
    preview_paths = []
    for dataset in datasets:
        current = argparse.Namespace(**vars(args))
        current.dataset = dataset
        if current.data is None:
            current.data = ROOT.parent/('data/bazinga/clean/dialogue.json' if dataset == 'bazinga' else 'data/mpdd/dialogue.json')
            if dataset == 'mpdd' and current.gold_emotions is None:
                current.gold_emotions = ROOT.parent/'data/mpdd/dialogue_chinese.json'
        run_dataset(current, local_client)
        if args.dry_run:
            selected_tasks = TASKS if current.task == 'all' else (('addressee_recognition', 'next_speaker_ranking') if current.task == 'downstream' else (current.task,))
            preview_paths.extend(str((current.output_dir/task/current.dataset/current.model_kind/current.naming/current.run_id/'preview.json').resolve())
                                 for task in selected_tasks)
    if args.dry_run and args.preview_manifest:
        args.preview_manifest.parent.mkdir(parents=True, exist_ok=True)
        write(args.preview_manifest, {'previews': preview_paths})


def run_dataset(args, local_client=None):
    if not args.metadata:
        candidate = args.data.with_name('metadata.json')
        # MPDD metadata is story metadata, not utterance provenance.
        if candidate.exists() and args.dataset == 'bazinga':
            args.metadata = candidate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = TASKS if args.task == 'all' else (('addressee_recognition', 'next_speaker_ranking') if args.task == 'downstream' else (args.task,))
    client = local_client or Client(args)
    client.args = args
    from emotion_context import load, attach, VERSION as EMOTION_CONTEXT_VERSION
    emotion_entries = load(args)
    output_format = getattr(args, 'output_format', 'unconstrained')
    server = None if args.dry_run else client.check_server()
    for task in tasks:
        builder = argparse.Namespace(**vars(args))
        selection = None
        if getattr(args, 'sample_selection', None):
            selection = json.loads(args.sample_selection.read_text())
            if selection['data_sha256'] != sha(args.data) or selection['metadata_sha256'] != sha(args.metadata):
                raise ValueError('Sample selection dataset/metadata checksum mismatch')
            if (args.context_turns, args.naming, args.audio_scope) != (selection['context_turns'], selection['naming'], selection['audio_scope']):
                raise ValueError('Sample selection history/naming/scope differs')
            if selection.get('audio_layout', 'utterance') != getattr(args, 'audio_layout', 'utterance'):
                raise ValueError('Sample selection audio layout differs')
            builder.max_samples = 0
        if args.smoke_audio_coverage:
            if args.max_samples < 1 or args.condition == 'text':
                raise ValueError('Audio coverage smoke requires audio and --max-samples > 0')
            builder.max_samples = 0
        ss = samples(builder, task)
        if selection is not None:
            allowed = set(selection['samples'][task])
            if len(allowed) != len(selection['samples'][task]) or not allowed <= {s['sample_id'] for s in ss}:
                raise ValueError('Duplicate/unknown selected sample IDs')
            ss = [s for s in ss if s['sample_id'] in allowed]
            if not args.smoke_audio_coverage and args.max_samples:
                ss = ss[:args.max_samples]
        if args.smoke_audio_coverage and ss:
            indices = [0]
            over_five = next((index for index, sample in enumerate(ss) if len(sample['history']) > 5), None)
            if over_five is not None: indices.append(over_five)
            if getattr(args, 'audio_layout', 'utterance') == 'dialogue_prefix':
                def audio_seconds(index):
                    return sum((end-start)/client.audio_info[path].samplerate for _,path,start,end in client.clips(ss[index]))
                indices.append(max(range(len(ss)), key=audio_seconds))
            else:
                indices.append(max(range(len(ss)), key=lambda index: len(ss[index]['history'])))
            indices = list(dict.fromkeys(indices))
            indices.extend(index for index in range(len(ss)) if index not in indices)
            ss = [ss[index] for index in indices[:args.max_samples]]
        if not ss:
            raise ValueError('No samples selected')
        emotion_audit = attach(ss, emotion_entries) if emotion_entries is not None else None
        for s in ss:
            s['audio_usage'] = client.audio_usage(s)
            s['audio_fallback_count'] = sum(e.get('fallback', False) for e in s['audio_usage'])
            s['audio_fallback'] = s['audio_fallback_count'] > 0
            client.clips(s)  # Preflight all audio, without silently shrinking the paired sample set.
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                  if k not in {'resume', 'dry_run', 'preview_manifest', 'output_dir', 'endpoint', 'concurrency', 'timeout', 'task'}}
        config.update(task=task, parser_version=PARSER_VERSION, prompt_version=RANKING_VERSION if task == "next_speaker_ranking" else VERSION, data_sha256=sha(args.data), metadata_sha256=sha(args.metadata),
                      sample_selection_sha256=sha(getattr(args, 'sample_selection', None)),
                      gold_emotions_sha256=sha(args.gold_emotions), audio_manifest_sha256=sha(args.audio_manifest),
                      implementation_sha256={name: sha(ROOT/name) for name in ('run.py', 'data.py', 'tasks.py', 'model_io.py')},
                      candidate_policy=f'full_dialogue_speaker_union_listener_{args.naming}; includes future participant membership',
                      next_speaker_policy='speaker_of_next_transcript_utterance_including_same_speaker')
        if output_format == 'plain':
            config.update(parser_version=PLAIN_PARSER_VERSION, prompt_version=PLAIN_PROMPT_VERSION)
        if emotion_audit is not None:
            config.update(history_emotion_version=EMOTION_CONTEXT_VERSION,
                          history_emotion_source='predicted' if getattr(args, 'history_emotions', None) else 'gold',
                          history_emotion_source_sha256=sha(getattr(args, 'history_emotions', None) or args.history_emotion_gold),
                          history_emotion_invalid_policy='explicit unavailable; no neutral substitution or sample exclusion',
                          history_emotion_audit=emotion_audit,
                          base_prompt_version=config['prompt_version'],
                          prompt_version=config['prompt_version']+'+'+EMOTION_CONTEXT_VERSION)
            config['implementation_sha256']['emotion_context.py'] = sha(ROOT/'emotion_context.py')
        if getattr(args, 'audio_layout', 'utterance') == 'dialogue_prefix':
            from dialogue_audio import LAYOUT
            config['dialogue_audio_transport'] = LAYOUT
            config['implementation_sha256']['dialogue_audio.py'] = sha(ROOT/'dialogue_audio.py')
        if args.runtime_metadata:
            config['runtime_metadata'] = json.loads(args.runtime_metadata.read_text())
            config['runtime_metadata_sha256'] = sha(args.runtime_metadata)
        if args.model_kind == 'voxtral-small-24b' and args.condition != 'text':
            config['audio_layout'] = VOXTRAL_AUDIO_LAYOUT
        if args.model_kind == 'audio-flamingo-3' and args.condition != 'text':
            config['audio_layout'] = FLAMINGO_AUDIO_LAYOUT
        if args.model_kind == 'phi-4-multimodal-instruct' and args.condition != 'text':
            config['audio_layout'] = PHI4_AUDIO_LAYOUT
        if getattr(args, 'audio_layout', 'utterance') == 'dialogue_prefix':
            config['audio_layout'] = LAYOUT
        if args.backend == 'local':
            config['implementation_sha256']['local_models.py'] = sha(ROOT/'local_models.py')
            config['checkpoint_manifest_sha256'] = sha(Path(args.model)/'voice_download.json')
            config['runtime_packages_sha256'] = sha(Path(sys.prefix)/'voice_packages.txt') if (Path(sys.prefix)/'voice_packages.txt').exists() else None
            if not args.dry_run:
                config['local_runtime'] = server
        seed_rows = []
        folder = args.output_dir/task/args.dataset/args.model_kind/args.naming/args.run_id
        folder.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            write(folder/'preview.json', {'config': config, 'sample_count': len(ss),
                  'dialogue_count': len({s['dialogue_id'] for s in ss}),
                  'samples': [{**s, 'prompt': prompt(s, output_format)} for s in ss[:3]]})
            print(f'{args.dataset}/{task}: {len(ss)} samples, {len({s["dialogue_id"] for s in ss})} dialogues; dry-run', flush=True)
            continue
        conf_path, results_path = folder/'config.json', folder/'predictions.jsonl'
        if conf_path.exists():
            if not args.resume or json.loads(conf_path.read_text()) != config:
                raise ValueError(f'Existing run/config mismatch: {folder}; use fresh output or --resume with identical settings')
        elif results_path.exists():
            raise ValueError('Results without config; refusing overwrite')
        else:
            write(conf_path, config)
        write(folder/'server.json', server)
        rows = []
        if results_path.exists():
            for line in results_path.read_text().splitlines():
                rows.append(json.loads(line))
        done = {r['sample_id'] for r in rows}
        if len(done) != len(rows) or not done <= {s['sample_id'] for s in ss}:
            raise ValueError('Resume contains duplicate/unknown sample IDs')
        if seed_rows and not rows:
            with results_path.open('a', encoding='utf-8') as output:
                for row in seed_rows:
                    output.write(json.dumps(row, ensure_ascii=False)+'\n')
            rows = seed_rows
            done = {row['sample_id'] for row in rows}
            print(f'[REUSE] {args.dataset}/{task}: {len(rows)} previous responses retained; {len(ss)-len(rows)} samples remain', flush=True)
        started = time.monotonic()
        initial_done = len(rows)
        status_counts = Counter(row['status'] for row in rows)
        if args.backend == 'local':
            print(f'[START] {args.dataset}/{task} {len(rows)}/{len(ss)} | naming={args.naming} | condition={args.condition}', flush=True)
        def infer(s):
            row = {k: v for k, v in s.items() if k != 'history'}
            row.update(condition=args.condition, model=args.model or args.served_model, prompt=prompt(s, output_format), prediction=None)
            stop = threading.Event()
            heartbeat = None
            if args.backend == 'local':
                request_started = time.monotonic()
                def report_wait():
                    while not stop.wait(30):
                        print(f'[WAIT] {args.dataset}/{task} sample {len(rows)+1}/{len(ss)} '
                              f'| speaker={s["history"][-1]["speaker"]} '
                              f'| current sample elapsed={duration(time.monotonic()-request_started)}', flush=True)
                heartbeat = threading.Thread(target=report_wait, daemon=True)
                heartbeat.start()
            try:
                response = client.complete(s)
                row['raw_response'] = response
                if args.model_kind == 'voxtral-small-24b' and args.condition != 'text':
                    source_clips = client.clips(s)
                    row['audio_transport'] = {'layout': VOXTRAL_AUDIO_LAYOUT, 'source_clip_count': len(source_clips),
                                              'api_audio_chunks': len(audio_groups(source_clips))}
                if args.model_kind == 'audio-flamingo-3' and args.condition != 'text':
                    row['audio_transport'] = {'layout': FLAMINGO_AUDIO_LAYOUT,
                                              'source_clip_count': len(client.clips(s)), 'api_audio_chunks': 1}
                if args.model_kind == 'phi-4-multimodal-instruct' and args.condition != 'text':
                    clip_count = len(client.clips(s))
                    row['audio_transport'] = {'layout': PHI4_AUDIO_LAYOUT,
                                              'source_clip_count': clip_count, 'api_audio_chunks': clip_count}
                choice = response['choices'][0]
                text = choice['message'].get('content') or ''
                pred, error = parse(text, task, s['candidates'], output_format)
                if choice.get('finish_reason') != 'stop':
                    pred, error = None, 'non-stop finish reason: '+str(choice.get('finish_reason'))
                row.update(raw_output=text, prediction=pred, status='invalid' if error else 'ok', error=error)
            except Exception as error:
                row.update(status='error', error=str(error), raw_output=None)
                if args.backend == 'local':
                    import traceback
                    row['traceback'] = traceback.format_exc()
                    print(row['traceback'], file=sys.stderr, flush=True)
            finally:
                stop.set()
                if heartbeat is not None:
                    heartbeat.join()
            return row
        executor = nullcontext() if args.backend == 'local' else ThreadPoolExecutor(max_workers=args.concurrency)
        with results_path.open('a', encoding='utf-8') as output, executor as pool:
            pending = [s for s in ss if s['sample_id'] not in done]
            responses = map(infer, pending) if args.backend == 'local' else pool.map(infer, pending)
            for row in responses:
                output.write(json.dumps(row, ensure_ascii=False)+'\n'); output.flush()
                rows.append(row)
                status_counts[row['status']] += 1
                if args.backend == 'local':
                    elapsed = time.monotonic()-started
                    completed = len(rows)-initial_done
                    eta = elapsed/completed*(len(ss)-len(rows)) if completed else None
                    print(f'[PROGRESS] {args.dataset}/{task} {len(rows)}/{len(ss)} '
                          f'({100*len(rows)/len(ss):.1f}%) | elapsed={duration(elapsed)} '
                          f'| ETA~{duration(eta)} | invalid={status_counts["invalid"]} error={status_counts["error"]}', flush=True)
                if args.backend == 'local' and row['status'] == 'error':
                    write(folder/'metrics.json', metrics(rows, task))
                    raise RuntimeError(f'{task}: local inference failed; {row["error"]}; inspect predictions.jsonl')
                if args.fail_fast_requests and row['status'] == 'error':
                    write(folder/'metrics.json', metrics(rows, task))
                    if pool is not None:
                        pool.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(f'{args.dataset}/{task}: request failure; canceled queued requests: {row["error"]}')
                if args.backend == 'local' and args.fail_on_all_invalid and len(rows) == min(10, len(ss)) and status_counts['ok'] == 0:
                    write(folder/'metrics.json', metrics(rows, task))
                    raise RuntimeError(f'{args.dataset}/{task}: first {len(rows)} samples all invalid; stopped to avoid wasting GPU time. Inspect predictions.jsonl before submitting full evaluation.')
                if args.backend != 'local' and len(rows) % 50 == 0:
                    print(f'{task}: {len(rows)}/{len(ss)}', flush=True)
        task_metrics = metrics(rows, task)
        write(folder/'metrics.json', task_metrics)
        if args.backend == 'local':
            print(f'[DONE] {args.dataset}/{task} | elapsed={duration(time.monotonic()-started)} | metrics={folder/"metrics.json"}', flush=True)
            print(f'{args.dataset}/{task}', json.dumps(task_metrics, ensure_ascii=False), flush=True)
        else:
            print(task, json.dumps(task_metrics), flush=True)
        if any(r['status'] == 'error' for r in rows):
            raise RuntimeError(f'{task}: request errors recorded; inspect predictions.jsonl')


if __name__ == '__main__':
    main()
