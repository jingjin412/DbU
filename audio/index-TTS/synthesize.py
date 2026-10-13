#!/usr/bin/env python3
"""GPU runner: only execute through a user-submitted Slurm job."""
from emotion_settings import validate_jobs
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from cache_lock import cache_entry_lock

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10,help='pilot default 10; 0 for all')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--cache-namespace',help='Optional fresh cache scope for throughput benchmarks')
    args=p.parse_args()
    if args.limit<0:raise ValueError('limit must be nonnegative')
    started=time.monotonic()
    plan=json.loads(args.jobs.read_text());jobs=plan['jobs'][:args.limit] if args.limit else plan['jobs']
    validate_jobs(plan['jobs'], plan['versions']['emotion_vector_order'])
    if plan['blocked']:raise ValueError(f'{len(plan["blocked"])} blocked utterances; resolve before synthesis, do not silently shrink paired data')
    if not jobs:raise ValueError('No jobs')
    versions=json.loads((ROOT/'versions.json').read_text())
    if plan['versions']!=versions:raise ValueError('Plan/runtime versions differ')
    for job in jobs:
        if digest(Path(job['reference_wav']))!=job['reference_sha256']:raise ValueError('Reference checksum changed')
    if args.validate_only:
        print(f'Validated {len(jobs)} planned utterances; no GPU used');return
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Submit run.sbatch; do not synthesize on a login node')
    # Allow normal Slurm termination to run the owned-cache-lock finally block.
    signal.signal(signal.SIGTERM,signal.default_int_handler)
    import numpy as np
    import soundfile as sf
    import torch
    from opencc import OpenCC
    source=ROOT/'vendor/index-tts'
    sys.path.insert(0,str(source))
    from indextts.infer_v2 import IndexTTS2
    if not torch.cuda.is_available():raise RuntimeError('Slurm job has no CUDA GPU')
    weights=ROOT/'checkpoints'
    model_started=time.monotonic()
    tts=IndexTTS2(cfg_path=str(weights/'config.yaml'),model_dir=str(weights),device='cuda',
                  use_fp16=True,use_cuda_kernel=False,use_deepspeed=False)
    model_init_seconds=time.monotonic()-model_started
    convert=OpenCC('t2s')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # A shared content-addressed cache prevents duplicate synthesis across model emotion predictions.
    cache=ROOT/'generated_cache';cache.mkdir(exist_ok=True)
    output=args.output_dir/'manifest.jsonl';audio_map={}
    config_path=args.output_dir/'config.json'
    config={'jobs_sha256':digest(args.jobs),'limit':args.limit,'versions':versions,
            'runner_sha256':digest(Path(__file__)),'runtime_lock_sha256':digest(ROOT/'runtime_lock.json'),
            'runtime_packages_sha256':digest(ROOT/'runtime_packages.txt'),'torch':torch.__version__,
            'cache_namespace':args.cache_namespace}
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Output config changed; use a fresh directory')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    records=[]
    output.write_text('')
    for j,job in enumerate(jobs,1):
        sample_started=time.monotonic();inference_seconds=0.0
        text=convert.convert(job['text']) if job['text_normalization']=='opencc_t2s' else job['text']
        signature={k:job[k] for k in ['reference_sha256','emotion_vector','emo_alpha','seed','use_random','text_normalization']}
        signature.update(text=text,versions=versions,runner_sha256=config['runner_sha256'],torch=torch.__version__,runtime_lock_sha256=config['runtime_lock_sha256'],
                         runtime_packages_sha256=config['runtime_packages_sha256'])
        if args.cache_namespace is not None:signature['cache_namespace']=args.cache_namespace
        key=hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        path=cache/(key+'.wav');lock=cache/(key+'.lock')
        # Another model/job may be generating the same content-addressed entry.
        # Wait for its atomic WAV publication rather than failing the whole run.
        with cache_entry_lock(path,lock) as owns_lock:
            cache_hit=not owns_lock
            if owns_lock:
                random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
                tmp=cache/(key+f'.{os.environ["SLURM_JOB_ID"]}.part.wav')
                infer_started=time.monotonic()
                tts.infer(spk_audio_prompt=job['reference_wav'],text=text,output_path=str(tmp),
                          emo_vector=job['emotion_vector'],emo_alpha=job['emo_alpha'],use_random=False,
                          interval_silence=180,verbose=False)
                inference_seconds=time.monotonic()-infer_started
                info=sf.info(tmp)
                if info.frames<=0:raise RuntimeError('Empty generated audio')
                tmp.replace(path)
        info=sf.info(path)
        if info.frames<=0:raise RuntimeError('Invalid cached audio')
        record={**job,'synthesis_text':text,'audio_path':str(path.resolve()),'audio_sha256':digest(path),
                'generation_key':key,'duration_seconds':info.duration,'cache_hit':cache_hit,
                'inference_seconds':inference_seconds,'sample_wall_seconds':time.monotonic()-sample_started,
                'qc_status':'pending_asr_similarity_emotion_and_listening'}
        records.append(record);audio_map[job['utterance_id']]=str(path.resolve())
        # Append each completed utterance; avoid quadratic rewrites on full-corpus runs.
        with output.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        if j%25==0 or j==len(jobs):
            (args.output_dir/'audio_manifest.json').write_text(json.dumps(audio_map,ensure_ascii=False,indent=2)+'\n')
        print(f'{j}/{len(jobs)} {job["utterance_id"]}',flush=True)
    metrics={'jobs':len(records),'wall_seconds':time.monotonic()-started,
             'model_init_seconds':model_init_seconds,'cache_hits':sum(r['cache_hit'] for r in records),
             'generated_jobs':sum(not r['cache_hit'] for r in records),
             'inference_seconds':sum(r['inference_seconds'] for r in records),
             'generated_audio_seconds':sum(r['duration_seconds'] for r in records if not r['cache_hit']),
             'gpu_name':torch.cuda.get_device_name(),
             'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')


if __name__=='__main__':main()
