#!/usr/bin/env python3
"""Independent IndexTTS processes on one Slurm GPU; coordinator never loads models."""
from emotion_settings import validate_jobs
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SIGNATURE_FIELDS=['reference_sha256','emotion_vector','emo_alpha','seed','use_random','text_normalization']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def normalized_text(job,converter):
    return converter.convert(job['text']) if job['text_normalization']=='opencc_t2s' else job['text']


def logical_key(job,converter):
    fields={k:job[k] for k in SIGNATURE_FIELDS}
    fields['text']=normalized_text(job,converter)
    return hashlib.sha256(json.dumps(fields,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def split_jobs(jobs,workers,converter):
    """Deduplicate synthesis inputs, then balance reference-group chunks."""
    unique={};aliases=defaultdict(list)
    for job in jobs:
        key=logical_key(job,converter)
        unique.setdefault(key,job);aliases[key].append(job)
    count=min(workers,len(unique));target=math.ceil(len(unique)/count)
    by_ref=defaultdict(list)
    for job in unique.values():by_ref[job['reference_sha256']].append(job)
    chunks=[]
    for group in by_ref.values():
        chunks.extend(group[start:start+target] for start in range(0,len(group),target))
    shards=[[] for _ in range(count)]
    for chunk in sorted(chunks,key=len,reverse=True):
        dest=min(range(count),key=lambda i:(len(shards[i]),i))
        shards[dest].extend(chunk)
    return [shard for shard in shards if shard],aliases


def merge_records(jobs,shards,aliases,converter):
    representative={}
    for inputs,records in shards:
        expected={j['utterance_id'] for j in inputs}
        actual=[r['utterance_id'] for r in records]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:
            raise ValueError('Worker output coverage mismatch')
        for record in records:
            key=logical_key(record,converter)
            if key not in aliases or key in representative:raise ValueError('Unexpected/duplicate synthesis output')
            if not Path(record['audio_path']).is_file():raise FileNotFoundError(record['audio_path'])
            if digest(record['audio_path'])!=record['audio_sha256']:raise ValueError('Generated audio checksum mismatch')
            if record['synthesis_text']!=normalized_text(record,converter):raise ValueError('Worker synthesis text mismatch')
            representative[key]=record
    if set(representative)!=set(aliases):raise ValueError('Incomplete synthesis coverage')
    # Restore all source IDs and original order, including identical-input aliases.
    return [{**representative[logical_key(job,converter)],**job,
             'shared_synthesis_alias':len(aliases[logical_key(job,converter)])>1} for job in jobs]


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cache-namespace',help='Use distinct fresh names for workers=1/2/4 benchmarks')
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    if args.workers<1 or args.limit<0:raise ValueError('workers must be positive; limit must be nonnegative')
    plan=json.loads(args.jobs.read_text());jobs=plan['jobs'][:args.limit] if args.limit else plan['jobs']
    validate_jobs(plan['jobs'], plan['versions']['emotion_vector_order'])
    if plan['blocked']:raise ValueError('Resolve blocked jobs before synthesis')
    if not jobs:raise ValueError('No jobs')
    if len({j['utterance_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate input utterance IDs')
    if plan['versions']!=json.loads((ROOT/'versions.json').read_text()):raise ValueError('Plan/runtime versions differ')
    verified=set()
    for job in jobs:
        key=(job['reference_wav'],job['reference_sha256'])
        if key not in verified:
            if digest(job['reference_wav'])!=job['reference_sha256']:raise ValueError('Reference checksum changed')
            verified.add(key)
    # OpenCC is text-only; no torch/model import in the coordinator or validation path.
    from opencc import OpenCC
    converter=OpenCC('t2s')
    shards,aliases=split_jobs(jobs,args.workers,converter)
    summary={'source_jobs':len(jobs),'unique_synthesis_inputs':len(aliases),
             'requested_workers':args.workers,'active_workers':len(shards),'worker_jobs':[len(s) for s in shards],
             'cache_namespace':args.cache_namespace}
    if args.validate_only:
        print(json.dumps({**summary,'validated':True,'gpu_used':False},ensure_ascii=False));return
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Submit run.sbatch; do not synthesize on a login node')
    # Children inherit the same Slurm CUDA_VISIBLE_DEVICES; all use the same allocated GPU.
    cpus=int(os.environ.get('SLURM_CPUS_PER_TASK',os.environ.get('OMP_NUM_THREADS','8')))
    threads=max(1,cpus//len(shards))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config={'jobs_sha256':digest(args.jobs),'runner_sha256':digest(ROOT/'synthesize.py'),
            'coordinator_sha256':digest(__file__),'limit':args.limit,'threads_per_worker':threads,
            'runtime_lock_sha256':digest(ROOT/'runtime_lock.json'),**summary}
    cfg=args.output_dir/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Use a fresh output directory')
    write(cfg,config)
    print(json.dumps(config,ensure_ascii=False),flush=True)
    processes=[];handles=[];folders=[]
    previous_sigterm=signal.signal(signal.SIGTERM,signal.default_int_handler)
    try:
        for index,shard in enumerate(shards):
            folder=args.output_dir/'workers'/f'w{index:02d}';folder.mkdir(parents=True,exist_ok=True)
            shard_plan={**plan,'jobs':shard,'blocked':[],
                        'reference_count':len({j['voice_character_id'] for j in shard})}
            input_path=folder/'jobs.json';write(input_path,shard_plan)
            command=[sys.executable,'-u',str(ROOT/'synthesize.py'),'--jobs',str(input_path.resolve()),
                     '--output-dir',str(folder.resolve()),'--limit','0']
            if args.cache_namespace is not None:command.extend(['--cache-namespace',args.cache_namespace])
            env=os.environ.copy()
            for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[name]=str(threads)
            log=(folder/'worker.log').open('w');handles.append(log);folders.append(folder)
            processes.append(subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
            print(f'Worker {index}: {len(shard)} jobs, {threads} CPU threads; log={folder / "worker.log"}',flush=True)
        while True:
            codes=[proc.poll() for proc in processes]
            failed=[i for i,code in enumerate(codes) if code not in (None,0)]
            if failed:
                index=failed[0]
                print((folders[index]/'worker.log').read_text()[-6000:],flush=True)
                raise RuntimeError(f'Worker {index} failed ({codes[index]}); see its worker.log')
            if all(code==0 for code in codes):break
            time.sleep(1)
    except BaseException:
        for proc in processes:
            if proc.poll() is None:proc.terminate()
        for proc in processes:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raise
    finally:
        for handle in handles:handle.close()
        signal.signal(signal.SIGTERM,previous_sigterm)
    outputs=[];metrics=[]
    for index,(folder,inputs) in enumerate(zip(folders,shards)):
        records=[json.loads(line) for line in (folder/'manifest.jsonl').read_text().splitlines() if line.strip()]
        outputs.append((inputs,records));metrics.append({'worker':index,**json.loads((folder/'metrics.json').read_text())})
    records=merge_records(jobs,outputs,aliases,converter)
    (args.output_dir/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write(args.output_dir/'audio_manifest.json',{r['utterance_id']:r['audio_path'] for r in records})
    wall=time.monotonic()-started
    generated=sum(m['generated_jobs'] for m in metrics)
    performance={**summary,'wall_seconds':wall,'unique_cache_hits':sum(m['cache_hits'] for m in metrics),
                 'newly_generated_inputs':generated,'new_inputs_per_total_wall_second':generated/wall,
                 'total_wall_includes_model_startup':True,'workers':metrics,
                 'gpu_peak_note':'Per-worker peaks are separate measurements; their sum is not a simultaneous device peak.'}
    write(args.output_dir/'performance.json',performance)
    print(f'Completed {len(records)} source jobs ({len(aliases)} unique inputs) in {wall:.2f}s; {generated} newly generated.',flush=True)
    print('Performance report: '+str(args.output_dir/'performance.json'),flush=True)


if __name__=='__main__':main()
