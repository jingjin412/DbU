#!/usr/bin/env python3
"""CPU-only reference selection and story-scoped casting. No voice identity verification claimed."""
import bisect
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent
VOICE=ROOT.parents[1]
SOURCES=ROOT/'sources'
RANGES=[(1,296),(297,1385),(1386,1840),(1841,2154),(2155,3056),(3057,4188)]
SPECIAL={'all','multiple_persons','not_available','?','_',''}


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def slug(value):return hashlib.sha256(value.encode()).hexdigest()[:16]


def stats(values):
    values=np.asarray(values,dtype=np.float32)
    return {'rms':float(np.sqrt(np.mean(values**2))), 'peak':float(np.max(np.abs(values))),
            'clipped_fraction':float(np.mean(np.abs(values)>=.999)), 'quiet_fraction':float(np.mean(np.abs(values)<.005))}


def save_clip(path, values, rate):
    path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,values,rate,subtype='PCM_16')
    return {'reference_wav':str(path.relative_to(ROOT)), 'reference_sha256':hashfile(path),
            'duration_seconds':len(values)/rate,'sample_rate':rate,**stats(values)}


def mpdd():
    # Rebuild from editorial character profiles, including dialogue-scoped aliases.
    from cast_mpdd import main as cast_main
    cast_main()
    return json.loads((ROOT/'manifests/mpdd_voices.json').read_text())


def bazinga():
    episodes=json.loads((VOICE/'data/bazinga/metadata.json').read_text())
    candidates=defaultdict(list);registry=defaultdict(lambda:{'speaker_tokens':0,'listener_mentions':0})
    for episode,em in episodes.items():
        show=em['series'];rows=[]
        for line_num,line in enumerate((VOICE/'datagen/raw'/em['source_path']).read_text().splitlines(),1):
            p=line.split()
            if not p:continue
            row={'speaker':p[1],'start':float(p[2]),'end':float(p[3]),'text':p[4],'confidence':float(p[5]),'line':line_num}
            rows.append(row)
            if p[1] not in SPECIAL:registry[(show,p[1])]['speaker_tokens']+=1
            if len(p)>7:
                for person in p[7].split('@'):
                    if person not in SPECIAL:registry[(show,person)]['listener_mentions']+=1
        source=SOURCES/'bazinga'/f'{episode}.en.wav';info=sf.info(source)
        valid=[r for r in rows if 0<=r['start']<=r['end']<=info.duration and r['end']-r['start']<10]
        ordered=sorted(valid,key=lambda r:r['start']);starts=[r['start'] for r in ordered]
        groups=[]
        for r in rows:
            if (not groups or r['speaker']!=groups[-1][-1]['speaker']
                or r['start']<groups[-1][-1]['start']-.05 or r['start']-groups[-1][-1]['end']>1.5):groups.append([])
            groups[-1].append(r)
        for group in groups:
            speaker=group[0]['speaker']
            if speaker in SPECIAL:continue
            for a in range(0,len(group),max(1,len(group)//10)):
                window=[]
                for r in group[a:]:
                    if r['end']-group[a]['start']>10:break
                    window.append(r)
                if not window:continue
                start=max(0,window[0]['start']-.02);end=max(r['end'] for r in window)+.02
                if end>info.duration or end-start<.25 or end-start>10.2:continue
                if any(not 0<=r['start']<=r['end'] or r['start']<window[0]['start']-.05 for r in window):continue
                # Reject source-word overlap with any differently labelled speaker.
                nearby=ordered[bisect.bisect_left(starts,start-10):bisect.bisect_left(starts,end)]
                if any(r['speaker']!=speaker and min(end,r['end'])-max(start,r['start'])>.03 for r in nearby):continue
                confidence=float(np.mean([r['confidence'] for r in window]))
                if confidence<.3:continue
                duration=end-start
                score=abs(duration-6)+(10 if duration<3 else 0)+(1-confidence)*2
                candidates[(show,speaker)].append({'source_path':source,'episode_id':episode,'start_time':start,'end_time':end,
                    'source_line_start':window[0]['line'],'source_line_end':window[-1]['line'],'transcript':' '.join(r['text'] for r in window),
                    'alignment_confidence_mean':confidence,'score':score})
    records=[]
    for (show,speaker),counts in sorted(registry.items()):
        known=json.loads((VOICE/f'data/bazinga/{show}/characters.json').read_text())
        record={'character_id':show+':'+speaker,'dataset':'bazinga','story_id':show,'character':speaker,
                'source_labels':[speaker],'in_official_character_list':speaker in known,**counts,'review_status':'no_usable_reference'}
        for candidate in sorted(candidates[(show,speaker)],key=lambda c:(c['score'],c['episode_id'],c['start_time']))[:30]:
            info=sf.info(candidate['source_path'])
            values,rate=sf.read(candidate['source_path'],start=int(candidate['start_time']*info.samplerate),stop=int(candidate['end_time']*info.samplerate),dtype='float32')
            quality=stats(values)
            if quality['rms']<.005 or quality['clipped_fraction']>.01:continue
            path=ROOT/'references/bazinga'/show/(slug(speaker)+'.wav')
            record.update({k:(str(v.relative_to(ROOT)) if k=='source_path' else v) for k,v in candidate.items()},
                          **save_clip(path,values,rate),review_status='pending_identity_and_background_listening_review',
                          short_reference=len(values)/rate<3)
            break
        records.append(record)
    write(ROOT/'manifests/bazinga_voices.json',records)
    return records


def audition(records):
    # Keep evidence-rich MPDD audition separate from the original Bazinga page.
    from cast_mpdd import audition as mpdd_audition
    mpdd_audition([r for r in records if r['dataset']=='mpdd'])
    records=[r for r in records if r['dataset']=='bazinga']
    parts=['<!doctype html><meta charset="utf-8"><title>Reference voices</title><style>body{font-family:sans-serif;max-width:1100px;margin:30px auto}td{padding:10px;border-bottom:1px solid #ddd}audio{width:280px}</style><h1>角色 reference voice 候選</h1><p>自動選取，尚未人工確認聲音身分／背景音／配音合適度。依劇本分組；固定角色不跨劇本合併。</p>']
    for story in dict.fromkeys(r['story_id'] for r in records):
        parts.append('<h2>'+html.escape(story)+'</h2><table><tr><th>角色</th><th>音檔</th><th>狀態</th></tr>')
        for r in records:
            if r['story_id']!=story:continue
            audio='<audio controls preload="none" src="'+html.escape(r['reference_wav'],quote=True)+'"></audio>' if r.get('reference_wav') else '無可用 reference'
            parts.append('<tr><td>'+html.escape(r['character'])+'</td><td>'+audio+'</td><td>'+html.escape(r['review_status']+' | 配音性別: '+r.get('gender','原音')+' | 角色性別: '+str(r.get('requested_gender','原音')))+'</td></tr>')
        parts.append('</table>')
    (ROOT/'bazinga_audition.html').write_text('\n'.join(parts))


def main():
    records=mpdd()+bazinga();audition(records)
    summary={}
    for story in sorted({r['story_id'] for r in records}):
        rs=[r for r in records if r['story_id']==story]
        summary[story]={'characters':len(rs),'references':sum('reference_wav' in r for r in rs),
                        'unresolved':[r['character'] for r in rs if 'reference_wav' not in r]}
    write(ROOT/'manifests/summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
