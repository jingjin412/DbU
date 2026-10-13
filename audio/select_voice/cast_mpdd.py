#!/usr/bin/env python3
"""CPU-only, evidence-backed MPDD casting and per-story list.csv export.

Character requirements are editorial annotations, not acoustic measurements.
Every input turn is indexed; unresolved character facts remain unresolved.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import soundfile as sf
import wave

from select_references import ROOT, VOICE, SOURCES, stats, save_clip, write, hashfile

CASTING = ROOT / 'casting'
AGE_RANGES = {'A': '<14', 'B': '14–25', 'C': '26–40', 'D': '>41'}
BASE_FIELDS = list(next(csv.reader((CASTING/'story_01_seed.csv').open())))
EXTRA_FIELDS = ['source_labels', 'dialogue_ids', 'gender_status', 'age_basis', 'evidence_path']
CUE = re.compile(r'男人|女人|男孩|女孩|男生|女生|先生|小姐|女士|太太|丈夫|妻子|老婆|老公|女兒|兒子|姐姐|妹妹|哥哥|弟弟|大哥|大姐|父親|母親|爸爸|媽媽|姑|姨|叔|伯|爺|奶|媳婦|嫂|懷孕|歲|退休')


def evidence(data, refs):
    result = []
    for ref in refs.split(';'):
        if not ref: continue
        did, turn = ref.split(':'); t = data[did][int(turn)-1]
        result.append({'dialogue_id': did, 'turn': int(turn), 'speaker': t['speaker'],
                       'listeners': [x['name'] for x in t['listener']], 'quote': t['utterance']})
    return result


def requirements(data, stories):
    notes = {}
    for line in (CASTING/'role_notes.tsv').read_text().splitlines():
        if not line or line.startswith('#'): continue
        story, name, gender, age, basis, note, refs = line.split('\t')
        name = name.replace('黄梅', '黃梅')
        notes[(int(story), name)] = {'requested_gender': None if gender=='flexible' else gender,
            'requested_age_group': age, 'age_basis': basis, 'voice_note': note,
            'evidence': evidence(data, refs)}
    seed_rows = list(csv.DictReader((CASTING/'story_01_seed.csv').open()))
    overrides=json.loads((ROOT/'casting_overrides.json').read_text()) if (ROOT/'casting_overrides.json').exists() else {}
    first_profiles = json.loads((CASTING/'story_01_profiles.json').read_text())['characters']
    old = {}
    for p in first_profiles:
        for n in [p['name'], *p.get('aliases', [])]: old[n] = p
    identity_notes=json.loads((CASTING/'identity_notes.json').read_text())
    records=[]; scan=[]
    for index, story in enumerate(stories, 1):
        sid=story['story_id']; groups=defaultdict(lambda:defaultdict(set)); examples=defaultdict(list)
        total=0
        def identity(label, did):
            if index==1:
                return {'小寶': '左小龍（小寶）', '左小龍': '左小龍（小寶）',
                        '劉父':'劉父／外公', '外公':'劉父／外公',
                        '劉母':'劉母／外婆', '外婆':'劉母／外婆'}.get(label,label)
            if index==3:
                if label=='女孩' and did==1414: return '柳慧'
                if label in ('女孩','女音') and did==1785: return '劉言鳳'
                if label=='男人' and did==1458: return '蔡宇楓'
                return {'夢':'柯婕妤'}.get(label,label)
            if index==5:
                if label=='媽媽': return '張華' if did in (2387,2526,2990,2991) else '楊雪母親'
                if label=='老大' and did==2192: return '向大爺'
                if label=='老大' and did==2878: return '連國'
                return {'大爺':'向大爺'}.get(label,label)
            if index==6:
                if label=='母親': return '小滿母親' if did<=3090 else '順子'
                return {'春梅':'吳春梅','秀英':'高秀英','韓美玉':'美玉',
                        '少女':'美玉','老先生':'韓長林'}.get(label,label)
            return label
        for did in story['dialogue_ids']:
            for turn,t in enumerate(data[did],1):
                total+=1
                for label in dict.fromkeys([t['speaker'],*[x['name'] for x in t['listener']]]):
                    name=identity(label,int(did)); variant='全劇共用'
                    if index==1 and name=='左小龍（小寶）': variant='兒童版' if int(did)<=235 else '成年版'
                    if index==2 and name=='張愛': variant='兒童版' if did=='298' else '青年／成年版'
                    key=(name,variant);groups[key][label].add(did)
                    if t['speaker']==label:
                        examples[key].append({'dialogue_id':did,'turn':turn,'speaker':label,'text':t['utterance']})
        seeded={}
        if index==1:
            for row in seed_rows:
                name=row['character'];variant=row['variant'] if name=='左小龍（小寶）' else '全劇共用'
                seeded[(name,variant)]=row
        for key,bindings in groups.items():
            name,variant=key; spoken=examples[key];seed=seeded.get(key)
            info=notes.get((index,name))
            if index==1 and seed:
                prior=old.get(name) or old.get(next(iter(bindings))) or {}
                refs=[]
                for did in prior.get('evidence_dialogue_ids',[]):
                    for turn,t in enumerate(data[str(did)],1):
                        if t['speaker'] in bindings or any(x['name'] in bindings for x in t['listener']):
                            refs.append({'dialogue_id':str(did),'turn':turn,'speaker':t['speaker'],
                                         'listeners':[x['name'] for x in t['listener']],'quote':t['utterance']});break
                info={'requested_gender':None if seed['gender']=='flexible' else seed['gender'],
                      'requested_age_group':seed['age_group'], 'age_basis':prior.get('age','使用者CSV配音目標；精確歲數未明'),
                      'voice_note':seed['voice_note'], 'evidence':refs,
                      'character_description':prior.get('description','')}
            if info is None:
                first=spoken[0] if spoken else None
                info={'requested_gender':None,'requested_age_group':'C',
                      'age_basis':'劇本未明示歲數；26–40僅為配音目標，非人物事實',
                      'voice_note':'劇本未明示性別，保留flexible。'+('泛稱角色可能涵蓋多個實際人物；固定原標籤配音。' if name in ('服務員','服務生','醫生','警察','警員','客人','客戶','老闆','手下','甲','乙','丙','同事','四人','旅客') else '人物資料有限；不從名字或職業推定性別。'),
                      'evidence':[] if first is None else [{**{k:v for k,v in first.items() if k!='text'},'quote':first['text']}],
                      'gender_status':'unresolved'}
                if index==6 and name=='範婷婷':
                    info.update(requested_age_group='B',age_basis='年輕飯店工作者，歲數未明；B僅為配音目標',
                                voice_note='小姨早點部工作者；性別沒有實名直接證據，保留flexible，不能因名字婷婷就確定女性。')
            info=dict(info)
            if index==2 and name=='張愛' and variant=='兒童版':
                info.update(requested_age_group='A',age_basis='298 的妹妹出生、爸爸安慰小愛場景；幼年為語境推定，未明說歲數',
                            voice_note='幼年張愛，與301中考以後的青年版分開。',evidence=evidence(data,'298:3;298:8;299:3'))
            if index==5 and name=='楊雪母親':
                info.update(requested_gender='female',requested_age_group='D',age_basis='成年楊雪之母；精確歲數未明',
                            voice_note='楊雪母親；後認苗苗為乾女兒，於團圓飯被孩子們稱奶奶，仍為同人。媽媽2387、2526、2990、2991屬張華，不使用此聲音。',
                            evidence=evidence(data,'2366:11;2939:11;3037:2;3043:2'))
            if index==6 and name=='小滿母親':
                info.update(requested_gender='female',requested_age_group='D',age_basis='18歲小滿之母；精確歲數未明',
                            voice_note='小滿、陳小秋姊妹的母親；回憶場景3081–3090。與3534以後的順子分開。',
                            evidence=evidence(data,'3080:2;3086:1'))
            override=overrides.get(sid+':'+name)
            if override:
                info.update(requested_gender=override['gender'],gender_status='user_confirmed',
                            user_gender_evidence=override['evidence'])
            if 'gender_status' not in info:
                info['gender_status']='unresolved' if info['requested_gender'] is None else 'text_supported'
                if '推定' in info['voice_note'] or (index,name) in [(2,'胡永成'),(4,'徐春友'),(4,'王厚寶'),(4,'張院長'),(5,'小慧'),(5,'隊長')]:
                    info['gender_status']='context_inference'
            if info['requested_gender'] is not None: info['gender_status']=info['gender_status'].replace('unresolved','text_supported')
            canonical_id=sid+':'+name
            cid=canonical_id+(':'+variant if (index,name) in [(1,'左小龍（小寶）'),(2,'張愛')] else '')
            row={'character_id':cid,'canonical_character_id':canonical_id,'dataset':'mpdd','story_id':sid,
                 'character':name,'variant':seed['variant'] if seed else variant,'source_labels':sorted(bindings),
                 'bindings':[{'source_label':label,'dialogue_ids':sorted(ids,key=int)} for label,ids in sorted(bindings.items())],
                 'speaker_utterances':len(spoken),'priority':seed['priority'] if seed else ('核心' if len(spoken)>=200 else '主要' if len(spoken)>=50 else '次要'),
                 'full_script_evidence_index':f'script_review/{sid}/dossiers.json',
                 'utterance_index':spoken, 'gender_cue_count':sum(bool(CUE.search(x['text'])) for x in spoken),
                 'group_label':name=='四人',
                 'identity_notes':[n for n in identity_notes if n['story']==index and n['identity']==name],**info}
            if seed:row['seed_reference']=seed
            records.append(row)
        scan.append({'story_id':sid,'dialogues':len(story['dialogue_ids']),'utterances_indexed':total,
                     'source_labels':len({label for v in groups.values() for label in v}),
                     'voice_slots':len(groups), 'manual_full_line_reading_claimed':False,
                     'method':'all turns indexed; editorial character evidence checked by role; uncertain facts explicitly retained'})
    annotated=set(notes)
    resolved={(int(r['story_id'][-2:]),r['character']) for r in records}
    assert not annotated-resolved, ('Unused editorial annotations', annotated-resolved)
    write(CASTING/'character_profiles.json',records)
    write(ROOT/'manifests/script_review_coverage.json',scan)
    return records


def inventory():
    speakers={}
    for line in (SOURCES/'AISHELL-3/spk-info.txt').read_text().splitlines():
        p=line.split()
        if len(p)>=4 and not line.startswith('#'):speakers[p[0]]={'age_group':p[1],'gender':p[2],'accent':p[3]}
    transcripts={}
    for corpus in ('AISHELL-3','AISHELL-3-supplement'):
        for split in ('train','test'):
            content=SOURCES/corpus/split/'content.txt'
            if not content.exists():continue
            for line in content.read_text().splitlines():
                p=line.split()
                if p:transcripts[p[0]]=''.join(p[1::2])
    candidates=defaultdict(list)
    paths=sorted((SOURCES/'AISHELL-3').glob('*/wav/*/*.wav'))+sorted((SOURCES/'AISHELL-3-supplement').glob('*/wav/*/*.wav'))
    for path in paths:
        with wave.open(str(path),'rb') as f:duration=f.getnframes()/f.getframerate()
        if 3<=duration<=10:candidates[path.parent.name].append((abs(duration-6),path))
    reps={}
    for speaker,paths in candidates.items():
        for _,path in sorted(paths)[:12]:
            values,rate=sf.read(path,dtype='float32');q=stats(values)
            if q['rms']>=.01 and q['clipped_fraction']<=.001:
                reps[speaker]={'path':path,'duration_seconds':len(values)/rate,**q};break
    return speakers,transcripts,reps


def bind_index(records):
    index={}
    for r in records:
        for b in r['bindings']:
            for did in b['dialogue_ids']:
                key=(r['story_id'],did,b['source_label'])
                if key in index:raise ValueError('Ambiguous voice binding '+str(key))
                index[key]=r
    return index


def main():
    data=json.loads((VOICE/'data/mpdd/dialogue_chinese.json').read_text())
    stories=json.loads((ROOT/'manifests/mpdd_stories.json').read_text())
    profiles=requirements(data,stories)
    supplement_path=SOURCES/'casting_supplement_manifest.json'
    supplement=json.loads(supplement_path.read_text()) if supplement_path.exists() else None
    if supplement:
        assert set(supplement['target_speakers'])<={r['speaker_id'] for r in supplement['files'] if r['usable']},'Incomplete casting supplement'
    speakers,transcripts,reps=inventory();result=[];summary=[]
    for story in stories:
        sid=story['story_id']; selected=[r for r in profiles if r['story_id']==sid]
        # Reserve all seeded and gender-constrained voices before assigning flexible roles.
        allocated=set(); choices={}
        order=sorted(selected,key=lambda r:(not bool(r.get('seed_reference')),r['requested_gender'] is None,
                    {'核心':0,'主要':1,'次要':2}.get(r['priority'],2),-r['speaker_utterances'],r['character_id']))
        for r in order:
            gender=r['requested_gender'];age=r['requested_age_group'];seed=r.get('seed_reference')
            if seed:
                speaker=seed['speaker_id'];path=SOURCES/seed['source_path']
                if speaker in allocated:raise ValueError('Duplicate seed speaker '+speaker)
                if not path.is_file():raise ValueError('Missing seeded source '+str(path))
                if gender and speakers[speaker]['gender']!=gender:raise ValueError('Seed gender mismatch '+r['character'])
                policy='user_story_01_csv_exact_source'
            else:
                eligible=[s for s in reps if s not in allocated and (not gender or speakers[s]['gender']==gender)]
                if not eligible:raise ValueError('Not enough same-gender distinct voices: '+r['character_id'])
                # Match known age targets first, then signal quality; never relax known gender.
                speaker=min(eligible,key=lambda s:(abs(ord(speakers[s]['age_group'])-ord(age)),
                      reps[s]['clipped_fraction'],reps[s]['quiet_fraction'],abs(reps[s]['rms']-.055),s))
                path=reps[speaker]['path'];policy='gender_constraint_then_nearest_age_then_signal_quality'
            allocated.add(speaker);choices[r['character_id']]=(speaker,path,policy)
        rows=[]
        # Preserve user's R01..R43 IDs in story 1; all other scripts prioritise core roles.
        display=sorted(selected,key=lambda r:(int(r['seed_reference']['reference_id'][1:]) if r.get('seed_reference') else 999,
                     {'核心':0,'主要':1,'次要':2}.get(r['priority'],2),-r['speaker_utterances'],r['character_id']))
        folder=ROOT/'references/mpdd'/sid;folder.mkdir(parents=True,exist_ok=True)
        for number,r in enumerate(display,1):
            speaker,path,policy=choices[r['character_id']];meta=speakers[speaker]
            rid=r['seed_reference']['reference_id'] if r.get('seed_reference') else f'R{number:02d}'
            wav=folder/'wav'/f'{rid}_{speaker}.wav';values,rate=sf.read(path,dtype='float32')
            delta=abs(ord(meta['age_group'])-ord(r['requested_age_group']))
            status='exact' if delta==0 else 'fallback_age'
            info={k:v for k,v in r.items() if k not in ('seed_reference','utterance_index')}
            info.update(reference_id=rid,aishell_speaker=speaker,**meta,
                       source_path=str(path.relative_to(ROOT)),source_transcript=transcripts.get(path.name,''),
                       casting_policy=policy,match_status=status,age_distance=delta,
                       review_status='text_requirements_checked; acoustic_audition_pending',
                       **save_clip(wav,values,rate))
            if 'AISHELL-3-supplement' in path.parts:
                if not supplement:raise ValueError('Supplement source lacks provenance manifest')
                info.update(source_repo=supplement['repo'],source_revision=supplement['revision'])
            else:
                info.update(source_repo='AISHELL/AISHELL-3',source_revision='f20d5db4a31fe779ef07bb1af4ea92da5c786622')
            note=r['voice_note']
            if delta:note+='；實際reference年齡不足，標示fallback，不宣稱已完成年齡變聲。'
            if r['gender_status']=='unresolved':note+=' 配音'+meta['gender']+'僅為選角選擇，非角色性別判定。'
            if '老化' in note:note+='（目前IndexTTS runner沒有可驗證的自動老化功能；沿用此聲音不代表後期年齡已匹配。）'
            info['voice_note']=note
            result.append(info)
            ev=folder/'profiles'/f'{rid}.json';write(ev,info)
            rows.append(dict(reference_id=rid,character=r['character'],variant=r['variant'],age_group=r['requested_age_group'],
                       age_range=AGE_RANGES[r['requested_age_group']],gender=r['requested_gender'] or 'flexible',priority=r['priority'],
                       voice_note=note,speaker_id=speaker,actual_age_group=meta['age_group'],actual_age_range=AGE_RANGES[meta['age_group']],
                       actual_gender=meta['gender'],accent=meta['accent'],match_status=status,age_distance=delta,
                       duration_seconds=f"{info['duration_seconds']:.3f}",rms=f"{info['rms']:.5f}",peak=f"{info['peak']:.5f}",
                       transcript=info['source_transcript'],source_path=str(path.relative_to(SOURCES)),preview_path=f'wav/{wav.name}',
                       source_labels=';'.join(r['source_labels']),dialogue_ids=';'.join(sorted({d for b in r['bindings'] for d in b['dialogue_ids']},key=int)),
                       gender_status=r['gender_status'],age_basis=r['age_basis'],evidence_path=f'profiles/{rid}.json'))
        with (folder/'list.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=BASE_FIELDS+EXTRA_FIELDS);writer.writeheader();writer.writerows(rows)
        summary.append({'story_id':sid,'voice_slots':len(rows),'gender_unresolved':sum(r['gender']=='flexible' for r in rows),
                        'age_fallback':sum(r['match_status']=='fallback_age' for r in rows),
                        'age_exact':sum(r['match_status']=='exact' for r in rows),'distinct_speakers':len(allocated)})
    bindings=bind_index(result)
    for story in stories:
        for did in story['dialogue_ids']:
            for t in data[did]:
                assert (story['story_id'],did,t['speaker']) in bindings
                for listener in t['listener']:assert (story['story_id'],did,listener['name']) in bindings
    write(ROOT/'manifests/mpdd_voices.json',result)
    write(ROOT/'manifests/mpdd_casting_summary.json',summary)
    for story in stories:
        n=next(r['voice_slots'] for r in summary if r['story_id']==story['story_id'])
        story['voice_count']=n
    write(ROOT/'manifests/mpdd_stories.json',stories)
    summary_path=ROOT/'manifests/summary.json'
    combined=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for row in summary:
        combined[row['story_id']]={'characters':row['voice_slots'],'references':row['voice_slots'],'unresolved':[],
                                  'unresolved_character_gender':row['gender_unresolved'],'age_fallback':row['age_fallback']}
    write(summary_path,combined)
    write(ROOT/'manifests/unconfirmed_character_genders.json',[{'story_id':r['story_id'],'character':r['character'],
          'assigned_voice_gender':r['gender'],'evidence_path':r['full_script_evidence_index']} for r in result if r['requested_gender'] is None])
    write(ROOT/'manifests/aishell_inventory.json',{'available_speakers':len(reps),'speaker_info':speakers,
          'available_age_gender_counts':{f'{g}_{a}':n for (g,a),n in Counter((speakers[s]['gender'],speakers[s]['age_group']) for s in reps).items()}})
    archive_obsolete(result)
    bazinga_path=ROOT/'manifests/bazinga_voices.json'
    if bazinga_path.exists():
        from select_references import audition as combined_audition
        combined_audition(result+json.loads(bazinga_path.read_text()))
    else:audition(result)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def archive_obsolete(records):
    """Move superseded generated clips aside; preserve user files and CSV seed."""
    keep={ROOT/r['reference_wav'] for r in records}
    keep_profiles={ROOT/'references/mpdd'/r['story_id']/'profiles'/f"{r['reference_id']}.json" for r in records}
    moved=[]
    for folder in (ROOT/'references/mpdd').glob('mpdd_story_*'):
        candidates=[p for p in folder.glob('*.wav') if re.fullmatch(r'[0-9a-f]{16}(?:_SSB[0-9]+)?',p.stem)]
        candidates.extend(p for p in (folder/'wav').glob('R*_SSB*.wav') if p not in keep)
        candidates.extend(p for p in (folder/'profiles').glob('R*.json') if p not in keep_profiles)
        for path in candidates:
            relative=path.relative_to(ROOT/'references/mpdd')
            dest=ROOT/'references/.archive/previous_mpdd'/relative
            if dest.exists():dest=dest.with_name(dest.stem+'_'+hashfile(path)[:10]+dest.suffix)
            if dest.exists():continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(path),str(dest));moved.append({'from':str(path.relative_to(ROOT)),'to':str(dest.relative_to(ROOT))})
    if moved:
        log=ROOT/'manifests/archived_provisional_mpdd_references.json'
        previous=json.loads(log.read_text()) if log.exists() else []
        write(log,previous+moved)


def audition(records):
    parts=['<!doctype html><meta charset="utf-8"><title>MPDD 角色配音</title>',
           '<style>body{font-family:sans-serif;margin:30px;max-width:1500px}td{padding:10px;border-bottom:1px solid #ddd;vertical-align:top}audio{width:260px}.warn{color:#a44}details{max-width:550px}summary{cursor:pointer}th{text-align:left}</style>',
           '<h1>MPDD 六劇本角色配音與 reference 試聽</h1>',
           '<p>台詞索引涵蓋全部劇本。人物設定依角色台詞查核；未明性別標示 flexible，不能把配音者性別當人物事實。聲線描述是演出目標，尚未人工聽取確認。A=&lt;14、B=14–25、C=26–40、D=&gt;41。</p>',
           '<p>AISHELL 原105人聲音池另補10人，目前115人、2位女童、1位D組男性。年齡 fallback 明列，IndexTTS沒有在此流程保證改變年齡。 <a href="bazinga_audition.html">Bazinga原音試聽</a></p>']
    for sid in dict.fromkeys(r['story_id'] for r in records):
        path='references/mpdd/'+sid+'/list.csv'
        parts.append(f'<h2>{sid} <a href="{path}">list.csv</a> · <a href="script_review/{sid}/full_script.txt">完整劇本</a></h2><table><tr><th>角色／別名</th><th>參考聲音</th><th>人物設定與依據</th></tr>')
        for r in records:
            if r['story_id']!=sid:continue
            esc=html.escape
            proof=''.join(f"<p>{e['dialogue_id']}:{e['turn']} {esc(e['speaker'])}：{esc(e['quote'])}</p>" for e in r['evidence'])
            proof+=''.join(f"<p>身分對應：{esc('／'.join(n['labels']))} → {esc(n['identity'])} [{esc(n['confidence'])}] {esc(n.get('rule',''))} {esc(n.get('note',''))}</p>" for n in r['identity_notes'])
            gender=r['requested_gender'] or 'flexible（劇本未明示）'
            status=esc(r['match_status']);css='warn' if r['age_distance'] or r['requested_gender'] is None else ''
            parts.append(f'<tr><td><b>{esc(r["character"])}</b> · {esc(r["variant"])}<br>來源：{esc("／".join(r["source_labels"]))}<br>{r["speaker_utterances"]}句 · {r["reference_id"]}</td>'
                         f'<td><audio controls preload="none" src="{esc(r["reference_wav"])}"></audio><br>{r["aishell_speaker"]} · {r["gender"]} · {r["age_group"]}<br><span class="{css}">{status} · {r["duration_seconds"]:.2f} 秒</span></td>'
                         f'<td>人物性別：{esc(gender)} [{esc(r["gender_status"])}]<br>年齡目標：{r["requested_age_group"]}；{esc(r["age_basis"])}<br>{esc(r["voice_note"])}<details><summary>台詞依據／適用場景</summary>{proof}<p>{esc(str(r["bindings"]))}</p></details></td></tr>')
        parts.append('</table>')
    (ROOT/'audition.html').write_text('\n'.join(parts))


if __name__=='__main__':main()
