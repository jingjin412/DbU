"""Lossless full-script export plus indexed character evidence; never infer gender from a name."""
import json,re
from pathlib import Path
from collections import defaultdict
R=Path(__file__).resolve().parent
V=R.parents[2]
data=json.loads((V/'data/mpdd/dialogue_chinese.json').read_text())
stories=json.loads((R.parent/'manifests/mpdd_stories.json').read_text())
pattern=re.compile(r'男人|女人|男孩|女孩|男生|女生|先生|小姐|女士|太太|丈夫|妻子|老婆|老公|女兒|兒子|姐姐|妹妹|姊|哥哥|弟弟|大哥|大姐|父親|母親|爸爸|媽媽|爸|媽|爹|娘|姑|姨|叔|伯|爺|奶|孫|媳婦|女婿|嫂|懷孕|結婚|歲|退休|同學|老師|校長|院長|書記|處長|局長|大學|中學|小學|警察|護士|醫生')
summary=[]
for story in stories:
 sid=story['story_id']; folder=R/sid;folder.mkdir(exist_ok=True)
 dossier=defaultdict(lambda:{'spoken':[],'mentions':[],'addressed':[]})
 full=[];n=0
 for did in story['dialogue_ids']:
  ts=data[did]; full.append(f'\n=== Dialogue {did} ===')
  for i,t in enumerate(ts):
   text=t['utterance'];names=[x['name'] for x in t['listener']]
   full.append(f'{did}:{i+1} {t["speaker"]} → {",".join(names)}: {text}');n+=1
   row={'dialogue_id':did,'turn':i+1,'speaker':t['speaker'],'listeners':names,'text':text,
        'context_before':[{'speaker':r['speaker'],'text':r['utterance']} for r in ts[max(0,i-1):i]],
        'context_after':[{'speaker':r['speaker'],'text':r['utterance']} for r in ts[i+1:i+2]],
        'cues':pattern.findall(text)}
   dossier[t['speaker']]['spoken'].append(row)
   for name in names:dossier[name]['addressed'].append(row)
 names=list(dossier)
 for did in story['dialogue_ids']:
  for i,t in enumerate(data[did]):
   for name in names:
    if len(name)>=2 and name in t['utterance']:
     dossier[name]['mentions'].append({'dialogue_id':did,'turn':i+1,'speaker':t['speaker'],'text':t['utterance'],'cues':pattern.findall(t['utterance'])})
 (folder/'full_script.txt').write_text('\n'.join(full)+'\n')
 (folder/'dossiers.json').write_text(json.dumps(dossier,ensure_ascii=False,indent=2)+'\n')
 summary.append({'story_id':sid,'dialogues':len(story['dialogue_ids']),'utterances':n,'characters':len(dossier)})
(R/'coverage.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
