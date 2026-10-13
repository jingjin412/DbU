#!/usr/bin/env python3
"""CPU planning: join dialogue speakers, frozen references and optional model predictions."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VOICE=ROOT.parents[1]
SELECT=ROOT.parent/'select_voice'
from emotion_settings import ORDER, STRENGTH, EMO_ALPHA, validate_jobs


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['mpdd','bazinga'],required=True)
    p.add_argument('--predictions',type=Path,help='One emotion_recognition predictions.jsonl from a single model/naming run')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,help='Custom reviewed dialogue subset; adjacent Bazinga metadata required')
    p.add_argument('--voices',type=Path,help='Custom reviewed voice manifest')
    p.add_argument('--min-participants',type=int,default=3)
    p.add_argument('--emotion-strength',type=float,choices=[STRENGTH],default=STRENGTH)
    args=p.parse_args()
    if not 0<args.emotion_strength<=1:raise ValueError('emotion strength must be (0,1]')
    data_path=args.data or VOICE/('data/mpdd/dialogue.json' if args.dataset=='mpdd' else 'data/bazinga/clean/dialogue.json')
    ds=json.loads(data_path.read_text())
    metadata=json.loads((data_path.parent/'metadata.json').read_text()) if args.dataset=='bazinga' else {}
    voices_path=args.voices or SELECT/f'manifests/{args.dataset}_voices.json';voices=json.loads(voices_path.read_text())
    lookup={}
    for voice in voices:
        if args.dataset=='mpdd':
            if not voice.get('bindings'):
                raise ValueError('Rebuild MPDD casting with select_voice/cast_mpdd.py: missing dialogue-scoped bindings')
            mappings=[(b['source_label'],did) for b in voice['bindings'] for did in b['dialogue_ids']]
        else:
            mappings=[(name,'*') for name in voice['source_labels']]
        for name,did in mappings:
            key=(voice['story_id'],did,name)
            if key in lookup:raise ValueError('Ambiguous reference binding: '+str(key))
            lookup[key]=voice
    stories=json.loads((SELECT/'manifests/mpdd_stories.json').read_text())
    story_by_dialogue={did:story['story_id'] for story in stories for did in story['dialogue_ids']}
    predictions={};prediction_run=None
    if args.predictions:
        cfg_path=args.predictions.parent/'config.json'
        if not cfg_path.exists():raise ValueError('Emotion predictions must have adjacent config.json')
        cfg=json.loads(cfg_path.read_text())
        if cfg['task']!='emotion_recognition' or cfg['dataset']!=args.dataset or cfg['condition']!='text':raise ValueError('Need same-dataset text emotion predictions')
        if cfg['data_sha256']!=digest(data_path):raise ValueError('Prediction data hash mismatch')
        prediction_run={'config':cfg,'sha256':digest(args.predictions),'source':str(args.predictions.resolve())}
        for line in args.predictions.read_text().splitlines():
            r=json.loads(line);uid=r['target_utterance_id']
            if uid in predictions:raise ValueError('Duplicate emotion utterance ID')
            predictions[uid]=r
    jobs=[];blocked=[];used_refs={}
    for did,turns in ds.items():
        participants={t['speaker'] for t in turns}|{l['name'] for t in turns for l in t['listener']}
        if len(participants)<args.min_participants:continue
        story=story_by_dialogue[did] if args.dataset=='mpdd' else did.split('.')[0]
        for i,t in enumerate(turns):
            uid=f'mpdd:{did}:u{i+1:05d}' if args.dataset=='mpdd' else metadata[did]['utterances'][i]['utterance_id']
            voice=lookup.get((story,did if args.dataset=='mpdd' else '*',t['speaker']))
            error=None;emotion='neutral'
            if not voice or not voice.get('reference_wav'):error='missing_reference'
            if args.predictions:
                pred=predictions.get(uid)
                if not pred or pred['status']!='ok' or pred['prediction'] not in ORDER or pred['prediction']=='melancholic':error='missing_or_invalid_emotion_prediction'
                else:emotion=pred['prediction']
            if error:
                blocked.append({'utterance_id':uid,'speaker':t['speaker'],'reason':error});continue
            ref=SELECT/voice['reference_wav']
            if digest(ref)!=voice['reference_sha256']:raise ValueError('Reference changed: '+str(ref))
            used_refs[voice['character_id']]=voice
            vector=[0.0]*8;vector[ORDER.index(emotion)]=args.emotion_strength
            job={'utterance_id':uid,'dialogue_id':did,'turn_index':i,'dataset':args.dataset,'story_id':story,
                 'speaker':t['speaker'],'text':t['utterance'],'voice_character_id':voice['character_id'],
                 'voice_variant':voice.get('variant','original_voice'),'reference_wav':str(ref.resolve()),
                 'reference_sha256':voice['reference_sha256'],'reference_review_status':voice['review_status'],
                 'emotion':emotion,'emotion_vector':vector,'emo_alpha':EMO_ALPHA,'use_random':False,
                 'seed':42,'text_normalization':'opencc_t2s' if args.dataset=='mpdd' else 'identity'}
            jobs.append(job)
    validate_jobs(jobs)
    reviewed=bool(used_refs) and all(v['review_status']=='human_review_accepted' for v in used_refs.values())
    result={'dataset':args.dataset,'condition':'predicted_audio' if args.predictions else 'neutral_audio',
            'data_sha256':digest(data_path),'voices_sha256':digest(voices_path),
            'versions':json.loads((ROOT/'versions.json').read_text()),'min_participants':args.min_participants,
            'prediction_run':prediction_run,'jobs':jobs,'blocked':blocked,
            'reference_count':len(used_refs),'reference_policy':('human-reviewed references; story-scoped casting' if reviewed else 'story_character_and_dialogue_scoped_variant; acoustic audition pending'),
            'neutral_policy':'explicit calm vector with same strength as other classes; needs pilot QC'}
    if args.data or args.voices:
        result.update(data_source=str(data_path.resolve()),voices_source=str(voices_path.resolve()))
        audit=data_path.with_name('reference_review.json')
        if audit.exists():
            result['reference_review']={'source':str(audit.resolve()),'sha256':digest(audit)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'jobs':len(jobs),'blocked':len(blocked),'references':len(used_refs),'output':str(args.output)}))


if __name__=='__main__':main()
