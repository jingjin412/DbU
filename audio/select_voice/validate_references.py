#!/usr/bin/env python3
"""CPU validation of manifests, independent clips, story coverage and task plans."""
import hashlib
import json
from pathlib import Path
import wave
import csv
from collections import Counter

ROOT=Path(__file__).resolve().parent
VOICE=ROOT.parents[1]


def load(p):return json.loads(p.read_text())


def main():
    mpdd=load(ROOT/'manifests/mpdd_voices.json');bazinga=load(ROOT/'manifests/bazinga_voices.json')
    records=mpdd+bazinga
    assert len({r['character_id'] for r in records})==len(records)
    supplement_path=ROOT/'sources/casting_supplement_manifest.json'
    if supplement_path.exists():
        supplement=load(supplement_path)
        for row in supplement['files']:
            assert row['lfs_sha256']==row['sha256']
            assert hashlib.sha256((ROOT/row['local_path']).read_bytes()).hexdigest()==row['sha256']
        assert supplement['metadata_matches_primary_snapshot']
    for r in records:
        if not r.get('reference_wav'):
            assert r['dataset']=='bazinga' and r['speaker_tokens']==0
            continue
        p=ROOT/r['reference_wav']
        assert not p.is_symlink()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['reference_sha256']
        with wave.open(str(p),'rb') as f:
            assert f.getnframes()>0 and f.getsampwidth()==2
            assert abs(f.getnframes()/f.getframerate()-r['duration_seconds'])<1e-9
        assert r['rms']>0
    stories=load(ROOT/'manifests/mpdd_stories.json')
    ds=load(VOICE/'data/mpdd/dialogue.json')
    ids=[i for s in stories for i in s['dialogue_ids']]
    assert len(ids)==len(set(ids)) and set(ids)==set(ds)
    for story in stories:
        selected=[r for r in mpdd if r['story_id']==story['story_id']]
        assert len({r['aishell_speaker'] for r in selected})==len(selected)
        mapped={label for r in selected for label in r['source_labels']}
        assert all(t['speaker'] in mapped for did in story['dialogue_ids'] for t in ds[did])
        bindings={}
        for r in selected:
            if r['requested_gender'] is not None:
                assert r['requested_gender']==r['gender'],r['character_id']
            for binding in r['bindings']:
                for did in binding['dialogue_ids']:
                    key=(did,binding['source_label'])
                    assert key not in bindings,('ambiguous',story['story_id'],key)
                    bindings[key]=r
            for e in r['evidence']:
                assert ds[e['dialogue_id']][e['turn']-1]['utterance']==e['quote']
        for did in story['dialogue_ids']:
            for t in ds[did]:
                assert (did,t['speaker']) in bindings
                assert all((did,l['name']) in bindings for l in t['listener'])
        csv_path=ROOT/'references/mpdd'/story['story_id']/'list.csv'
        rows=list(csv.DictReader(csv_path.open()))
        assert len(rows)==len(selected)
        by_ref={r['reference_id']:r for r in selected}
        for row in rows:
            voice=by_ref[row['reference_id']]
            assert (csv_path.parent/row['preview_path']).resolve()==(ROOT/voice['reference_wav']).resolve()
            assert (ROOT/'sources'/row['source_path']).is_file()
            assert row['speaker_id']==voice['aishell_speaker']
            assert row['actual_gender']==voice['gender']
            assert row['gender']=='flexible' or row['gender']==row['actual_gender']
            assert row['match_status']==('exact' if row['age_group']==row['actual_age_group'] else 'fallback_age')
    plans={}
    for dataset in ('bazinga','mpdd'):
        plan=load(ROOT.parent/f'index-TTS/plans/{dataset}_neutral.json')
        assert plan['voices_sha256']==hashlib.sha256((ROOT/f'manifests/{dataset}_voices.json').read_bytes()).hexdigest()
        assert not plan['blocked']
        assert len({j['utterance_id'] for j in plan['jobs']})==len(plan['jobs'])
        if dataset=='mpdd':
            all_bindings={(r['story_id'],did,b['source_label']):r for r in mpdd for b in r['bindings'] for did in b['dialogue_ids']}
            for job in plan['jobs']:
                voice=all_bindings[(job['story_id'],job['dialogue_id'],job['speaker'])]
                assert voice['character_id']==job['voice_character_id']
                assert voice['reference_sha256']==job['reference_sha256']
        plans[dataset]=len(plan['jobs'])
    # Check boundary and generic-label routing on real corpus cases.
    all_bindings={(r['story_id'],did,b['source_label']):r for r in mpdd for b in r['bindings'] for did in b['dialogue_ids']}
    def v(story,did,name):return all_bindings[(f'mpdd_story_{story:02d}',str(did),name)]
    assert v(1,118,'小寶')['variant']=='兒童版'
    assert v(1,236,'小寶')['variant']=='成年版'
    assert v(1,296,'左小龍')['aishell_speaker']==v(1,236,'小寶')['aishell_speaker']
    assert v(2,298,'張愛')['age_group']=='A' and v(2,301,'張愛')['age_group']=='B'
    assert v(2,298,'張愛')['gender']==v(2,301,'張愛')['gender']=='female'
    assert v(3,1476,'夢')['character']=='柯婕妤'
    assert v(3,1414,'女孩')['character']=='柳慧'
    assert v(3,1785,'女音')['character']==v(3,1785,'女孩')['character']=='劉言鳳'
    assert v(5,2366,'媽媽')['character']=='楊雪母親'
    assert v(5,2991,'媽媽')['character']=='張華'
    assert v(5,3043,'媽媽')['character']=='楊雪母親'
    assert v(5,2192,'老大')['character']=='向大爺'
    assert v(5,2878,'老大')['character']=='連國'
    assert v(6,3084,'母親')['character']=='小滿母親'
    assert v(6,4023,'母親')['character']=='順子'
    assert v(6,3380,'亞玲')['aishell_speaker']!=v(6,3380,'亞琳')['aishell_speaker']
    full_plan=load(ROOT.parent/'index-TTS/plans/mpdd_full_neutral.json')
    assert full_plan['voices_sha256']==hashlib.sha256((ROOT/'manifests/mpdd_voices.json').read_bytes()).hexdigest()
    assert not full_plan['blocked']
    assert len(full_plan['jobs'])==sum(len(ts) for ts in ds.values())
    assert len({j['utterance_id'] for j in full_plan['jobs']})==len(full_plan['jobs'])
    for job in full_plan['jobs']:
        voice=all_bindings[(job['story_id'],job['dialogue_id'],job['speaker'])]
        assert voice['character_id']==job['voice_character_id']
        assert voice['reference_sha256']==job['reference_sha256']
    coverage=load(ROOT/'manifests/script_review_coverage.json')
    assert sum(r['utterances_indexed'] for r in coverage)==sum(len(ts) for ts in ds.values())
    report={'passed':True,'mpdd_stories':len(stories),'mpdd_voice_slots':len(mpdd),
            'bazinga_speaking_labels':sum(r['speaker_tokens']>0 for r in bazinga),
            'bazinga_references':sum(bool(r.get('reference_wav')) for r in bazinga),
            'bazinga_short_references':sum(r.get('short_reference',False) for r in bazinga),
            'listener_only_or_annotation_artifact_labels':sum(not r.get('reference_wav') for r in bazinga),
            'neutral_plan_utterances':plans,'full_mpdd_plan_utterances':len(full_plan['jobs']),
            'gender_unresolved_voice_slots':sum(r['requested_gender'] is None for r in mpdd),
            'age_fallback_voice_slots':sum(r['age_distance']>0 for r in mpdd),
            'checks':['reference hashes, decodable PCM WAV and duration','all MPDD dialogue IDs covered exactly once',
                      'unique AISHELL speaker per voice slot within each story','gender constraints enforced; fallback ages explicit',
                      'all speaker and listener labels have exactly one scene-specific binding',
                      'evidence quotes match corpus; CSV paths and speaker metadata agree',
                      'age variants and generic-label identity routing checked on real scenes',
                      'no missing references for current >=3-participant task plans; each MPDD job uses the correct variant'],
            'not_validated':['manual line-by-line reading of all scripts','human voice identity','background music/noise','perceived age/gender/role fit','TTS GPU runtime and generated audio quality']}
    (ROOT/'manifests/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
