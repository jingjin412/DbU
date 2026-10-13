import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import soundfile as sf

from data import samples
from emotion_context import attach, load
from tasks import prompt
from model_io import Client


class EmotionContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data = {
            'd': [{'speaker': 'A', 'utterance': 'FIRST', 'listener': [{'name': 'B'}, {'name': 'C'}]},
                  {'speaker': 'B', 'utterance': 'CURRENT', 'listener': [{'name': 'A'}]},
                  {'speaker': 'C', 'utterance': 'SECRET_FUTURE', 'listener': [{'name': 'A'}]}]}
        self.args = SimpleNamespace(data=root / 'dialogue.json', metadata=None, gold_emotions=None,
                                    dataset='bazinga', naming='original', min_participants=3,
                                    context_turns=0, max_dialogues=0, max_samples=0, condition='text',
                                    task='downstream', history_emotions=root / 'predictions.jsonl',
                                    history_emotion_gold=None)
        self.args.data.write_text(json.dumps(self.data))
        self.rows = [{**s, 'condition': 'text', 'prompt': prompt(s), 'status': 'ok',
                      'prediction': label} for s, label in zip(samples(self.args, 'emotion_recognition'),
                                                              ['angry', 'sadness', 'surprise'])]
        self.save()

    def save(self):
        self.args.history_emotions.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def test_visible_labels_only_and_unchanged_task_contract(self):
        selected = samples(self.args, 'next_speaker_ranking')
        original = prompt(selected[1])
        audit = attach(selected, load(self.args))
        augmented = prompt(selected[1])
        self.assertIn('A [emotion=angry]: FIRST', augmented)
        self.assertIn('B [emotion=sadness]: CURRENT', augmented)
        self.assertNotIn('surprise', augmented)
        self.assertNotIn('SECRET_FUTURE', augmented)
        self.assertEqual(augmented.split('\n\n')[-1], original.split('\n\n')[-1])
        self.assertEqual(selected[1]['gold'], 'C')
        self.assertEqual(audit['visible_unique_utterances'], 2)

    def test_invalid_emotion_explicitly_unavailable_keeps_samples(self):
        self.rows[0].update(status='invalid', prediction=None)
        self.save()
        selected = samples(self.args, 'addressee_recognition')
        audit = attach(selected, load(self.args))
        self.assertEqual(len(selected), 3)
        self.assertIn('[emotion=unavailable]', prompt(selected[0]))
        self.assertEqual(audit['unavailable_utterances'], 1)
        self.assertEqual(audit['samples_with_unavailable_emotion'], 3)

    def test_mismatched_future_prompt_missing_and_duplicate_sources_rejected(self):
        original = self.rows[0]['prompt']
        self.rows[0]['prompt'] += ' SECRET_FUTURE'
        self.save()
        with self.assertRaisesRegex(ValueError, 'history/prompt differs'):
            load(self.args)
        self.rows[0]['prompt'] = original
        missing = self.rows.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, 'Missing emotion prediction'):
            load(self.args)
        self.rows.extend([missing, missing])
        self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate emotion sample'):
            load(self.args)

    def test_gold_alignment_and_future_isolation(self):
        gold_path = self.args.data.with_name('gold.json')
        gold = {'d': [{**t, 'emotion': label} for t, label in zip(self.data['d'],
                                                               ['angry', 'sadness', 'surprise'])]}
        gold_path.write_text(json.dumps(gold))
        self.args.history_emotions = None
        self.args.history_emotion_gold = gold_path
        selected = samples(self.args, 'next_speaker_ranking')
        attach(selected, load(self.args))
        self.assertNotIn('surprise', prompt(selected[1]))
        gold['d'][0]['speaker'] = 'WRONG'
        gold_path.write_text(json.dumps(gold))
        with self.assertRaisesRegex(ValueError, 'text/speaker differs'):
            load(self.args)

    def test_baseline_prompt_ignores_scoring_gold_and_condition_guard(self):
        selected = samples(self.args, 'next_speaker_ranking')
        self.assertNotIn('[emotion=', prompt(selected[1]))
        self.args.condition = 'predicted_audio'
        with self.assertRaisesRegex(ValueError, 'text or neutral_audio downstream'):
            load(self.args)

    def test_neutral_audio_keeps_emotion_prompt_and_excludes_future_audio(self):
        self.args.condition='neutral_audio'
        self.args.model_kind='gemma4-e4b'
        self.args.audio_scope='history'
        self.args.max_audio_clips=64
        self.args.audio_manifest=self.args.data.with_name('audio.json')
        manifest={}
        for i,s in enumerate(samples(self.args,'emotion_recognition')):
            path=self.args.data.with_name(f'{i}.wav')
            sf.write(path,np.full(1600,.1*(i+1),dtype=np.float32),16000)
            manifest[s['target_utterance_id']]={'path':str(path),'emotion':'neutral','source':'neutral_audio','fallback':False}
        self.args.audio_manifest.write_text(json.dumps(manifest))
        selected=samples(self.args,'next_speaker_ranking')
        attach(selected,load(self.args))
        content=Client(self.args).messages(selected[1])[0]['content']
        self.assertEqual(content[0]['text'],prompt(selected[1]))
        self.assertIn('[emotion=angry]',content[0]['text'])
        self.assertIn('[emotion=sadness]',content[0]['text'])
        self.assertNotIn('SECRET_FUTURE',str(content))
        self.assertNotIn('[emotion=surprise]',str(content))
        self.assertEqual(sum(p['type']=='audio_url' for p in content),2)
        usage=Client(self.args).audio_usage(selected[1])
        self.assertEqual(len(usage),2)
        self.assertTrue(all(e['emotion']=='neutral' for e in usage))


if __name__ == '__main__':
    unittest.main()
