import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import sys

from data import samples
from tasks import parse, prompt, metrics, TASKS, output_schema
from model_io import Client

ROOT = Path(__file__).resolve().parent


class Contracts(unittest.TestCase):
    def test_plain_outputs_remain_strict(self):
        candidates = ['左母', '左父', '左正鵬']
        self.assertEqual(parse('左父, 左母', TASKS[0], candidates, 'plain'), (['左父', '左母'], None))
        self.assertEqual(parse('左父，左母', TASKS[0], candidates, 'plain'), (['左父', '左母'], None))
        self.assertEqual(parse('neutral', TASKS[1], candidates, 'plain'), ('neutral', None))
        self.assertEqual(parse('左父 > 左母 > 左正鵬', TASKS[2], candidates, 'plain'),
                         (['左父', '左母', '左正鵬'], None))
        for text in ('左父, 左母, 左正鵬', '左父，左母，左正鵬'):
            self.assertEqual(parse(text, TASKS[2], candidates, 'plain'),
                             (['左父', '左母', '左正鵬'], None))
        for task, text in [(TASKS[0], '左父, 左父'), (TASKS[0], '左父,'),
                           (TASKS[0], '答案是左父'), (TASKS[0], '左康'),
                           (TASKS[1], 'Emotion: neutral'), (TASKS[1], '"neutral"'),
                           (TASKS[2], '左父 > 左母'), (TASKS[2], '左父 > 左母 > 左父'),
                           (TASKS[2], '左父, 左母, 左父'), (TASKS[2], '左父, 左母'),
                           (TASKS[2], '左父 > 左母, 左正鵬'), (TASKS[2], '左父, 左母, 左正鵬,'),
                           (TASKS[2], '左父\n左母\n左正鵬'),
                           (TASKS[2], '["左父","左母","左正鵬"]')]:
            with self.subTest(task=task, text=text):
                self.assertIsNotNone(parse(text, task, candidates, 'plain')[1])
        # Plain parsing does not silently replace the old JSON parser.
        self.assertIsNotNone(parse('neutral', TASKS[1], candidates)[1])
        sample = {'task': TASKS[2], 'history': [{'index': 0, 'speaker': '左父', 'utterance': 'Hello'}],
                  'candidates': candidates}
        self.assertNotIn('JSON', prompt(sample, 'plain'))
        self.assertIn('separated by >', prompt(sample, 'plain'))

    def test_plain_http_runner(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({'data': [{'id': 'voice-model'}]}).encode())
            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append(req)
                text = req['messages'][0]['content'][0]['text']
                if 'Who is the LAST' in text: answer = 'P02, P03'
                elif 'dominant emotion' in text: answer = 'neutral'
                else: answer = 'P01 > P02 > P03'
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}]}).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td)/'dialogue.json'
                path.write_text(json.dumps({'d': [
                    {'speaker': 'A', 'utterance': 'VISIBLE', 'listener': [{'name': 'B'}, {'name': 'C'}], 'emotion': 'neutral'},
                    {'speaker': 'B', 'utterance': 'SECRET_FUTURE', 'listener': [{'name': 'A'}], 'emotion': 'neutral'}]}))
                cmd = [sys.executable, str(ROOT/'run.py'), '--dataset', 'mpdd', '--data', str(path),
                       '--model-kind', 'audio-flamingo-3', '--output-format', 'plain', '--max-samples', '1',
                       '--endpoint', f'http://127.0.0.1:{server.server_port}/v1',
                       '--output-dir', str(Path(td)/'out'), '--run-id', 'plain-test']
                completed = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(len(seen), 3)
                for request in seen:
                    self.assertNotIn('response_format', request)
                    self.assertNotIn('SECRET_FUTURE', json.dumps(request))
                    self.assertNotIn('JSON', json.dumps(request))
                for task in TASKS:
                    folder = Path(td)/'out'/task/'mpdd'/'audio-flamingo-3'/'anonymous'/'plain-test'
                    config = json.loads((folder/'config.json').read_text())
                    self.assertEqual(config['output_format'], 'plain')
                    self.assertEqual(config['parser_version'], 'voice-plain-parser-v2-delimiters')
                    row = json.loads((folder/'predictions.jsonl').read_text())
                    self.assertEqual(row['status'], 'ok')
                    self.assertEqual(row['prompt'], seen[TASKS.index(task)]['messages'][0]['content'][0]['text'])
                resumed = subprocess.run(cmd+['--resume'], capture_output=True, text=True)
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertEqual(len(seen), 3)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_structured_http_output_contract(self):
        sample = {'candidates': ['左母', '左父', '左正鵬'], 'gold': ['SECRET'],
                  'history': [], 'naming': 'original'}
        args = SimpleNamespace(audio_manifest=None, condition='text', model_kind='audio-flamingo-3',
                               served_model='voice-model', max_new_tokens=512, temperature=0, seed=42,
                               output_format='json-schema')
        client = Client(args)
        for task, key in zip(TASKS, ('addressees', 'emotion', 'ranking')):
            current = {**sample, 'task': task}
            with patch.object(client, 'request', return_value={}) as request:
                client.complete(current)
            schema = request.call_args.args[1]['response_format']['json_schema']['schema']
            self.assertEqual(schema['required'], [key])
            self.assertNotIn('SECRET', json.dumps(schema))
            self.assertFalse(schema['additionalProperties'])
            if key == 'ranking':
                self.assertEqual(schema['properties'][key]['minItems'], 3)
                self.assertEqual(schema['properties'][key]['items']['enum'], sample['candidates'])
        # Constrained output must still fail semantic checks for duplicated IDs.
        self.assertIsNotNone(parse('{"ranking":["左母","左父","左母"]}', TASKS[2], sample['candidates'])[1])
        args.output_format = 'unconstrained'
        with patch.object(client, 'request', return_value={}) as request:
            client.complete(current)
        self.assertNotIn('response_format', request.call_args.args[1])

    def test_strict_outputs(self):
        self.assertEqual(parse('{"addressees":["P01","P02"]}', TASKS[0], ['P01','P02']), (['P01','P02'],None))
        for text in ['{"ranking":["P01"]}', '{"ranking":["P01","P01"]}', '{"ranking":["P01","X"]}']:
            self.assertIsNotNone(parse(text, TASKS[2], ['P01','P02'])[1])
        self.assertIsNotNone(parse('{"emotion":"neutral","emotion":"angry"}', TASKS[1], [])[1])

    def test_ranking_prompt_candidate_count(self):
        sample = {'task': TASKS[2], 'history': [{'index': 0, 'speaker': 'P01', 'utterance': 'Hello'}],
                  'candidates': ['P01', 'P02', 'P03', 'P04', 'P05']}
        text = prompt(sample)
        self.assertIn('exactly 5 unique IDs', text)
        self.assertIn('P01, P02, P03, P04, P05', text)
        self.assertNotIn('{"ranking": ["P01", "P02", "P03"]}', text)

    def test_ranking_zero_padding(self):
        self.assertEqual(parse('{"ranking":["P2","P3","P1"]}', TASKS[2], ['P01','P02','P03']),
                         (['P02','P03','P01'], None))
        for values in [['P1','P01','P3'], ['P1','P2'], ['P1','P2','P9']]:
            self.assertIsNotNone(parse(json.dumps({'ranking': values}), TASKS[2], ['P01','P02','P03'])[1])

    def test_hand_metrics(self):
        rows = [{'gold':['A','B'], 'prediction':['A'], 'status':'ok'}, {'gold':['C'], 'prediction':None, 'status':'invalid'}]
        m = metrics(rows, TASKS[0])
        self.assertEqual(m['exact_match'], 0)
        self.assertAlmostEqual(m['sample_f1'], 1/3)
        self.assertAlmostEqual(m['micro_f1'], .5)
        rows = [{'gold':'B','prediction':['A','B'], 'status':'ok'}, {'gold':'A','prediction':None,'status':'invalid'}]
        self.assertEqual(metrics(rows,TASKS[2])['mrr'], .25)
        self.assertEqual(metrics([{'gold':None,'prediction':'neutral','status':'ok'}], TASKS[1])['scored_samples'], 0)

    def test_cutoff_and_multilabel(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'data.json'
            path.write_text(json.dumps({'d':[
                {'speaker':'A','utterance':'VISIBLE','listener':[{'name':'B'},{'name':'C'}], 'emotion':'angry'},
                {'speaker':'A','utterance':'SECRET_FUTURE','listener':[{'name':'B'}], 'emotion':'neutral'}]}))
            args = SimpleNamespace(data=path, metadata=None, gold_emotions=None, min_participants=3,
                                   max_dialogues=0,max_samples=0,context_turns=0,dataset='test')
            ar = samples(args,TASKS[0]); er = samples(args,TASKS[1]); ns = samples(args,TASKS[2])
            self.assertEqual(len(ar),2)
            self.assertEqual(ar[0]['gold'], ['P02','P03'])
            self.assertEqual(len(ns),1)
            self.assertEqual(ns[0]['gold'],'P01')  # Same speaker remains a legal answer.
            for s in (ar[0],er[0],ns[0]):
                self.assertNotIn('SECRET_FUTURE', prompt(s))
                self.assertNotIn('listener', prompt(s))
                self.assertNotIn('emotion', s['history'][0])
            self.assertEqual(er[0]['gold'],'angry')
            args.naming = 'original'
            for task, anonymous in zip(TASKS, [ar, er, ns]):
                named = samples(args, task)
                self.assertEqual([s['sample_id'] for s in named], [s['sample_id'] for s in anonymous])
                self.assertEqual(named[0]['candidates'], ['A','B','C'])
                self.assertEqual(named[0]['history'][0]['speaker'], 'A')
                self.assertEqual(named[0]['history'][0]['utterance'], anonymous[0]['history'][0]['utterance'])
                self.assertNotIn('P01', prompt(named[0]))
                self.assertNotIn('SECRET_FUTURE', prompt(named[0]))
            self.assertEqual(samples(args, TASKS[0])[0]['gold'], ['B','C'])
            self.assertEqual(samples(args, TASKS[2])[0]['gold'], 'A')
            self.assertEqual(parse('{"ranking":["B","A","C"]}', TASKS[2], ['A','B','C']), (['B','A','C'],None))
            self.assertEqual(metrics([{'gold':'angry','prediction':None,'status':'invalid'}],TASKS[1])['accuracy'],0)

    def test_http_runner_and_resume(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                data = {'data':[{'id':'voice-model'}]}
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(data).encode())
            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append(req)
                text = req['messages'][0]['content'][0]['text']
                if 'Identify ALL' in text: answer = {'addressees':['P02','P03']}
                elif 'dominant emotion' in text: answer = {'emotion':'neutral'}
                else: answer = {'ranking':['P01','P02','P03']}
                data = {'choices':[{'message':{'content':json.dumps(answer)},'finish_reason':'stop'}]}
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(data).encode())
        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                fixture = Path(td)/'dialogue.json'
                fixture.write_text(json.dumps({'fixture': [
                    {'speaker': 'A', 'utterance': text,
                     'listener': [{'name': 'B'}, {'name': 'C'}]}
                    for text in ('first', 'second', 'third')]}))
                for profile in ['gemma4-e4b','qwen3-instruct']:
                    cmd = [sys.executable,str(ROOT/'run.py'),'--model-kind',profile,'--max-samples','2','--dataset','bazinga','--run-id','test',
                           '--data',str(fixture),
                           '--endpoint',f'http://127.0.0.1:{server.server_port}/v1','--output-dir',str(Path(td)/profile)]
                    first = subprocess.run(cmd,capture_output=True,text=True)
                    self.assertEqual(first.returncode,0,first.stderr)
                    count = len(seen)
                    resumed = subprocess.run(cmd+['--resume'],capture_output=True,text=True)
                    self.assertEqual(resumed.returncode,0,resumed.stderr)
                    self.assertEqual(len(seen),count)
                    conflict = subprocess.run(cmd+['--resume','--context-turns','1'],capture_output=True,text=True)
                    self.assertNotEqual(conflict.returncode,0)
                    for task in TASKS:
                        self.assertEqual(json.loads((Path(td)/profile/task/'bazinga'/profile/'anonymous'/'test'/'metrics.json').read_text())['samples'],2)
                self.assertTrue(all(r['chat_template_kwargs']=={'enable_thinking':False} for r in seen))
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__ == '__main__':
    unittest.main()
