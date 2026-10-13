#!/usr/bin/env python3
"""Check schema, provenance, full token coverage, labels, and MPDD copy integrity."""
import json
from pathlib import Path
from prepare_data import ROOT, VOICE, SHOWS, digest, read_rows, detokenize, listeners


def load(path):
    return json.loads(path.read_text())


def main():
    base = VOICE/'data/bazinga'
    data, meta = load(base/'dialogue.json'), load(base/'metadata.json')
    inventory = load(base/'episode_inventory.json')
    assert set(data) == set(meta) == {r['episode_id'] for r in inventory if r['selected']}
    assert len(inventory) == len({r['episode_id'] for r in inventory})
    total_tokens = 0
    combined = {}
    for show in SHOWS:
        combined.update(load(base/show/'dialogue.json'))
    assert combined == data
    utterance_ids = set()
    for episode, turns in data.items():
        m = meta[episode]
        path = ROOT/'raw'/m['source_path']
        assert digest(path) == m['source_sha256']
        rows = read_rows(path)
        cursor = 0
        assert len(turns) == len(m['utterances'])
        for i, (turn, provenance) in enumerate(zip(turns, m['utterances'])):
            assert set(turn) == {'speaker', 'utterance', 'listener'}
            assert all(set(x) == {'name'} for x in turn['listener'])
            assert turn['utterance'].strip()
            assert provenance['turn_index'] == i
            assert provenance['utterance_id'] not in utterance_ids
            utterance_ids.add(provenance['utterance_id'])
            chunk = rows[cursor:cursor+provenance['token_count']]
            cursor += len(chunk)
            assert chunk[0]['line'] == provenance['source_line_start']
            assert chunk[-1]['line'] == provenance['source_line_end']
            assert all(r['speaker'] == turn['speaker'] and r['addressee'] == provenance['raw_addressee'] for r in chunk)
            assert turn['utterance'] == detokenize([r['token'] for r in chunk])
            assert turn['listener'] == listeners(provenance['raw_addressee'])
        assert cursor == len(rows)
        total_tokens += cursor
    provenance = load(VOICE/'data/mpdd/provenance.json')
    original = Path(provenance['source'])
    assert digest(original) == provenance['sha256']
    assert original.read_bytes() == (VOICE/'data/mpdd/dialogue_chinese.json').read_bytes()
    mpdd = load(original)
    normalized = load(VOICE/'data/mpdd/dialogue.json')
    assert mpdd.keys() == normalized.keys()
    for key in mpdd:
        assert len(mpdd[key]) == len(normalized[key])
        for raw, clean in zip(mpdd[key], normalized[key]):
            assert clean == {'speaker': raw['speaker'], 'utterance': raw['utterance'], 'listener': [{'name': x['name']} for x in raw['listener']]}
    assert listeners('a@b') == [{'name': 'a'}, {'name': 'b'}]
    assert listeners('_') == []
    assert listeners('multiple_persons') == [{'name': 'multiple_persons'}]
    assert detokenize(['I', "'", 've', 'seen', 'it', '.']) == "I've seen it."
    report = {'passed': True, 'scanned_episodes': len(inventory), 'selected_episodes': len(data),
              'bazinga_utterances': len(utterance_ids), 'bazinga_tokens_accounted_for': total_tokens,
              'mpdd_dialogues': len(mpdd), 'mpdd_utterances': sum(map(len, mpdd.values())),
              'checks': ['exact schema', 'per-series and combined equality', 'source hashes',
                         'no lost or duplicated source tokens', 'speaker/addressee boundaries',
                         'multi-addressee preservation', 'stable unique utterance IDs', 'MPDD byte-identical copy and normalized equality']}
    (base/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
