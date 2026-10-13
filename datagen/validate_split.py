#!/usr/bin/env python3
"""Verify complete partition, no false adjacency, and exact multi-label preservation."""
import json
from collections import defaultdict
from split_dialogues import BASE, OUT, reasons, runs
from prepare_data import SHOWS, digest, write_json


def load(path):
    return json.loads(path.read_text())


def main():
    source, source_meta = load(BASE/'dialogue.json'), load(BASE/'metadata.json')
    clean, meta, removed = load(OUT/'dialogue.json'), load(OUT/'metadata.json'), load(OUT/'removed_utterances.json')
    config = load(OUT/'config.json')
    assert config['source_dialogue_sha256'] == digest(BASE/'dialogue.json')
    assert config['source_metadata_sha256'] == digest(BASE/'metadata.json')
    assert clean.keys() == meta.keys()
    retained_positions, removed_positions = set(), set()
    for row in removed:
        pos = row['episode_id'], row['source_turn_index']
        assert pos not in removed_positions
        removed_positions.add(pos)
        assert row['reasons'] == reasons(source[pos[0]][pos[1]])
        assert row['reasons']
    for key, turns in clean.items():
        m = meta[key]
        episode, start, end = m['episode_id'], m['source_turn_start'], m['source_turn_end']
        assert turns and len(turns) == len(m['utterances']) == end-start+1
        assert start == 0 or reasons(source[episode][start-1])
        assert end == len(source[episode])-1 or reasons(source[episode][end+1])
        for j, turn in enumerate(turns):
            i = start+j
            assert not reasons(turn)
            assert turn == source[episode][i]  # Includes every multi-label listener, in order.
            assert (episode, i) not in retained_positions
            retained_positions.add((episode, i))
            assert m['utterances'][j] == {**source_meta[episode]['utterances'][i], 'source_turn_index': i, 'turn_index': j}
    expected = {(e, i) for e, turns in source.items() for i in range(len(turns))}
    assert retained_positions.isdisjoint(removed_positions)
    assert retained_positions | removed_positions == expected
    combined = {}
    for show in SHOWS:
        combined.update(load(OUT/show/'dialogue.json'))
    assert combined == clean
    # Hand-built cases cover consecutive/leading/trailing removals and mixed labels.
    valid = {'speaker': 'a', 'utterance': 'hello', 'listener': [{'name': 'b'}, {'name': 'c'}]}
    bad = {**valid, 'listener': []}
    mixed = {**valid, 'listener': [{'name': 'b'}, {'name': 'multiple_persons'}]}
    assert runs([bad, valid, valid, bad, bad, valid, bad]) == [[1, 2], [5]]
    assert runs([valid, mixed, valid]) == [[0], [2]]
    assert runs([bad, bad]) == []
    assert runs([valid, valid]) == [[0, 1]]
    assert reasons({**valid, 'speaker': 'not_available'}) == ['invalid_speaker']
    assert not reasons(valid)
    report = {'passed': True, 'dialogues': len(clean), 'retained_utterances': len(retained_positions),
              'removed_utterances': len(removed_positions),
              'checks': ['source hashes match', 'complete disjoint retained/removed partition',
                         'all segments are maximal contiguous source runs', 'no bridging over deleted utterances',
                         'retained text and multi-label lists unchanged', 'per-series outputs match combined output',
                         'stable utterance provenance', 'hand-built boundary and mixed-label cases']}
    write_json(OUT/'validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
