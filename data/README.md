# Local datasets

Data is generated locally and ignored by Git. Obtain MPDD separately and
use the root README's preprocessing commands. Bazinga downloads are pinned
to revision `96db30fff0964b6fa1ae9e3c713a1e2f42081360`.

Normalized dialogue JSON maps dialogue IDs to lists of turns:

```json
{"dialogue-id": [{"speaker": "Alice", "utterance": "Hello.",
                   "listener": [{"name": "Bob"}]}]}
```

MPDD keeps its original annotated copy in `mpdd/dialogue_chinese.json` and
writes normalized turns to `mpdd/dialogue.json`. Emotion and relation are
removed only from the normalized copy. `provenance.json` records the source
path and checksum; validation compares the original bytes and normalized turns.

Bazinga's `dialogue.json` preserves episode context; `metadata.json` tracks
stable utterance IDs, word-row ranges, speaker/addressee status, and alignment
timestamps. `clean/dialogue.json` removes invalid speaker/addressee spans
and splits at removed spans without connecting nonadjacent turns. Multiple
listeners are preserved. Splits do not infer true scene boundaries.

Reference review creates the separate `bazinga/reference_qc_v1/` subset.
For paired text/audio experiments, select this same dataset with `--data`
and use its adjacent metadata for every condition. Official timestamps are
silver alignments; reference quality requires review.
