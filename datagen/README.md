# Preprocessing

Run commands from the root README. These scripts use Python's standard library:

1. `fetch_bazinga.py`: download pinned word transcripts and character metadata.
2. `prepare_data.py --mpdd-source <directory>`: normalize Bazinga and MPDD,
   preserving source provenance and stable utterance IDs.
3. `validate_data.py`: verify source checksums, token coverage, schema,
   multiple listeners, and MPDD byte-identical copy.
4. `split_dialogues.py`: remove invalid annotations and split at removed spans.
5. `validate_split.py`: verify clean turn IDs, source order, and split boundaries.

`raw/` is a local download cache; outputs are under `data/`. Both are ignored.
No results analysis is performed here.
