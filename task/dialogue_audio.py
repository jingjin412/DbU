"""Causal transport for timestamp-trimmed continuous dialogue audio."""
from pathlib import Path

LAYOUT = 'bazinga-trimmed-dialogue-prefix-v1'


def keep_ranges(turns, rate, cutoff, source_frames, padding=.2, gap_threshold=2.):
    """Keep short pauses; replace longer gaps by the two boundary margins."""
    margin = round(padding * rate)
    threshold = round(gap_threshold * rate)
    ranges = []
    previous_end = None
    for turn in turns:
        start, end = round(turn['start_time'] * rate), round(turn['end_time'] * rate)
        a, b = max(0, start-margin), min(source_frames, cutoff, end+margin)
        if b <= a:
            continue
        if ranges and (a <= ranges[-1][1] or start - previous_end <= threshold):
            ranges[-1][1] = max(ranges[-1][1], b)
        else:
            ranges.append([a, b])
        previous_end = max(previous_end if previous_end is not None else end, end)
    return ranges


def mapped_frame(frame, ranges):
    """Map a source boundary to the concatenated kept PCM frame position."""
    total = 0
    for a, b in ranges:
        if frame <= a:
            return total
        total += min(frame, b)-a
        if frame <= b:
            return total
    return total


def prefix_entry(client, sample):
    if client.args.audio_scope != 'history' or sample['context_start'] != 0:
        raise ValueError('Dialogue prefix requires full visible history')
    turn = sample['history'][-1]
    entry = client.audio[turn['utterance_id']]
    if entry.get('audio_layout') != LAYOUT:
        raise ValueError('Missing/mismatched dialogue-prefix layout')
    expected = [t['utterance_id'] for t in sample['history']]
    if entry['covered_utterance_ids'] != expected:
        raise ValueError('Dialogue prefix history IDs differ')
    return turn, entry


def prefix_clips(client, sample):
    import hashlib
    import soundfile as sf
    turn, entry = prefix_entry(client, sample)
    path = Path(entry['path'])
    if not path.is_absolute():
        path = client.args.audio_manifest.parent / path
    if path not in client.audio_info:
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['audio_sha256']:
            raise ValueError('Dialogue PCM asset checksum changed')
        client.audio_info[path] = sf.info(path)
    info = client.audio_info[path]
    frames = entry['prefix_frames']
    if not isinstance(frames, int) or not 0 < frames <= info.frames or info.samplerate != entry['sample_rate']:
        raise ValueError('Invalid dialogue prefix boundary')
    seconds = 30 if client.args.model_kind == 'gemma4-e4b' else 300
    step = round(seconds * info.samplerate)
    clips = [(turn, path, a, min(a+step, frames)) for a in range(0, frames, step)]
    cap = min(client.args.max_audio_clips, 5) if client.args.model_kind == 'voxtral-small-24b' else client.args.max_audio_clips
    if len(clips) > cap:
        raise ValueError('Dialogue prefix exceeds audio item cap')
    return clips


def prefix_label(client, sample, group):
    _, entry = prefix_entry(client, sample)
    rate = entry['sample_rate']
    begin, end = group[0][2], group[-1][3]
    lines = [f'Visible dialogue-prefix audio segment: {begin/rate:.6f}-{end/rate:.6f}s '
             '(times below relative to the full trimmed prefix).']
    for item in entry['timeline']:
        if item['end_frame'] >= begin and item['start_frame'] <= end:
            lines.append(f"utterance {item['index']+1}, speaker {item['speaker']}: "
                         f"{item['start_frame']/rate:.6f}-{item['end_frame']/rate:.6f}s")
    return '\n'.join(lines)
