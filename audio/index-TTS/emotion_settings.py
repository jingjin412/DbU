"""Fixed DbU TTS controls shared by planning and synthesis."""
STRENGTH = 0.6
EMO_ALPHA = 1.0
ORDER = ['happiness', 'angry', 'sadness', 'fear', 'disgust',
         'melancholic', 'surprise', 'neutral']
LABELS = [label for label in ORDER if label != 'melancholic']
STRENGTHS = dict.fromkeys(LABELS, STRENGTH)


def validate_jobs(jobs, order=ORDER):
    if list(order) != ORDER:
        raise ValueError('Unexpected emotion vector order; rebuild the plan')
    for job in jobs:
        label = job.get('emotion')
        if label not in LABELS:
            raise ValueError('Unknown DbU emotion label')
        expected = [STRENGTH if item == label else 0.0 for item in order]
        if job.get('emotion_vector') != expected or job.get('emo_alpha') != EMO_ALPHA:
            raise ValueError('DbU requires strength 0.6 for neutral and all emotions, '
                             'with emo_alpha=1.0; rebuild the plan')
