import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    '''
    frame_length = int(0.025 * Fs)   # 25 ms
    step = int(0.010 * Fs)           # 10 ms

    energies = []
    starts = []

    for start in range(0, len(waveform) - frame_length + 1, step):
        frame = waveform[start:start + frame_length]
        energies.append(np.sum(frame ** 2))
        starts.append(start)

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    segments = []
    for i, energy in enumerate(energies):
        if energy > threshold:
            start = starts[i]
            segments.append(waveform[start:start + frame_length])

    return segments


def segments_to_models(segments, Fs):
    '''
    Create an average log spectrum for each speech segment.
    '''
    models = []

    frame_length = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)           # 2 ms

    for segment in segments:
        # Pre-emphasis
        pre = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])

        spectra = []

        for start in range(0, len(pre) - frame_length + 1, step):
            frame = pre[start:start + frame_length]
            spectrum = np.abs(np.fft.fft(frame))
            spectrum = spectrum[:len(spectrum)//2]
            spectra.append(20 * np.log10(np.maximum(spectrum, 1e-6)))

        models.append(np.mean(spectra, axis=0))

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize speech using cosine similarity.
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    sims = np.zeros((len(models), len(test_models)))
    test_outputs = []

    for j, test in enumerate(test_models):
        best = -1
        best_label = ""

        for i, model in enumerate(models):
            sim = np.dot(model, test) / (np.linalg.norm(model) * np.linalg.norm(test))
            sims[i, j] = sim

            if sim > best:
                best = sim
                best_label = labels[i]

        test_outputs.append(best_label)

    return sims, test_outputs