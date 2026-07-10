import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord.
    '''
    N = int(0.5 * Fs)
    t = np.arange(N) / Fs

    # Root, major third, and perfect fifth
    f1 = f
    f2 = f * (2 ** (4 / 12))
    f3 = f * (2 ** (7 / 12))

    x = (np.cos(2 * np.pi * f1 * t) +
         np.cos(2 * np.pi * f2 * t) +
         np.cos(2 * np.pi * f3 * t))

    return x


def dft_matrix(N):
    '''
    Create an NxN DFT transform matrix.
    '''
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    X = np.fft.fft(x)
    magnitude = np.abs(X)

    # Only use the positive-frequency half
    magnitude = magnitude[:len(x)//2]
    freqs = np.fft.fftfreq(len(x), d=1/Fs)[:len(x)//2]

    # Indices of the three largest magnitudes
    idx = np.argsort(magnitude)[-3:]

    # Corresponding frequencies
    loudest = np.sort(freqs[idx])

    return loudest[0], loudest[1], loudest[2]