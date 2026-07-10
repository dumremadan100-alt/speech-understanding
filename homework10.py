import numpy as np
import torch
import torch.nn as nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    '''
    # Pre-emphasis
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # Spectrogram parameters
    frame_length = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)           # 2 ms

    features = []

    for start in range(0, len(waveform) - frame_length + 1, step):
        frame = waveform[start:start + frame_length]
        spectrum = np.abs(np.fft.fft(frame))
        features.append(spectrum[:len(spectrum)//2])

    features = np.array(features)

    # Simple labels (5 frames per label)
    labels = np.arange(len(features)) // 5

    return features, labels


def train_neuralnet(features, labels, iterations):
    '''
    Train a neural network.
    '''
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    nfeats = features.shape[1]
    nlabels = int(np.max(labels)) + 1

    model = nn.Sequential(
        nn.LayerNorm(nfeats),
        nn.Linear(nfeats, nlabels)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    lossvalues = np.zeros(iterations)

    for i in range(iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        lossvalues[i] = loss.item()

    return model, lossvalues


def test_neuralnet(model, features):
    '''
    Test the neural network.
    '''
    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1).detach().numpy()

    return probabilities