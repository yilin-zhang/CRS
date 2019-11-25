import matplotlib.pyplot as plt
import numpy as np
import math
import scipy as sp
from scipy.io.wavfile import read


block_size = 4096
hop_size = 256


def block_audio(x, blockSize, hopSize, fs):
    """
    Sample audio blocking code from Alex
    """
    # allocate memory
    numBlocks = int(np.ceil(x.size / hopSize))
    xb = np.zeros([numBlocks, blockSize])

    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return xb, t

def compute_hann(iWindowLength):
    """
    Sample compute hann window code from Alex
    """
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))


def compute_stft(filepath):
    fs, x = read(filepath)

    if len(x.shape) > 1:
        x = x[:,1]

    xb, t = block_audio(x, block_size, hop_size, fs)

    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1] / 2 + 1), numBlocks])

    for n in range(0, numBlocks):
        # apply window
        tmp = abs(sp.fft(xb[n, :] * afWindow)) * 2 / xb.shape[1]

        # compute magnitude spectrum
        X[:, n] = tmp[range(math.ceil(tmp.size / 2 + 1))]
        X[[0, math.ceil(tmp.size / 2)], n] = X[[0, math.ceil(tmp.size / 2)], n] / np.sqrt(
            2)  # let's be pedantic about normalization

    return X, fs, t

def extract_pitch_chroma(X, fs, tfInHz):
   block_length = X.shape[0]
   num_blocks = X.shape[1]
   pitch_chroma = np.zeros((12, block_length), dtype=np.int16)

   Y = np.abs(X) ** 2

   # Need to calculate pitch chroma from C3 to B5 --> 48 to 83
   lower_bound = 48
   upper_bound = 84

   k = np.arange(1, (block_length+1))
   k_freq = k * fs / (2 * (block_length-1))

   irange = (upper_bound-lower_bound)

   logfreq_X = np.zeros([irange, num_blocks])

   for n, i in enumerate(range(lower_bound, upper_bound)):

       midi_pitch_lower = 2 ** (((i - 0.5) - 69) / 12) * tfInHz
       midi_pitch_upper = 2 ** (((i + 0.5) - 69) / 12) * tfInHz

       mask = np.logical_and(midi_pitch_lower <= k_freq, k_freq < midi_pitch_upper)

       logfreq_X[n, :] = Y[k[mask], :].sum(axis=0)


   pitch_chroma = np.zeros((12, logfreq_X.shape[1]))
   p = np.arange(48,84)
   for c in range(12):
       mask = (p % 12) == c
       pitch_chroma[c, :] = logfreq_X[mask, :].sum(axis=0)

   idx = [7,8,9,10,11,0,1,2,3,4,5,6]
   # idx = [8,7,6,5,4,3,2,1,0,11,10,9]

   pitch_chroma = pitch_chroma[idx, :]

   l2norm = np.linalg.norm(pitch_chroma, ord=2, axis=0)
   l2norm[l2norm == 0] = 1
   pitch_chroma /= l2norm

   return pitch_chroma
