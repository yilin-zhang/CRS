import os
import json
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
import librosa
from scipy.io.wavfile import read

# This function reads audio file and outputs a normalized audio array
def audio_read(audio):
    fs, x = read(audio)

    if len(x.shape) > 1:
        y = x[:, 0]

    if x.dtype == 'uint8':
        y = (y / 128.) - 1
    else:
        bits = x.dtype.itemsize * 8
        y = y / (2 ** (bits - 1))
    return y


def compute_stft(audio, block_size, hop_size):
    # Computation of STFT
    N = block_size
    H = hop_size
    w = scipy.signal.get_window('hann', N)
    X = librosa.stft(y, n_fft=N, hop_length=H, win_length=N, window=w, pad_mode='constant')
    t = librosa.frames_to_time(np.arange(X.shape[1]), sr=fs, hop_length=H, n_fft=N)
    return X, t

def F_pitch(p, pitch_ref=69, freq_ref=440):
    """Computes the center frequency/ies of a MIDI pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p: MIDI pitch value(s)
        pitch_ref: Reference pitch (default: 69)
        freq_ref: Frequency of reference pitch (default: 440.0)

    Returns:
        im: Frequency value(s)
    """
    return 2 ** ((p - pitch_ref) / 12) * freq_ref


def F_coef(k, Fs, N):
    """Computes the center frequency/ies of a Fourier coefficient

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        k: Fourier coefficients
        Fs: Sampling rate
        N: Window size of Fourier fransform

    Returns:
        im: Frequency value(s)
    """
    return k * Fs / N


def P(p, Fs, N, pitch_ref=69, freq_ref=440):
    """Computes the set of frequency indices that are assigned to a given pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p: MIDI pitch value
        Fs: Sampling rate
        N: Window size of Fourier fransform
        pitch_ref: Reference pitch (default: 69)
        freq_ref:  Frequency of reference pitch (default: 440.0)

    Returns:
        im: Set of frequency indices
    """
    lower = F_pitch(p - 0.5, pitch_ref, freq_ref)
    upper = F_pitch(p + 0.5, pitch_ref, freq_ref)
    k = np.arange(N // 2 + 1)
    k_freq = F_coef(k, Fs, N)
    mask = np.logical_and(lower <= k_freq, k_freq < upper)
    return k[mask]


def compute_Y_LF(Y, Fs, N):
    """Computes a log-frequency spectrogram

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        Y: Magnitude or power spectrogram
        Fs: Sampling rate
        N: Window size of Fourier fransform
        pitch_ref: Reference pitch (default: 69)
        freq_ref: Frequency of reference pitch (default: 440.0)

    Returns:
        Y_LF: Log-frequency spectrogram
        F_coef_pitch: Pitch values
    """
    Y_LF = np.zeros((128, Y.shape[1]))
    for p in range(128):
        k = P(p, Fs, N)
        Y_LF[p, :] = Y[k, :].sum(axis=0)
    F_coef_pitch = np.arange(128)
    return Y_LF, F_coef_pitch


def chromgram(Y_LF):
    """Computes a chromagram

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        Y_LF: Log-frequency spectrogram

    Returns:
        C: Chromagram
    """
    C = np.zeros((12, Y_LF.shape[1]))
    p = np.arange(128)
    for c in range(12):
        mask = (p % 12) == c
        C[c, :] = Y_LF[mask, :].sum(axis=0)
    return C


# This is the caller function that takes in audio and outputs Chromagram
def compute_chromagram(audio, block_size, hop_size):
    X, timestamp = compute_stft(audio, block_size, hop_size)
    Y = np.abs(X) ** 2
    Y_LF, F_coef_pitch = compute_Y_LF(Y, fs, N)
    C = compute_chromgram(Y_LF)
    return C


compute_chromagram('')

# This is the template matching based classifier

with open('chord_templates.json', 'r') as fp:
    templates_json = json.load(fp)

chords = ['N','G','G#','A','A#','B','C','C#','D','D#','E','F','F#','Gm','G#m','Am','A#m','Bm','Cm','C#m','Dm','D#m','Em','Fm','F#m']
templates = []

for chord in chords:
    if chord is 'N':
        continue
    templates.append(templates_json[chord])

nFrames =C.shape[1]

id_chord = np.zeros(nFrames, dtype='int32')
timestamp = np.zeros(nFrames)
max_cor = np.zeros(nFrames)

for n in range(nFrames):
    """Correlate 12D chroma vector with each of 24 major and minor chords"""
    cor_vec = np.zeros(24)
    for ni in range(24):
        cor_vec[ni] = np.dot(C[:,n], np.array(templates[ni]))
    max_cor[n] = np.max(cor_vec)
    id_chord[n] = np.argmax(cor_vec) + 1

print(id_chord)

#if max_cor[n] < threshold, then no chord is played
#might need to change threshold value
id_chord[np.where(max_cor < 0.8*np.max(max_cor))] = 0
# for n in range(nFrames):
	# print(timestamp[n],chords[id_chord[n]])


#Plotting all figures
plt.figure(1)
notes = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#']
plt.xticks(np.arange(12),notes)
plt.title('Pitch Class Profile')
plt.xlabel('Note')
plt.grid(True)

plt.figure(2)
plt.yticks(np.arange(25), chords)
plt.plot(t, id_chord)
plt.xlabel('Time in seconds')
plt.ylabel('Chords')
plt.title('Identified chords')
plt.grid(True)
plt.show()
