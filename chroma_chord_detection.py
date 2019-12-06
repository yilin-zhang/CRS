import json
import os
import numpy as np
import librosa as lb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import preprocessing
from chord_recognition.chromagram import extract_pitch_chroma, compute_stft, file_read, block_audio
from chord_recognition.feature import chromagram, onset_detection

with open('chord_recognition/chord_templates.json', 'r') as fp:
    templates_json = json.load(fp)

chords = ['N','G maj','G# maj','A maj','A# maj','B maj','C maj','C# maj','D maj','D# maj','E maj','F maj','F# maj','G min','G# min','A min','A# min','B min','C min','C# min','D min','D# min','E min','F min','F# min']
templates = []

block_size = 8192
hop_size = 1024

reference_frequency = 440

def load_model(model_path):
    print('loading_model...')
    model = joblib.load(model_path)

    return model

df = pd.read_csv('./chord_recognition/data - Copy.csv')
df = pd.DataFrame(df)
label = list(df['label'])
le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(label)
model = load_model('./chord_recognition/gnb_mine.pkl')


def chord_detection_baseline(filepath):

    for chord in chords:
        if chord is 'N':
            continue
        templates.append(templates_json[chord])

    fs, x = file_read(filepath)

    if len(x.shape) > 1:
        x = x[:,1]

    xb, t = block_audio(x, block_size, hop_size, fs)

    X, fs = compute_stft(xb, fs, block_size, hop_size)

    chroma = extract_pitch_chroma(X, fs, reference_frequency, 1)


    chroma_template = np.mean(chroma, axis=1)

    """Correlate 12D chroma vector with each of 24 major and minor chords"""
    cor_vec = np.zeros(24)
    for idx in range(24):
        cor_vec[idx] = np.dot(chroma_template, np.array(templates[idx]))
    idx_max_cor = np.argmax(cor_vec)

    idx_chord = int(idx_max_cor + 1)
    chord_name = tuple(chords[idx_chord].split(" "))

    ## Plotting all figures
    # plt.figure(1)
    # notes = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#']
    # plt.xticks(np.arange(12),notes)
    # plt.title('Pitch Class Profile')
    # plt.xlabel('Notes')
    # plt.ylim((0.0,1.0))
    # plt.grid(True)
    # plt.plot(notes, chroma_template)
    # plt.show()

    return chord_name

def chord_detection_improved(filepath):

    for chord in chords:
        if chord is 'N':
            continue
        templates.append(templates_json[chord])

    fs, x = file_read(filepath)

    if len(x.shape) > 1:
        x = x[:,1]

    xb, t = block_audio(x, block_size, hop_size, fs)

    X, fs = compute_stft(xb, fs, block_size, hop_size)

    chroma = extract_pitch_chroma(X, fs, reference_frequency, 1)


    chroma_template = np.mean(chroma, axis=1)

    """Correlate 12D chroma vector with each of 24 major and minor chords"""
    cor_vec = np.zeros(24)
    for idx in range(24):
        cor_vec[idx] = np.dot(chroma_template, np.array(templates[idx]))
    idx_max_cor = np.argmax(cor_vec)

    idx_chord = int(idx_max_cor + 1)
    chord_name = tuple(chords[idx_chord].split(" "))

    ## Plotting all figures
    # plt.figure(1)
    # notes = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#']
    # plt.xticks(np.arange(12),notes)
    # plt.title('Pitch Class Profile')
    # plt.xlabel('Notes')
    # plt.ylim((0.0,1.0))
    # plt.grid(True)
    # plt.plot(notes, chroma_template)
    # plt.show()

    return chord_name


if __name__ == "__main__":
    count = 0
    chord = chord_detection_baseline("./cache/chord-A.wav")
    print("Estimated: " + str(chord) + "    |    " + "Ground Truth: " + str(('A', 'maj')))
    est = str(chord[0]) + " " + str(chord[1])

    # chord_detection_baseline("./single-chord-dataset/G maj/Grand Piano - Fazioli - major G middle.wav")
