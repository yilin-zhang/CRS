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

block_size = 4096
hop_size = 256

reference_frequency = 440

def load_model(model_path):
    print('loading_model...')
    model = joblib.load(model_path)

    return model

df = pd.read_csv('./data.csv')
df = pd.DataFrame(df)
label = list(df['label'])
le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(label)
model = load_model('./gnb_mine.pkl')


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

    chroma = extract_pitch_chroma(X, fs, reference_frequency)

    chroma_template = np.mean(chroma, axis=1)

    """Correlate 12D chroma vector with each of 24 major and minor chords"""
    cor_vec = np.zeros(24)
    for idx in range(24):
        cor_vec[idx] = np.dot(chroma_template, np.array(templates[idx]))
    idx_max_cor = np.argmax(cor_vec)

    idx_chord = int(idx_max_cor + 1)
    chord_name = tuple(chords[idx_chord].split(" "))

    # Plotting all figures
    #plt.figure(1)
    #notes = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#']
    #plt.xticks(np.arange(12),notes)
    #plt.title('Pitch Class Profile')
    #plt.xlabel('Note')
    #plt.grid(True)
    #plt.plot(chroma, notes)
    #plt.show()

    # plt.figure(2)
    # plt.yticks(np.arange(25), chords)
    # plt.plot(t, idx_chord)
    # plt.xlabel('Time in seconds')
    # plt.ylabel('Chords')
    # plt.title('Identified chords')
    # plt.grid(True)
    # plt.show()

    print(chord_name)

    return chord_name

def chord_detection_improved(filepath):

    x, fs = lb.load(filepath, sr=None)
    wave_peak = onset_detection(x, fs)
    xb, _ = chromagram(x, fs, wave_peak)
    xb = xb.T

    X, fs = compute_stft(xb, fs, block_size, hop_size)

    chroma = extract_pitch_chroma(X, fs, reference_frequency)

    chroma = chroma.T

    print(chroma)

    chord_name = le.inverse_transform(model.predict(chroma))

    print(chord_name)

    return chord_name

if __name__ == "__main__":
    for file in os.listdir("../../Project/chord-detection-prediction/test_chords"):
        print(file)
        if file.endswith(".wav"):
            chord_detection_improved("../../Project/chord-detection-prediction/test_chords/" + file)