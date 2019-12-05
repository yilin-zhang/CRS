import json
import os
import numpy as np
import matplotlib.pyplot as plt
from chord_recognition.chromagram import extract_pitch_chroma
from chord_recognition.chromagram import compute_stft

with open('chord_recognition/chord_templates.json', 'r') as fp:
    templates_json = json.load(fp)

chords = ['N','G maj','G# maj','A maj','A# maj','B maj','C maj','C# maj','D maj','D# maj','E maj','F maj','F# maj','G min','G# min','A min','A# min','B min','C min','C# min','D min','D# min','E min','F min','F# min']
templates = []

def chord_detection(filepath):

    for chord in chords:
        if chord is 'N':
            continue
        templates.append(templates_json[chord])

    X, fs, t = compute_stft(filepath)

    chroma = extract_pitch_chroma(X, fs, 440)

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


if __name__ == "__main__":

    for file in os.listdir("../../Project/chord-detection-prediction/test_chords"):
        print(file)
        if file.endswith(".wav"):
            chord_detection("../../Project/chord-detection-prediction/test_chords/" + file)