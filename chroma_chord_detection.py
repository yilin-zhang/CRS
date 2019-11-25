import json
import numpy as np
import matplotlib.pyplot as plt
from chromagram import extract_pitch_chroma
from chromagram import compute_stft

with open('chord_templates.json', 'r') as fp:
    templates_json = json.load(fp)

chords = ['N','G','G#','A','A#','B','C','C#','D','D#','E','F','F#','Gm','G#m','Am','A#m','Bm','Cm','C#m','Dm','D#m','Em','Fm','F#m']
templates = []

def chord_detection(filepath):

    for chord in chords:
        if chord is 'N':
            continue
        templates.append(templates_json[chord])

    X, fs, t = compute_stft(filepath)

    chroma = extract_pitch_chroma(X, fs, 440)

    nFrames = chroma.shape[1]

    id_chord = np.zeros(nFrames, dtype='int32')
    # timestamp = np.zeros(nFrames)
    max_cor = np.zeros(nFrames)

    for n in range(nFrames):
        """Correlate 12D chroma vector with each of 24 major and minor chords"""
        cor_vec = np.zeros(24)
        for ni in range(24):
            cor_vec[ni] = np.dot(chroma[:,n], np.array(templates[ni]))
        max_cor[n] = np.max(cor_vec)
        id_chord[n] = np.argmax(cor_vec) + 1

    # print(max_cor)


    #if max_cor[n] < threshold, then no chord is played
    #might need to change threshold value
    id_chord[np.where(max_cor < 0.8*np.max(max_cor))] = 0
    # for n in range(nFrames):
    # 	print(timestamp[n],chords[id_chord[n]])


    #Plotting all figures
    plt.figure(1)
    notes = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#']
    plt.xticks(np.arange(12),notes)
    plt.title('Pitch Class Profile')
    plt.xlabel('Note')
    plt.grid(True)
    plt.plot(chroma, notes)
    # plt.show()

    plt.figure(2)
    plt.yticks(np.arange(25), chords)
    plt.plot(t, id_chord)
    plt.xlabel('Time in seconds')
    plt.ylabel('Chords')
    plt.title('Identified chords')
    plt.grid(True)
    plt.show()

    return


if __name__ == "__main__":
    chord_detection("./Gmaj.wav")