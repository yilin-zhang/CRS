import librosa as lb
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob
import csv
import pandas as pd

def onset_detection(wave, sr, block_length=512, show=False):
    #############################################
    # Step 1: onset detection
    # Input: 
    #   wave: input signal
    #   sr: sample rate
    #   show: show figure and print results
    # Output:
    #   wave_peak: onset positions 
    #############################################

    # RMS
    wave_RMS = lb.feature.rms(y=wave, frame_length=block_length, hop_length=block_length)
    # db
    wave_RMS_db = lb.core.amplitude_to_db(wave_RMS.ravel())
    # LPF
    B, A = signal.butter(2, 0.1)
    wave_RMS_db = signal.filtfilt(B, A, wave_RMS_db)
    # Differential
    wave_onset = np.array([wave_RMS_db[i+1]-wave_RMS_db[i] for i in range(len(wave_RMS_db)-1)])
    for i in range(len(wave_onset)):
        if wave_onset[i] < 0.5:
            wave_onset[i] = 0
    wave_peak = lb.util.peak_pick(wave_onset, 50,50,10,10,0,50)

    if show:
        fig1 = plt.figure()
        ax = fig1.add_subplot(311)
        ax.plot(wave)
        bx = fig1.add_subplot(312)
        bx.plot(wave_RMS_db)
        cx = fig1.add_subplot(313)
        cx.plot(wave_onset)
        plt.show()

    return wave_peak


def chromagram(wave, sr, wave_peak, block_length=512, slice_len=1024*8, show=True):
    #############################################
    # Step 2: chromagram
    #############################################
    chromagrams = []
    offset = 200 #in ms
    offset_index = round(sr/(1000/offset))
    wave_peak = wave_peak * block_length
    real_peak = []
    nsum = 4 # add four frames of chroma together as one output

    i=0
    slice_matrix = np.zeros((len(wave[wave_peak[0]+offset_index:wave_peak[0]+offset_index+slice_len]), len(wave_peak)))
    for peak in wave_peak:
        slice = wave[peak+offset_index:peak+offset_index+slice_len]
        slice_matrix[:,i] = slice
        if len(slice) < slice_len:
            continue
        slice_rms = np.sqrt(np.mean(np.square(slice)))
        if slice_rms < 0.01:
            continue
        # lb.output.write_wav('test{}.wav'.format(i),slice, sr)
        PCP = lb.feature.chroma_cqt(y=slice, sr=sr)
        for j in range(PCP.shape[1]-nsum+1):
            chromagrams.append(np.sum(PCP[:,j:j+nsum],axis=1))
        real_peak.append(peak)
        i+=1
        #print('PCP of slice {} finished'.format(i))

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax.plot(wave)
        bx = fig.add_subplot(312)   
        bx.plot([1 if i in real_peak else 0 for i in range(len(wave))])
        cx = fig.add_subplot(313)
        cx.bar(range(12), chromagrams[1])
        plt.xticks(np.arange(12),('C ','C#','D ','D#','E ','F ','F#','G ','G#','A ','A#','B '))
        print('Number of Peaks: {}'.format(len(wave_peak)))
        print('Number of PCPs: {}'.format(np.array(chromagrams).shape))
        plt.show()

    return slice_matrix, chromagrams


def calculate_PCP(wave, sr, fft_len=2048, show=False):
    fmin = 55
    octave = 5

    _, _, spectrum = signal.spectrogram(wave, nfft=fft_len)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(spectrum)

        plt.show()


def extract_feature(data_dir):
    # data = [] # element: tuple (label, feature)
    data = pd.DataFrame(columns=['label','C ','C#','D ','D#','E ','F ','F#','G ','G#','A ','A#','B '])
    # dir = './chord audio/*.wav'
    dir = 'C:/Users/bhxxl/Desktop/Project - Copy - Copy/single-chord-dataset 2/single-chord-dataset/Gm/*.wav'
    file_list = glob.glob(dir)
    for file in file_list:
        # label = os.path.split(file)[1].split('.wav')[0]
        label = 'Gm'
        print('Extracting: {}'.format(label))
        wave, sr= lb.load(file, sr=44100)
        wave_peak = onset_detection(wave, sr, show=False)
        PCP = chromagram(wave, sr, wave_peak, show=False)
        for feature in PCP:
            # sample = feature.tolist()
            # sample.append(label)
            # data.append(sample)
            data = data.append(pd.DataFrame({'label':label,'C ':feature[0],'C#':feature[1],
                                            'D ':feature[2],'D#':feature[3],'E ':feature[4],
                                            'F ':feature[5],'F#':feature[6],'G ':feature[7],
                                            'G#':feature[8],'A ':feature[9],'A#':feature[10],'B ':feature[11]}, index=['c']))
    # with open(data_dir, "w") as f:
    #     wr = csv.writer(f)
    #     wr.writerow(data)
    data.to_csv(data_dir, index=False)
    print('\n')
    print('Success! Data saved as {}'.format(data_dir))




if __name__ == "__main__":
    extract_feature('./dataGm.csv')