import numpy as np
from sklearn.externals import joblib
import sounddevice as sd
import librosa as lb
import matplotlib.pyplot as plt
import drawnow
import pandas as pd
import feature
from collections import Counter
import timeit

def load_model(model_path):
    print('loading_model...')
    model = joblib.load(model_path)

    return model

# print(sd.query_devices())
df = pd.read_csv('./data/data.csv')
df = pd.DataFrame(df)
label = list(df['label'])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(label)
model = load_model('./model/gnb_mine.pkl')




def realtime_recognition(model, le, wav_file_path):
    SR = 44100
    BLOCK_LEN = 2048
    CHANNELS = 1
    DEVICE = 3
    # s = sd.Stream(samplerate=SR,blocksize=BLOCK_LEN)
    # s.start()
    # drawnow.figure(figsize=(4, 4))
    # plt.figure()
    def draw_fig():
        plt.subplot(211)
        plt.bar(range(12), PCP)
        plt.ylim([0, 4])
        plt.subplot(212)
        plt.text(0.5, 0.5, result, horizontalalignment='center',verticalalignment='center', fontsize = 30)
        plt.axis('off')
        plt.show()


    data, sr = lb.load(wav_file_path,sr=None)
    # plt.figure()
    # plt.plot(data)
    
    wave_peak = feature.onset_detection(data, sr, show=False)
    PCP = feature.chromagram(data,sr,wave_peak,show=False)
    PCP = np.array(PCP)
    # data_rms = np.sqrt(np.mean(np.square(data)))
    # print(data_rms)
    # if data_rms < 0.05:
        # continue
    # PCP = lb.feature.chroma_cqt(y, sr)
    # PCP = np.sum(PCP[:,0:4], axis=1)
    # print(PCP)
    # result = le.inverse_transform(model.predict(PCP.reshape(1,-1)))
    result = le.inverse_transform(model.predict(PCP))
    # if result == 'F Maj':
    #     continue
    # plt.show()
    # print(result)
    # plt.subplot(211)
    # plt.bar(range(12), PCP)
    # plt.ylim([0, 4])
    # plt.subplot(212)
    # plt.text(0.5, 0.5, result, horizontalalignment='center',verticalalignment='center', fontsize = 30)
    # plt.axis('off')
    # plt.pause(0.05)

    # drawnow.drawnow(draw_fig)
    # print('Predict: {}'.format(result))
    # plt.show()
    return wave_peak, result


def most_frequent(List): 
	occurence_count = Counter(List) 
	return occurence_count.most_common(1)[0][0] 


if __name__ == "__main__":
    start = timeit.default_timer()
    # wav_file_path = 'C:/Users/bhxxl/Desktop/Project - Copy/ACR/chord audio/A Min.wav'
    wav_file_path = 'C:/Users/bhxxl/Desktop/Project - Copy/chords/E-15634125.wav'
    wave_peak, result = realtime_recognition(model, le, wav_file_path)
    # print(result)
    # take the majority of each chord played
    final_result = []
    j = len(result) / len(wave_peak)
    for i in range(0,len(wave_peak)):
        temp = result[int(i*j):int((i+1)*j)]
        # print(temp)
        final_result.append(most_frequent(temp))
    print(final_result)
    stop = timeit.default_timer()
    print('Time: ', stop-start)