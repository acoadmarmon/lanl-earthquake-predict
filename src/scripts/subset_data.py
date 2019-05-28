import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import random
import numpy as np
import pywt
import os

train_df = pd.read_csv('./data/train.csv')
print('done')

# def get_signal_features(df_segment, window_size):
#     df = pd.DataFrame(index=range(150000/window_size    ), dtype=np.float64)
#     fft = np.fft.fft(df_segment.acoustic_data)
#     realFFT = np.real(zc)
#     imagFFT = np.imag(zc)
#     df['Rmean'] = realFFT.mean()
#     X.loc[seg_id, 'Rstd'] = realFFT.std()
#     X.loc[seg_id, 'Rmax'] = realFFT.max()
#     X.loc[seg_id, 'Rmin'] = realFFT.min()
#     X.loc[seg_id, 'Imean'] = imagFFT.mean()
#     X.loc[seg_id, 'Istd'] = imagFFT.std()
#     X.loc[seg_id, 'Imax'] = imagFFT.max()
#     X.loc[seg_id, 'Imin'] = imagFFT.min()
# scaler = StandardScaler(copy=False)
# train_df['acoustic_data'] = scaler.fit_transform(train_df['acoustic_data'].values.reshape(-1, 1))

#eval_index = list(np.random.permutation([i for i in range(4194)]))[:838]

import matplotlib.pyplot as plt
train_labels = dict()
eval_labels = dict()
prev_is_eval = False
for i in range(4194*2 - 838*2):
    train_labels['signal_' + str(i) + '.csv'] = train_df['time_to_failure'].iloc[i*75000 + 150000]
    wave = train_df['acoustic_data'].iloc[i*75000: i*75000 + 150000]

    fft = np.fft.fft(df_segment.acoustic_data)

    realFFT = np.real(fft)
    imagFFT = np.imag(fft)
    for i in range(1000):
        x = wave.values[150*i: 150*i + 150]
        r = realFFT[150*i: 150*i + 150]
        i = imagFFT[150*i: 150*i + 150]
        df['mean'] = x.mean()
        df['max'] = x.max()
        df['min'] = x.min()
        df['std'] = x.std()
        df['Rmean'] = r.mean()
        df['Rstd'] = r.std()
        df['Rmax'] = r.max()
        df['Rmin'] = r.min()
        df['Imean'] = i.mean()
        df['Istd'] = i.std()
        df['Imax'] = i.max()
        df['Imin'] = i.min()

    #train_df['acoustic_data'].iloc[i*75000: i*75000 + 150000].to_csv('./data/15000_processed_data/train/signal_' + str(i) + '.csv', index=False, header=False)

for i in range(4194*2 - 838*2, 4194*2 - 1, 1):
    eval_labels['signal_' + str(i) + '.csv'] = train_df['time_to_failure'].iloc[i*75000 + 150000]
    train_df['acoustic_data'].iloc[i*75000: i*75000 + 150000].to_csv('./data/15000_processed_data/eval/signal_' + str(i) + '.csv', index=False, header=False)

with open('./data/15000_processed_data/eval_labels.pkl', 'wb') as f:
    pickle.dump(eval_labels, f, protocol=2)
# with open('./data/15000_processed_data/train_labels.pkl', 'wb') as f:
#     pickle.dump(train_labels, f, protocol=2)
