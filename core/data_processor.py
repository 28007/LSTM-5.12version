import math
import random

import numpy as np
import pandas as pd
from scipy.io import loadmat

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, chanl_inf_name, seq_len):
        '''
        seq_len: length of the window
        pred_len: the length of the pred output
        pred_end: the end index in the time-axis of HMC data
        '''
        dataframe = loadmat(filename)['exp_data2']['time_data'][0, 0]
        dataframe = pd.DataFrame(dataframe)
        chanl_inf = loadmat(chanl_inf_name)['chanl_inf']
        self.hmc_data = dataframe
        self.seq_len = seq_len
        self.pred_len = chanl_inf['diff_index'][0, 0]
        self.pred_end = chanl_inf['l_time_index'][0, 0]


    def get_test_data(self, index, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_test = self._select_channel(index)[2]
        data_windows = []
        for i in range(len(data_test) - seq_len):
            data_windows.append(data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, data_test, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1]
        return x, y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def cal_steps_per_epoch(self, batch_size):
        num_wins = []
        for i in range(self.hmc_data.shape[1]):
            data_train, len_train = self._select_channel(i)[:2]
            num_win = len_train - self.seq_len   # 当前通道下能产生的windows数量
            num_wins.append(num_win)
            i += 1
        total_num_wins = sum(num_wins)
        print('total number of train windows: {}'.format(total_num_wins))
        steps_per_epoch = total_num_wins // batch_size
        print('steps_per_epoch is: {}'.format(steps_per_epoch))
        return steps_per_epoch, total_num_wins


    def generate_train_batch(self, seq_len, batch_size, normalise, total_num_wins):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0  # window start index in a train sequence
        n = 0  # channel
        t = 0  # total number which has been produced
        # 选取通道
        num_channel = self.hmc_data.shape[1]   # 通道总数
        random.seed(0)
        chan_list = random.sample(range(num_channel), num_channel)  # 打乱通道索引
        data_train, len_train = self._select_channel(chan_list[n])[:2]
        # 从当前序列提取window
        while True:
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i == (len_train - seq_len)+1:
                    # 选择下一个通道，且i重置为0
                    n += 1
                    i = 0
                    data_train, len_train = self._select_channel(chan_list[n])[:2]
                if t == total_num_wins:
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    i = 0
                    n = 0
                    t = 0
                    yield np.array(x_batch), np.array(y_batch)
                    data_train, len_train = self._select_channel(chan_list[n])[:2]
                x, y = self._next_window(i, seq_len, normalise, data_train)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
                t += 1
            yield np.array(x_batch), np.array(y_batch)

    def _select_channel(self, n, ):
        dataframe = self.hmc_data[n]
        zero_index = dataframe[dataframe.values == 0].index.tolist()
        dataframe_end = self.pred_end[0, n]
        dataframe = dataframe[zero_index[-1] + 1:dataframe_end+2]
        # 划分训练序列和测试序列
        i_split = int(len(dataframe) - self.pred_len[0, n])
        data_train = dataframe.values[:i_split]
        len_train = len(data_train)
        data_test = dataframe.values[i_split-self.seq_len:]
        return data_train, len_train, data_test

    def _next_window(self, i, seq_len, normalise, data):
        '''Generates the next data window from the given index location i'''
        window = data[i:i+seq_len]
        window = self.normalise_windows(window, data, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1]
        return x, y

    def normalise_windows(self, window_data, data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            window = window[:, np.newaxis]
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(np.max(data)))) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    def cal_max_test_value(self,):
        max_test_value = []
        for i in range(self.hmc_data.shape[1]):
            test_df = self._select_channel(i)[2]
            max_test_value.append(max(test_df))
        return np.array(max_test_value)