__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"


'''
predict for all A-scans
'''

import os
import io
import json
import wandb
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
from PIL import Image
from keras.utils.vis_utils import plot_model
from scipy.io import savemat
from tensorflow.keras.losses import MSE
from multiprocessing import Pool


def plot_results(predicted_data, true_data, chan_num, mode, save_folder):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.title(mode + ' result: ' + str(chan_num))
    # plt.savefig(os.path.join(save_folder, '{}_channel {}'.format(mode, str(chan_num))))
    # plt.show()
    return fig


def plot_results_multiple(predicted_data, true_data, prediction_len, chan_num):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.title('multi-seq result: ' + str(chan_num[0]))
    plt.show(block=True)


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_model_dir']):
        os.makedirs(configs['model']['save_model_dir'])

    if not os.path.exists(configs['model']['save_pic_dir']):
        os.makedirs(configs['model']['save_pic_dir'])

    if not os.path.exists(configs['model']['save_numeric_dir']):
        os.makedirs(configs['model']['save_numeric_dir'])

    setting = '_{}_sl{}_bs{}_{}_{}_{}'.format(
        configs['data']['filename'][11:15],
        configs['data']['sequence_length'],
        configs['training']['batch_size'],
        configs['model']['loss'],
        configs['model']['optimizer'],
        configs['model']['layers'][0]['neurons'],
        # configs['model']['layers'][2]['neurons']
    )

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['channel_information'],
        configs['data']['sequence_length']
    )

    model = Model()
    model.build_model(configs)
    # plot_model(model.model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    # plt.figure(figsize=(10, 10))
    # img = plt.imread("model.png")
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    # x, y = data.get_train_data(
    #     seq_len=configs['data']['sequence_length'],
    #     normalise=configs['data']['normalise']
    # )

    '''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	'''

    # out-of memory generative training
    steps_per_epoch, total_num_wins = data.cal_steps_per_epoch(
        batch_size=configs['training']['batch_size']
    )

    data_gen = data.generate_train_batch(
        seq_len=configs['data']['sequence_length'],
        batch_size=configs['training']['batch_size'],
        normalise=configs['data']['normalise'],
        total_num_wins=total_num_wins
    )

    save_model_name = model.train_generator(
        data_gen=data_gen,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_m_dir=configs['model']['save_model_dir'],
        para_str=setting
    )

    # calculate the max value of test data for inverse normalization
    max_test_list = data.cal_max_test_value()

    # create the folder to save images of one training
    save_pic_folder = os.path.join(configs['model']['save_pic_dir'], save_model_name)
    if not os.path.exists(save_pic_folder):
        os.makedirs(save_pic_folder)

    full_seq_result = []
    pointbypoint_result = []
    # create the Table to visualize results in wandb
    pred_case = wandb.Table(columns=['scan_id', 'full_seq_result', 'full_seq_mse', 'point_by_point_result', 'point_by_point_mse'])
    for n in range(data.hmc_data.shape[1]):
        # 一次处理一个通道的预测
        x_test, y_test = data.get_test_data(
            index=n,
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )

        # predict
        predictions_fullseq = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
        predictions_pointbypoint = model.predict_point_by_point(x_test)

        full_seq_mse = MSE(tf.reshape(predictions_fullseq, [1, -1]), tf.reshape(y_test, [1, -1]))
        point_by_point_mse = MSE(tf.reshape(predictions_pointbypoint, [1, -1]), tf.reshape(y_test, [1, -1]))

        # inverse normalizaton
        predictions_fullseq_inv = [(float(p) * float(max_test_list[n])) for p in predictions_fullseq]
        predictions_pointbypoint_inv = [(float(p) * float(max_test_list[n])) for p in predictions_pointbypoint]

        full_seq_result.append(predictions_fullseq_inv)
        pointbypoint_result.append(predictions_pointbypoint_inv)

        y_test_inv = y_test * max_test_list[n]

        # plot and save the image into wandb
        full_seq_fig = plot_results(predictions_fullseq_inv, y_test_inv, n, 'full seq', save_pic_folder)
        full_seq_img = wandb.Image(fig2img(full_seq_fig))
        pyp_fig = plot_results(predictions_pointbypoint_inv, y_test_inv, n, 'point-by-point', save_pic_folder)
        pyp_img = wandb.Image(fig2img(pyp_fig))
        pred_case.add_data(n, full_seq_img, full_seq_mse, pyp_img, point_by_point_mse)

        # predictions_multiseq = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
        #                                                configs['data']['sequence_length'])
        # predictions_multiseq_inv = [(float(p) * float(max_test_list[n])) for p in predictions_multiseq]
        # plot_results_multiple(predictions_multiseq_inv, y_test, configs['data']['sequence_length'], configs['model']['columns'])

        '''
        without inverse normalization
        
        # predict
        # predictions_multiseq = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
        #                                                configs['data']['sequence_length'])
        predictions_fullseq = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
        predictions_pointbypoint = model.predict_point_by_point(x_test)

        # # plot and save the image & numeric
        # # plot_results_multiple(predictions_multiseq_inv, y_test, configs['data']['sequence_length'], configs['model']['columns'])
        # plot_results(predictions_fullseq, y_test, n, 'full seq', save_pic_folder)
        # plot_results(predictions_pointbypoint, y_test, n, 'point-by-point', save_pic_folder)
         '''

    # create the folder to save numeric result of one training
    save_numeric_folder = os.path.join(configs['model']['save_numeric_dir'], save_model_name)
    if not os.path.exists(save_numeric_folder):
        os.makedirs(save_numeric_folder)
    print('full_seq_result shape:{}'.format(np.array(full_seq_result).shape))
    print('full_seq_result shape:{}'.format(np.array(pointbypoint_result).shape))
    savemat(os.path.join(save_numeric_folder, 'full_seq.mat'), {'full_seq_result': np.array(full_seq_result)})
    savemat(os.path.join(save_numeric_folder, 'pointbypoint.mat'), {'pointbypoint_result': np.array(pointbypoint_result)})

    wandb.log({'pred_case': pred_case})
    wandb.finish()


if __name__ == '__main__':
    main()
