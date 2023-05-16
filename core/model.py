import os
import wandb
import numpy as np
import datetime as dt
import json

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_m_dir, para_str):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        save_m_name = '%s-e%s' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs))
        save_m_fname = os.path.join(save_m_dir, save_m_name)
        save_m_fname = save_m_fname + para_str + '.h5'
        callbacks = [
            ModelCheckpoint(filepath=save_m_fname, monitor='loss', save_best_only=True)
        ]

        wandb.init(project='hmc-ulstm-prediction', config=json.load(open('config.json', 'r')), name=save_m_name)
        run_id = wandb.run.id
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[
                callbacks,
                WandbMetricsLogger(log_freq='batch'),
                # WandbModelCheckpoint(save_m_fname,
                #                      monitor='loss',
                #                      save_best_only=True)
            ],
            workers=1
        )

        print('run_id: ', run_id)
        print('[Model] Training Completed. Model saved as %s' % save_m_fname)
        timer.stop()

        # save config into wandb
        arti_config = wandb.Artifact('config', type='code')
        arti_config.add_file('./config.json')
        wandb.log_artifact(arti_config)

        # save model into wandb
        arti_model = wandb.Artifact('ulstm', type='model')
        arti_model.add_file(save_m_fname)
        wandb.log_artifact(arti_model)

        # save core code into wandb
        arti_core = wandb.Artifact('core', type='code')
        arti_core.add_dir('./core')
        wandb.log_artifact(arti_core)

        return save_m_name

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        timer = Timer()
        timer.start()
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        timer.stop()
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        timer = Timer()
        timer.start()
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        timer.stop()
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        timer = Timer()
        timer.start()
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        timer.stop()
        return predicted
