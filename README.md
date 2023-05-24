# LSTM-5.12version
In this version, we use all 2080 channels  and all time steps except the zero value of a HMC sample to train the net and applying the model to a new HMC data. We use **Keras**.

We create training dataset from all 2080 channels. We need to use MATLAB and Python both. Here we only upload the Python predicting part. 

![image](https://github.com/28007/LSTM-5.12version/assets/119842972/d30ef3c2-c7c6-4fd5-9886-8f0d868b81ff)

## 1. Calculate the index-list
First, according to the coordinate of ROI region, we calculate the related information about the channels, including the start-time index, end-time index and the prediction length of each channel. We use the `TFM_adaptive_v4.mat`to finish this and  save the channel information in the `chanl_inf.mat` .
## 2. Time series predicting
We conduct the time series predicting in Python. We give a track for `run.py`. 

### 2.1 data
In the initialization of  class `Dataloader`, we get 3 files: the HMC file using to train, the channel information in step 1 and give them to `self`. We using function`building_model` to build a sequential model defined in `config.json`. 

We calculate the steps_per_epoch using function `cal_steps_per_epoch`,  i.e. how many batches is needed in a epoch. The number of time windows for each channel is the substraction of train length and window length. Each window is a train sample, so we divide total number of time windows by batchsize then could get the steps_per_epoch.

 Let's note the function `_select_channel`.  It gives us 3 result for certain channel: `data_train`, the data for training in this channel, the `len_train`, the length of data_train, and `the data_test`, the data for testing in this channel. This function is highly related to the channel information.
### 2.2 train
The funtion `train_generator`, we mainly realize 2 aims: connecting the training progress to wandb and perform training. The function `generate_train_batch` yield batch data for training .  Parameter `i` is for record of the window start index in a certain train sequence. Parameter `n` is for record of the selected channel. Parameter `t` is for record of total number of windows that has been produced.  The fucntion `fit_generator` will call it when get the `data_gen`.  

During the progress, we use function `_next_window` to extract a window and normalization for it. We using the maximum of the train sequence as the standard value for normalization. 
### 2.3. Test
The function `cal_max_test_value` is for getting the max value of each test data sequence, which is preparing for the same normalization for test data. 

We have 3 ways to predict: point-by-point, predict all value point using the true observed value; full sequence,  using the prediction result of one previous windows to predict next value, means each value in the result sequence comes from LSTM model; multi-sequnce, we split the data into several sub-sequnce to predict.

We save the result in `full_seq.mat` or  `pointbypoint.mat`.
# 3. Concat and Imaging and SNR
We finish this in MATLAB, using the script `TFM_adaptive_v16.mat`.
## 3.1 concat
We replace the ROI part of HMC data with the predict result,  according to the mentioned channel information.
## 3.2 Imaging
Conduct the TFM using new HMC data.
## 3.3 SNR
Besides the ROI region, we set the noise area. We do abs and maximum-normalization and dB-normalization in TFM.  We divide the maximum in ROI region `defect_amp` by the average in noise area `noise_amp`, transforming into dB as the SNR.

