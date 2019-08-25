from logger import Logger
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

class Feature:
    def __init__(self, train_size, list_of_data, forecast,  window_size, batch_size):
        self.list_of_data = list_of_data
        self.forecast = forecast
        self.train_size = train_size
        self.window_size = window_size
        self.batch_size = batch_size

        self.qtd_of_features = len(self.list_of_data)
        Logger.log('The chosen forecast is '+str(self.forecast))
        Logger.log('The window size is '+str(self.window_size))
        Logger.log('Feature instance created')
        
    def crete_feature_array(self):
        Logger.log('Going to create feature array')

        self.data = pd.concat(([xi.get_series()
                                for xi in self.list_of_data]), axis=1)
        self.real_qtd_of_features = self.data.shape[1]
        self.qtd = self.data.values.shape[0]

        self.index = 0

        self.train_index = int(round(self.data.values.shape[0]*self.train_size)) - self.window_size + 1
        self.train_index = self.train_index + self.train_index%2
        self.batch_iter = self.train_index-self.window_size

        batch_size = math.ceil(self.batch_iter/2)
        if self.batch_size > batch_size:
            self.batch_size = batch_size

        self.train_data = np.zeros((self.batch_size, self.window_size, self.data.values.shape[1]))
        self.train_label = np.zeros((self.batch_size,1))
        
        self.validation_data = np.zeros((self.batch_size,self.window_size,self.data.values.shape[1]))
        self.validation_label = np.zeros((self.batch_size,1))

        self.to_predict = np.zeros((self.batch_size,self.window_size,self.data.values.shape[1]))
        self.ground_truth = np.zeros((self.batch_size,1))
        
        Logger.log('Feature array created with ' +
                   str(self.data.shape[0])+' samples and '+str(self.data.shape[1])+' features')
    
    def get_batch_size(self):
        return self.batch_size

    def get_feature_size(self):
        return self.data.shape[1]

    def get_data(self, pos):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        train_validation_test = self.data.values[self.index:self.index+self.train_index,:]
        
        train_validation_test = scaler.fit_transform(train_validation_test)
        labels = pd.Series(train_validation_test[:,pos]).shift(-(self.forecast+self.window_size-1))
        
        for i in range(self.batch_size):
            assert train_validation_test[2*i:2*i+self.window_size].shape[0] == self.window_size
            assert np.isnan(train_validation_test[2*i:2*i+self.window_size,:]).any() == False
            self.train_data[i] = train_validation_test[2*i:2*i+self.window_size,:]
            assert np.isnan(labels[2*i]).any() == False
            self.train_label[i] = labels[2*i]
    
            assert train_validation_test[2*i+1:2*i+1+self.window_size].shape[0] == self.window_size
            assert np.isnan(train_validation_test[2*i+1:2*i+1+self.window_size,:]).any() == False
            self.validation_data[i] = train_validation_test[2*i+1:2*i+1+self.window_size,:]
            assert np.isnan(labels[2*i+1]).any() == False
            self.validation_label[i] = labels[2*i+1]
        
        for i in range(self.batch_size):
            assert train_validation_test[i+2*self.batch_size:i+2*self.batch_size+self.window_size,:].shape[0] == self.window_size
            assert np.isnan(train_validation_test[i+2*self.batch_size:i+2*self.batch_size+self.window_size,:]).any() == False
            self.to_predict[i] = train_validation_test[i+2*self.batch_size:i+2*self.batch_size+self.window_size,:]
            assert np.isnan(labels[i+2*self.batch_size]).any() == False
            self.ground_truth[i] = labels[i+2*self.batch_size]

        return self.train_data.copy(), self.train_label.copy(), self.validation_data.copy(), self.validation_label.copy(), self.to_predict.copy(), self.ground_truth.copy(), scaler

    def has_ended(self):
        return self.index+self.train_index == self.qtd

    def normalize_inverse(self, ground_truth, predicted, index, scaler):
        predicted_X = np.zeros((self.batch_size,self.real_qtd_of_features))
        ground_truth_X = np.zeros((self.batch_size,self.real_qtd_of_features))
        predicted_X[:,index] = np.reshape(predicted, (1,self.batch_size))
        ground_truth_X[:,index] = np.reshape(ground_truth, (1,self.batch_size))
        predicted = scaler.inverse_transform(predicted_X)
        real = scaler.inverse_transform(ground_truth_X)
        return real[0, index], predicted[0, index]

    def get_last_date(self):
        return (self.list_of_data[0].get_dates())[self.index+self.train_index]

    def move_window(self):
        self.index = self.index + 1
    
    def get_index(self):
        return self.index+self.train_index

    def get_total(self):
        return self.qtd

    def get_qtd_of_features(self):
        return self.qtd_of_features
