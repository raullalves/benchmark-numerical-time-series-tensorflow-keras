import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
if int((tf.__version__).split('.')[1]) <= 6:
    from keras.preprocessing.sequence import TimeseriesGenerator
else:
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from logger import Logger
class Feature:
    def __init__(self, train_size, list_of_data, forecast, validation_rate, window_size, batch_size):
        self.list_of_data = list_of_data
        self.forecast = forecast
        self.train_size = train_size
        self.validation_rate = validation_rate
        self.window_size = window_size
        self.batch_size = batch_size

        self.qtd_of_features = len(self.list_of_data)
        Logger.log('The chosen forecast is '+str(self.forecast))
        Logger.log('The validation rate is '+str(self.validation_rate))
        Logger.log('The window size is '+str(self.window_size))
        Logger.log('The batch size is '+str(self.batch_size))
        Logger.log('Feature instance created')
    
    def crete_feature_array(self):
        Logger.log('Going to create feature array')

        self.data = pd.concat(([xi.get_series() for xi in self.list_of_data]),axis=1)
        self.qtd = self.data.values.shape[0]
        self.index = int(self.train_size*self.data.shape[0])
        
        Logger.log('Feature array created with '+str(self.data.shape[0])+' samples and '+str(self.data.shape[1])+' features')
    
    def initialize(self):
        Logger.log('Feature initialized')    
    
    def get_data(self, pos):
        scaler = MinMaxScaler(feature_range=(-1,1))

        data_normalized = pd.DataFrame(scaler.fit_transform(self.data.values[:self.index+self.forecast]))
        self.actual_date = (self.list_of_data[0].get_dates())[self.index+self.forecast-1]
        
        train_validation, test = data_normalized.values[:-(self.forecast+1),:], data_normalized.values[-self.window_size-self.forecast:,:]
        
        val_size = int(train_validation.shape[0] * (1-self.validation_rate))

        train, validation = train_validation[0:val_size,:], train_validation[val_size-self.window_size+1:train_validation.shape[0],:]
        
        label_train = pd.Series(train[:,pos])
        label_validation = pd.Series(validation[:,pos])

        if self.forecast == 1:
            train_generator = TimeseriesGenerator(train,label_train.shift(-self.forecast+1).values, length=self.window_size,batch_size=self.batch_size)
            validation_generator = TimeseriesGenerator(validation,(label_validation.shift(-self.forecast+1).values), length=self.window_size,batch_size=self.batch_size)
        else:
            train_generator = TimeseriesGenerator(train[:-self.forecast+1,:],label_train.shift(-self.forecast+1).values[:-self.forecast+1], length=self.window_size,batch_size=self.batch_size)
            validation_generator = TimeseriesGenerator(validation[:-self.forecast+1,:],(label_validation.shift(-self.forecast+1).values)[:-self.forecast+1], length=self.window_size,batch_size=self.batch_size)
  
        ground_truth = (pd.Series(test[:,pos]).shift(-self.forecast)).values[-self.forecast-1]
   
        return train_generator, validation_generator, test[:-self.forecast,:], ground_truth, scaler

    def has_ended(self):
        return self.index == self.qtd
    
    def normalize_inverse(self, ground_truth, predicted, index, scaler=0):
        x_real = np.zeros(shape=(1,self.qtd_of_features))
        x_predicted = np.zeros(shape=(1,self.qtd_of_features))
        x_real[:,index] = ground_truth
        x_predicted[:,index] = predicted[0]
        predicted = scaler.inverse_transform(x_predicted)
        real = scaler.inverse_transform(x_real)
        return real, predicted

    def get_last_date(self):
        return self.actual_date

    def create_labels(self):
        self.labels = self.data_normalized.shift(-self.forecast)
        self.labels.drop(self.labels.tail(self.forecast).index,inplace=True)

        assert self.data_normalized.shape[0] == self.labels.shape[0]+self.forecast
        for i in range(len(self.list_of_data)):
            np.testing.assert_array_almost_equal(self.data_normalized[i].values[self.forecast:], self.labels[i])

    def get_train_size(self):
        return self.train_size_multiplied

    def get_index(self):
        return self.index
    
    def get_total(self):
        return self.qtd

    def move_window(self):
        self.index = self.index + 1

    def initialize(self):
        self.index = int(self.train_size*self.data.values.shape[0])
        self.qtd = self.data.values.shape[0]

        Logger.log('The train size is composed by '+str(self.index)+' samples')

    def get_train_set(self):
        self.normalize()
        self.create_labels()
        
        return self.data_normalized.values[:self.index]
    
    def get_label_set(self, index):
        return (self.labels[index])[:self.index+self.forecast]
    
    def get_next_sample(self):
        return self.data_normalized.values[-self.window_size:]
        
    def get_qtd_of_features(self):
        return self.qtd_of_features