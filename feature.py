import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from logger import Logger
class Feature:
    def __init__(self, train_size, list_of_data, forecast, window_size):
        self.list_of_data = list_of_data
        self.forecast = forecast
        self.train_size = train_size/100
        self.index = 0
        self.window_size = window_size

        self.qtd_of_features = len(self.list_of_data)

        Logger.log('The chosen forecast is '+str(self.forecast))
        Logger.log('Feature instance created')
    
    def get_qtd_of_features(self):
        return self.qtd_of_features

    def normalize(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data_normalized = pd.DataFrame(self.scaler.fit_transform(self.data.values[:self.index+self.forecast]))
    
    def normalize_inverse(self, x, predicted, index):
        x = x[-1]
        #replace only the predicted feature
        x_predicted = x.copy()
        x_predicted[index] = predicted
        x_predicted = np.reshape(x_predicted, (1, x_predicted.shape[0]))
        predicted = self.scaler.inverse_transform(x_predicted)

        x = np.reshape(x, (1, x.shape[0]))
        real = self.scaler.inverse_transform(x)   

        for i in range(self.qtd_of_features):
            if i == index:
                continue
            assert (real[0])[i]==(predicted[0])[i]
        
        return real, predicted

    def crete_feature_array(self):
        Logger.log('Going to create feature array')

        self.data = pd.concat(([xi.get_series() for xi in self.list_of_data]),axis=1)
        self.qtd = self.data.values.shape[0]
        self.train_size_multiplied = int(self.train_size*self.data.shape[0])

        Logger.log('Feature array created with '+str(self.data.shape[0])+' samples and '+str(self.data.shape[1])+' features')
    
    def get_last_date(self):
        return (self.list_of_data[0].get_dates())[self.index-1]

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

    def has_ended(self):
        return self.index == self.qtd