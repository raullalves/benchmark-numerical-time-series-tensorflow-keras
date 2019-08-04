import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout

from model import Model
from logger import Logger

class Lstm(Model):
    def __init__(self, input_shape, index, epochs = 500, batch_size = 512, verbose = 0):
        super(Lstm, self).__init__(input_shape,index, epochs, batch_size, verbose)
        self.file_log = 'lstm'
        self.temp_filename = self.file_log+'_'+str(index)+'.hdf5'
        self.neurons = 64
        
        Logger.log_to_file('LSTM instance created', self.file_log)
    
    def create_model(self):
        Logger.log_to_file('Going to create LSTM model', self.file_log)

        self.model = Sequential()
        self.model.add(LSTM(self.neurons, input_shape=self.input_shape))
        self.model.add(Dense(1))
        self.model.compile(loss='mae', optimizer='adam')

        Logger.log_to_file('LSTM model created', self.file_log)