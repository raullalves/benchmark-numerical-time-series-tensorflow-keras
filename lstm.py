import tensorflow as tf
if int((tf.__version__).split('.')[1]) <= 6:
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, GRU, Dropout
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LeakyReLU
    from tensorflow.keras.layers import LSTM, BatchNormalization
    from tensorflow.keras.layers import GRU, CuDNNLSTM, CuDNNGRU
    from tensorflow.keras.layers import Dropout, TimeDistributed, Activation

from model import Model
from logger import Logger

class Lstm(Model):
    def __init__(self, input_shape, index, window_size = 5, epochs = 10000, batch_size = 64, verbose = 0):
        super(Lstm, self).__init__(input_shape,index, epochs, batch_size, verbose)
        
        self.file_log = 'lstm'
        self.temp_filename = self.file_log+'_'+str(index)+'.hdf5'
        self.neurons = 64
        self.dropout = 0.5
        self.window_size = window_size
        
        Logger.log('The window size is '+str(self.window_size))
        Logger.log_to_file('LSTM instance created', self.file_log)
        
    def create_model(self):
        Logger.log_to_file('Going to create LSTM model', self.file_log)
        Logger.log_to_file('Model shape is '+str(self.input_shape), self.file_log)

        self.model = Sequential()
        self.model.add(LSTM(self.neurons, batch_input_shape=self.input_shape, return_sequences=False, stateful=True, recurrent_dropout=self.dropout, kernel_initializer=tf.keras.initializers.glorot_normal(seed=777), bias_initializer='zeros'))
        
        self.model.add(Dense(1))
        self.model.compile(loss='mse',optimizer='adam', metrics=['mse'])

        Logger.log_to_file('LSTM model created', self.file_log)
