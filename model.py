import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU

import numpy as np

from logger import Logger
class Model:
    def __init__(self, input_shape, index, validation_rate, epochs = 2000, batch_size = 512, verbose = 0):
        Logger.log('Going to create model instance', verbose=0)

        self.file_log = 'model'
        self.input_shape = input_shape
        
        self.qtd_of_features = self.input_shape[1]
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_rate = validation_rate/100
        self.verbose = verbose
        self.temp_filename = self.file_log+'_'+str(index)+'.hdf5'
        Logger.log('Model instance created', verbose=0)

    def create_model(self):
        return None

    def fit(self, X, y):
        Logger.log_to_file('Going to fit', self.file_log, verbose=0)
        
        #sequence shifting for training
        X_ = np.zeros((X.shape[0]- self.input_shape[0] +1, self.input_shape[0], self.qtd_of_features), dtype=np.float64)
        y_ = np.zeros((X.shape[0]- self.input_shape[0] +1,), dtype=np.float64)
        
        for i in range(X.shape[0]-self.input_shape[0] +1):
            X_[i,:,:] = X[i:i+self.input_shape[0],:]
            y_[i] = y[i+self.input_shape[0]-1]

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,patience=100, min_lr=0.000001, verbose=self.verbose)
        early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=150, verbose=self.verbose)
        mcp_save = tf.keras.callbacks.ModelCheckpoint(self.temp_filename,verbose=self.verbose, save_best_only=True, monitor='loss', mode='min')
        
        self.model.fit(X_, y_,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,callbacks=[reduce_lr, mcp_save, early], shuffle=False)
        
        self.model.load_weights(self.temp_filename)

        Logger.log_to_file('Model '+self.file_log+' fit', self.file_log, verbose=0)
    
    def predict(self, X):
        X = np.reshape(X, (1, self.input_shape[0], self.qtd_of_features))
        pred = self.model.predict(X)
        
        #reset weights
        self.model.reset_states()

        return pred