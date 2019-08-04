import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU

import numpy as np

from logger import Logger
class Model:
    def __init__(self, input_shape, index, epochs = 3000, batch_size = 512, verbose = 0):
        Logger.log('Going to create model instance')

        self.file_log = 'model'
        self.input_shape = input_shape
        
        self.qtd_of_features = self.input_shape[1]
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.temp_filename = self.file_log+'_'+str(index)+'.hdf5'

        Logger.log('Model instance created')

    def create_model(self):
        return None

    def fit(self, X, y):
        Logger.log_to_file('Going to fit', self.file_log)

        X = np.reshape(X, (X.shape[0], 1, self.qtd_of_features))
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,patience=100, min_lr=0.000001, verbose=self.verbose)
        early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200, verbose=self.verbose)
        mcp_save = tf.keras.callbacks.ModelCheckpoint(self.temp_filename,verbose=self.verbose, save_best_only=True, monitor='loss', mode='min')
        
        self.model.fit(X, y,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,callbacks=[reduce_lr, mcp_save, early], shuffle=False)
        self.model.reset_states()
        self.model.load_weights(self.temp_filename)

        Logger.log_to_file('Model '+self.file_log+' fit', self.file_log)
    
    def predict(self, X):
        X = np.reshape(X, (1, 1, self.qtd_of_features))
        return self.model.predict(X)