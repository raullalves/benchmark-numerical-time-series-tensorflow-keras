import tensorflow as tf
if int((tf.__version__).split('.')[1]) <= 6:
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, GRU, Dropout
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import GRU
import math
import numpy as np

from logger import Logger
class Model:
    def __init__(self, input_shape, index, epochs = 50, batch_size = 32, verbose = 0):
        Logger.log('Going to create model instance', verbose=0)

        self.file_log = 'model'
        self.input_shape = input_shape
        
        self.qtd_of_features = self.input_shape[1]
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.temp_filename = self.file_log+'_'+str(index)+'.hdf5'
        Logger.log('Model instance created', verbose=0)

    def create_model(self):
        return None

    def fit(self, train_generator, validation_generator):
        Logger.log_to_file('Going to fit', self.file_log, verbose=0)

        steps_per_epoch=math.floor(len(train_generator)/self.batch_size)
        if int((tf.__version__).split('.')[1]) <= 6:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=100, min_lr=0.000001, verbose=self.verbose)
            early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, verbose=self.verbose)
            mcp_save = keras.callbacks.ModelCheckpoint(self.temp_filename,verbose=self.verbose, save_best_only=True, monitor='val_loss', mode='min')
        else:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=100, min_lr=0.000001, verbose=self.verbose)
            early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, verbose=self.verbose)
            mcp_save = tf.keras.callbacks.ModelCheckpoint(self.temp_filename,verbose=self.verbose, save_best_only=True, monitor='val_loss', mode='min')
        
        self.model.fit_generator(train_generator, validation_data=validation_generator, steps_per_epoch=steps_per_epoch, epochs=self.epochs,verbose=self.verbose,callbacks=[reduce_lr, mcp_save, early], shuffle=False)
        
        self.model.load_weights(self.temp_filename)

        Logger.log_to_file('Model '+self.file_log+' fit', self.file_log, verbose=0)

    def predict(self, X):
        
        X = np.reshape(X, (1, self.input_shape[0], self.qtd_of_features))
        pred = self.model.predict(X)
        
        #reset weights
        self.model.reset_states()

        return pred