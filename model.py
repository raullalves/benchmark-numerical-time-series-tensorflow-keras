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
    from tensorflow.keras.callbacks import Callback
    import tensorflow.keras.backend as K
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
        self.epoch_terminate_early = int(self.epochs/4)
        Logger.log('Model instance created', verbose=0)

    def create_model(self):
        return None

    def fit(self, train_data,train_label, validation_data, validation_label):
        Logger.log_to_file('Going to fit', self.file_log, verbose=0)

        if int((tf.__version__).split('.')[1]) <= 6:
            mcp_save = keras.callbacks.ModelCheckpoint(self.temp_filename,verbose=self.verbose, save_best_only=True, monitor='val_loss', mode='min')
        else:
            mcp_save = tf.keras.callbacks.ModelCheckpoint(self.temp_filename,verbose=self.verbose, save_best_only=True, monitor='val_loss', mode='min')
        v_loss_best = np.inf
        qtd_epochs_no_improvement = 0
        for i in range(self.epochs):
            h = self.model.fit(train_data, train_label, batch_size=self.batch_size, validation_data=(validation_data, validation_label),callbacks=[mcp_save], epochs=1, verbose=self.verbose, shuffle=False)
            v_loss = h.history['val_loss'][0]
            if v_loss < v_loss_best:
                if self.verbose:
                    print('Loss improved from '+str(v_loss_best)+' to '+str(v_loss))
                v_loss_best = v_loss
                qtd_epochs_no_improvement = 0
            else:
                qtd_epochs_no_improvement+=1
                if self.verbose:
                    print('Model has '+str(qtd_epochs_no_improvement)+' iterations without improving validation loss')
            self.model.reset_states()
            if qtd_epochs_no_improvement == self.epoch_terminate_early:
                if self.verbose:
                    print('Model has gone '+str(self.epoch_terminate_early)+' iterations without improvement in the validation loss. Gonna terminate')
                break
        self.model.load_weights(self.temp_filename)

        Logger.log_to_file('Model '+self.file_log+' fit', self.file_log, verbose=0)

    def predict(self, X):
        pred = self.model.predict(X)
        self.model.reset_states()

        return pred