from sklearn.metrics import mean_squared_error
import math
import numpy as np
import os
from logger import Logger

class Result:
    def __init__(self, name):
        self.name = name
        self.index = 0
        self.qtd = 0
        self.rmses = []
        self.real_storing = []
        self.predicted_storing = []

        if os.path.isfile(os.path.join('logs',self.name+'_real.txt')):
            os.remove(os.path.join('logs',self.name+'_real.txt'))

        if os.path.isfile(os.path.join('logs',self.name+'_predicted.txt')):
            os.remove(os.path.join('logs',self.name+'_predicted.txt'))

        Logger.log_to_file('DATE,'+self.name+' real', self.name+'_real', verbose=0, show_date=0)
        Logger.log_to_file('DATE,'+self.name+' _predicted', self.name+'_predicted', verbose=0, show_date=0)

        Logger.log('Created result instance for feature '+self.name)
    
    def calculate_loss_and_show_partial_results(self, x, x_pred):
        self.qtd = self.qtd + 1

        #rmse only on specific element
        x = (x[0])[self.index]
        x_pred = (x_pred[0])[self.index]

        self.real_storing.append(x)
        self.predicted_storing.append(x_pred)

        Logger.log_to_file(x, self.name+'_real', verbose=0, show_date=0)
        Logger.log_to_file(x_pred, self.name+'_predicted', verbose=0, show_date=0)
        
        rmse = math.sqrt(mean_squared_error([x],[x_pred]))
        self.rmses.append(rmse)
        Logger.log('For feature '+self.name+' predicted '+str(x_pred)+ ' where actual value is '+str(x))
        Logger.log('Partial rmse for feature '+self.name+' = '+str((np.asarray(self.rmses).sum())/self.qtd))


    def set_feature_position(self, index):
        self.index = index

        Logger.log('Feature '+self.name+' at index '+str(self.index))
    
    def get_feature_position(self):
        return self.index