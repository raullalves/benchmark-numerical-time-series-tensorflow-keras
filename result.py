from sklearn.metrics import mean_squared_error
import math
import numpy as np
from logger import Logger

class Result:
    def __init__(self, name):
        self.name = name
        self.index = 0
        self.qtd = 0
        self.rmses = []
        self.real_storing = []
        self.predicted_storing = []

        Logger.log('Created result instance for feature '+self.name)
    
    def calculate_loss_and_show_partial_results(self, x, x_pred):
        self.qtd = self.qtd + 1
        
        #rmse only on specific element
        x = (x[0])[self.index]
        x_pred = (x_pred[0])[self.index]

        self.real_storing.append(x)
        self.predicted_storing.append(x_pred)

        Logger.log_to_file(x, self.name+'_real', 0)
        Logger.log_to_file(x_pred, self.name+'_predicted', 0)
        
        rmse = math.sqrt(mean_squared_error([x],[x_pred]))
        self.rmses.append(rmse)
        Logger.log_to_file('For feature '+self.name+' predicted '+str(x_pred)+ ' where actual value is '+str(x), self.name)
        Logger.log_to_file('Partial rmse for feature '+self.name+' = '+str((np.asarray(self.rmses).sum())/self.qtd), self.name)


    def set_feature_position(self, index):
        self.index = index

        Logger.log('Feature '+self.name+' at index '+str(self.index))
    
    def get_feature_position(self):
        return self.index