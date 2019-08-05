from numpy import nan
from numpy import isnan
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import to_numeric
from pandas import Series
from matplotlib import pyplot
import os
from logger import Logger
class Data:
    def __init__(self, folder, name, index):
        self.folder = folder
        self.name = name
        self.index = index
        
        Logger.log('Data '+self.name+' at index '+str(self.index))

    def get_feature_position(self):
        return self.index

    def get_name(self):
        return self.name

    def read(self):
        Logger.log('Going to read csv file '+self.name)

        if int((pd.__version__).split('.')[1]) <= 24:
            self.series = Series.from_csv(os.path.join(self.folder, self.name), header=0)
        else:
            Logger.log('-------WARNING------')
            Logger.log('This SW was tested with pandas 0.24.2')
            self.series = Series.read_csv(os.path.join(self.folder, self.name), header=0)

        self.series = self.series.resample('M').mean()
        self.series = self.series.astype('float32')
        
        self.size = len(self.series.values)

        self.X = self.series.values
 
        self.dates = self.series.index

        Logger.log('File '+self.name+' read: Total of '+str(self.size)+' timestamps. From '+str(self.dates[0])+ ' to '+str(self.dates[-1]))

    def get_values(self):
        return self.X
    
    def get_dates(self):
        return self.dates
    
    def get_series(self):
        return self.series

    #only for demo
    def show_train_test_info(self):
        self.train_size=0.3
        train_size = int(self.size*self.train_size)
        train, test = self.X[0:train_size], self.X[train_size:self.size]

        dateArray = []
        temp = []
        for index, date in enumerate(self.dates):
            year = date.year
            month = date.month
            toBeXAxis = ''
            if(month == 1):
                if int(year) % 3 == 0:
                    toBeXAxis = str(year)
            dateArray.append(toBeXAxis)
            temp.append(index)

        pyplot.xlabel('data')
        pyplot.ylabel(self.name)
        pyplot.xticks(temp, np.array(dateArray))
        pyplot.plot(train, label = 'train')
        pyplot.plot(temp, [None for i in train] + [x for x in test], label = 'test')
        pyplot.title('Divisão da série temporal entre treino e teste')
        pyplot.legend(loc='upper left')
        pyplot.show()
