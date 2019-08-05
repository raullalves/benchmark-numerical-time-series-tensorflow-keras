import os
import datetime

try:os.mkdir('logs')
except:None

class Logger:
    @staticmethod
    def log(msg, verbose = 1):
        msg = str(datetime.datetime.now()) + ' - '+str(msg)
        if verbose:
            print(msg)
        msg = msg + '\n'
        with open(os.path.join('logs','log.txt'), "a") as myfile:
            myfile.write(msg)

    @staticmethod 
    def log_to_file(msg, file, verbose = 1, show_date=1):
        if show_date:
            msg = str(datetime.datetime.now()) + ' - '+str(msg)
        else:
            msg = str(msg)
        if verbose:
            print(msg)
        msg = msg + '\n'
        with open(os.path.join('logs',file+'.txt'), "a") as myfile:
            myfile.write(msg)