import sys
import os
import argparse

from feature import Feature
from data import Data
from lstm import Lstm
from feature import Feature
from result import Result

from logger import Logger

def create_models(features, window_size, validation_rate):
    models = []
    for i in range(features.get_qtd_of_features()):
        model = Lstm(features.get_qtd_of_features(), i, validation_rate, window_size=window_size)
        model.create_model()
        models.append(model)

    return models

def load_csv(folder):
    data_csv = []
    index = 0
    for r, d, f in os.walk(folder):
        for file in f:
            if '.csv' in file:
                csv = Data(folder, file, index)
                csv.read()
                data_csv.append(csv)
                index = index + 1

    Logger.log(str(len(data_csv))+' csvs were loaded')

    return data_csv

def create_array_of_features_and_labels(train_size, list_of_features, forecast, window_size):
    features = Feature(train_size, list_of_features, forecast, window_size)
    features.crete_feature_array()

    return features

def create_results_instances(list_of_features):
    results = []

    for feature in list_of_features:
        result = Result(feature.get_name())
        result.set_feature_position(feature.get_feature_position())
        results.append(result)
    
    return results

def loop(features, models, results):
    features.initialize()

    while not features.has_ended():
        X = features.get_train_set()
        Logger.log('Training until date '+str(features.get_last_date()))
        for result in results:
            y = features.get_label_set(result.get_feature_position())
            models[result.get_feature_position()].fit(X,y)

            x = features.get_next_sample()
            x_predicted = models[result.get_feature_position()].predict(x)
            x, x_predicted = features.normalize_inverse(x, x_predicted, result.get_feature_position())

            result.calculate_loss_and_show_partial_results(x,x_predicted)
        Logger.log('Run '+str(features.get_index())+ ' of '+str(features.get_total()))
        features.move_window()

def main(args):
    list_of_features = load_csv(args.d)

    features = create_array_of_features_and_labels(args.train_size, list_of_features, args.forecast, args.window_size)
    models = create_models(features, args.window_size, args.validation_rate)
    results = create_results_instances(list_of_features)

    loop(features, models, results)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'EQM Raul Alves')
    parser.add_argument('-d', action="store", default='Data', dest='d')
    parser.add_argument('-f', action="store",type=int, default=1, dest='forecast')
    parser.add_argument('-t', action="store",type=int, default=30, dest='train_size')
    parser.add_argument('-w', action="store", type=int, default=12, dest='window_size')
    parser.add_argument('-v', action="store", type=int, default=30, dest='validation_rate')
    args = parser.parse_args(sys.argv[1:])
    main(args)