import sys
import os
import argparse

from feature import Feature
from data import Data
from lstm import Lstm
from feature import Feature
from result import Result

from logger import Logger

def create_models(input_shape):
    models = []
    for i in range(input_shape[1]):
        model = Lstm(input_shape, i)
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

def create_array_of_features_and_labels(train_size, list_of_features, forecast):
    features = Feature(train_size, list_of_features, forecast)
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
    if not args.windowing_tipe == 'expanding':
        Logger.log('Only expanding window is currently implemented')
        os._exit(1)

    list_of_features = load_csv(args.d)

    models = create_models((1, len(list_of_features)))

    features = create_array_of_features_and_labels(args.train_size, list_of_features, args.forecast)

    results = create_results_instances(list_of_features)
    loop(features, models, results)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'EQM Raul Alves')
    parser.add_argument('-d', action="store", default='Data', dest='d')
    parser.add_argument('-f', action="store",type=int, default=1, dest='forecast')
    parser.add_argument('-t', action="store",type=int, default=30, dest='train_size')
    parser.add_argument('-w', action="store", default='expanding', dest='windowing_tipe')
    args = parser.parse_args(sys.argv[1:])
    main(args)