import sys
import os
import argparse
from feature import Feature
from data import Data
from lstm import Lstm
from feature import Feature
from result import Result

from logger import Logger

def create_models(features, window_size, batch_size):
    models = []
    for i in range(features.get_qtd_of_features()):
        model = Lstm(input_shape=(window_size, features.get_qtd_of_features()), index=i, batch_size=batch_size, window_size=window_size)
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

def create_array_of_features_and_labels(train_size, list_of_features, forecast, validation_rate, window_size, batch_size):
    features = Feature(train_size, list_of_features, forecast, validation_rate, window_size, batch_size)
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
        
        for result in results:
            train_generator, validation_generator, test, ground_truth, scaler = features.get_data(result.get_feature_position())
            models[result.get_feature_position()].fit(train_generator, validation_generator)
            
            predicted = models[result.get_feature_position()].predict(test)
            x, x_predicted = features.normalize_inverse(ground_truth, predicted, result.get_feature_position(), scaler)

            date = features.get_last_date()
            results[result.get_feature_position()].calculate_loss_and_show_partial_results(x,x_predicted, date)
        
        Logger.log('Training until date '+str(date))
        Logger.log('Run '+str(features.get_index())+ ' of '+str(features.get_total()))
        features.move_window()

def main(args):
    if args.validation_rate < 0 or args.validation_rate > 1:
        Logger.log('Validation should be from 0 to 1')
        os._exit(1)
    if args.train_size < 0 or args.train_size > 0.9:
        Logger.log('train size should be from 0 to 0.9')
        os._exit(1)

    list_of_features = load_csv(args.d)

    features = create_array_of_features_and_labels(args.train_size, list_of_features, args.forecast, args.validation_rate, args.window_size, args.batch_size)
    models = create_models(features, args.window_size, args.batch_size)
    results = create_results_instances(list_of_features)

    loop(features, models, results)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'EQM Raul Alves')
    parser.add_argument('-d', action="store", default='Data', dest='d')
    parser.add_argument('-f', action="store",type=int, default=1, dest='forecast')
    parser.add_argument('-t', action="store",type=float, default=0.3, dest='train_size')
    parser.add_argument('-w', action="store", type=int, default=12, dest='window_size')
    parser.add_argument('-v', action="store", type=float, default=0.3, dest='validation_rate')
    parser.add_argument('-b', action="store", type=int, default=3, dest='batch_size')
    args = parser.parse_args(sys.argv[1:])
    main(args)