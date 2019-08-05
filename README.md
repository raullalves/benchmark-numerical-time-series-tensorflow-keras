## Benchmark for numerical time series forecast using Tensorflow Keras

## Under development, but works

It implements the sliding windowing. All windows are re-normalized at every run.

It has by default a simple LSTM. Other networks can be easily integrated by your will.

The methodology merges all series in one with multiple features.

At the class data.py, by default it takes the montly mean of each time series. It obviously can be changed

# Run
python main.py
## Optional new arguments
-f: How many units of time to forecast (default: 1)

-t: The rate of the train size, integer (default: 30)

-w: windowing size (default: 12)

TODO: -v: validation size (default: 30)

# Add model
It has a simple LSTM by default.

Create your model class that inherits the class Model.

Override the method create_model(), where returns your model instance, compiled

# Data
Put your csv data in the Data folder.

It is expected to have each time series in a different csv.

It is also expected that the csv contains only a index (could be the timestamp) and the values of your series.

The name of the csv file is by default taken as the name of the series

# Important:
Tested on Pandas 0.24.2. It seems that Pandas 0.25 has a problem with Series.from_csv

# TODO:
Add validation mechanisms at fitting
