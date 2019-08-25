## Benchmark for numerical time series forecast using Tensorflow Keras

## Under development, but works

It implements the sliding windowing. All windows are re-normalized at every run.

It has by default a stateful LSTM trained in batches. The default batch size is 10.

It takes a window for training and the next window for validation. It has stride = 1 by default. The default window size is 12

The methodology merges all series in one with multiple features.

At the class data.py, by default it takes the montly mean of each time series. It obviously can be changed

# Run
python main.py
## Optional new arguments
-d: The folder to look for your csv's (Default 'Data')

-f: The forecast unit (Default 1)

-t: The train percentage size (Default 0.3)

-w: The window size (Default 12 months)

-b: The batch size (Default 10)

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
