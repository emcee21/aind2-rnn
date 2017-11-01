import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [ series[idx - window_size:idx] for idx in range(window_size,len(series)) ]
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    return Sequential([
        LSTM(5, input_shape=(window_size,1)),
        Dense(1)
    ])


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    text = list(text)
    for idx, c in enumerate(text):
        text[idx] = (c if ((c >= 'a' and c <= 'z') or c in punctuation) else ' ')
    return ''.join(text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [ text[idx - window_size:idx] for idx in range(window_size,len(text),step_size) ]
    outputs = [ text[idx] for idx in range(window_size,len(text),step_size) ]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    return Sequential([
        LSTM(200, input_shape=(window_size,num_chars)),
        Dense(num_chars, activation='softmax')
    ])