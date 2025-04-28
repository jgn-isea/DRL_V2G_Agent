import numpy as np


def create_sequences(data, input_length=168*4, output_length=24*4):
    """
    Generates sequences of input and output data for time-series forecasting.

    Args:
        data (np.array): Scaled data as a numpy array.
        input_length (int): Number of time steps in the input sequence.
        output_length (int): Number of time steps in the output sequence.

    Returns:
        tuple: A tuple containing input (X) and output (y) sequences as numpy arrays.
    """
    X, y = [], []
    for i in range(len(data) - input_length - output_length):
        X.append(data[i:i + input_length])
        y.append(data[i + input_length:i + input_length + output_length])
    return np.array(X).squeeze(), np.array(y).squeeze()