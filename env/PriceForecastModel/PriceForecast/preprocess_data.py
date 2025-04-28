import numpy as np
from sklearn.preprocessing import MinMaxScaler

def subtract_first_element(arr):
    first_elements = arr[:, 0][:, np.newaxis]
    result = arr - first_elements
    return result

def preprocess_data(X_train, X_test, y_train, y_test, q_lb=0.15, q_ub=0.85):
    """
    Preprocesses the input data by handling outliers and scaling the data.

    Args:
        X_train (np.array): Training input data.
        X_test (np.array): Testing input data.
        y_train (np.array): Training target data.
        y_test (np.array): Testing target data.

    Returns:
        tuple: A tuple containing the scaled data (X_train, X_test, y_train, y_test) and the scaler objects used for transformation.
    """

    # Concatenate data to calculate quantiles for outlier clipping
    X_all = np.concatenate([X_train, X_test], axis=0)
    Y_all = np.concatenate([y_train, y_test], axis=0)
    data_all = np.concatenate([X_all, Y_all], axis=1)
    q_lb = np.quantile(data_all, q_lb)
    q_ub = np.quantile(data_all, q_ub)

    # Clip data to the range between the 15th and 85th quantiles
    X_train = np.clip(X_train, q_lb, q_ub)
    X_test = np.clip(X_test, q_lb, q_ub)
    y_train = np.clip(y_train, q_lb, q_ub)


    return X_train, X_test, y_train, y_test