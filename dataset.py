# MODIFIED: dataset.py

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Define base directory for loading data
BASE_DIR = "./QEncoder_SP500_prediction/"
datafiles_dir = os.path.join(BASE_DIR, 'processed_data/')
dataset_dir = os.path.join(BASE_DIR, 'datasets/')


def load_and_split_csv(filename: str, train_end_date: str):
    """
    MODIFIED: Loads a CSV and performs a temporal split based on a date.
    This ensures the model is trained on past data and tested on future data,
    as specified in the reports[cite: 1185].
    """
    path = os.path.join(dataset_dir, filename)
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Temporal split
    train_df = df[df['Date'] <= train_end_date]
    test_df = df[df['Date'] > train_end_date]

    # Drop date column after splitting
    train_features = train_df.drop(columns=['Date', 'Name']).to_numpy()
    test_features = test_df.drop(columns=['Date', 'Name']).to_numpy()

    return train_features, test_features


def split_features_labels(features: np.ndarray, labels: np.ndarray, val_ratio: float = 0.2):
    """
    Splits features and labels into training and validation sets.
    """
    assert len(features) == len(labels), "Mismatch between features and labels length"
    split_idx = int(len(features) * (1 - val_ratio))
    X_train, X_val = features[:split_idx], features[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]
    return X_train, X_val, y_train, y_val


def create_windows(data, window_size=2):
    """
    MODIFIED: Creates windows based on the 2-day aggregation requirement[cite: 1159].
    Each input sample will contain 10 features (5 features/day * 2 days).
    The target variable is changed to the 'Close' price of the next day.
    """
    X, Y = [], []

    # Normalize each feature column
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    op_norm = normalized_data[:, 0]
    hi_norm = normalized_data[:, 1]
    lo_norm = normalized_data[:, 2]
    cl_norm = normalized_data[:, 3]
    vo_norm = normalized_data[:, 4]

    for i in range(len(normalized_data) - window_size):
        # Flatten the 2 days of 5 features into a single 10-element vector
        feature_window = np.hstack([
            op_norm[i:i + window_size],
            hi_norm[i:i + window_size],
            lo_norm[i:i + window_size],
            cl_norm[i:i + window_size],
            vo_norm[i:i + window_size]
        ])

        X.append(feature_window)
        # MODIFIED: Target is the 'Close' price of the day after the window
        Y.append(cl_norm[i + window_size])

    return np.array(X), np.array(Y)


def load_dataset(args):
    """
    MODIFIED: Simplified data loading logic. Caching is now consistent across all datasets.
    The S&P500 dataset now uses the same temporal split logic as the others.
    """
    dataset_map = {
        'sp500': ("combined_dataset.csv", "2017-12-31"),
        'nifty': ("NIFTY50_Cleaned_Data.csv", "2016-07-31"),
        'wti': ("WTI_Offshore_Cleaned_Data.csv", "2015-03-31")
    }

    if args.dataset not in dataset_map:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    filename, train_end_date = dataset_map[args.dataset]

    # Define cached file paths
    X_path = os.path.join(datafiles_dir, f"X_{args.dataset}.npy")
    Y_path = os.path.join(datafiles_dir, f"Y_{args.dataset}.npy")
    tX_path = os.path.join(datafiles_dir, f"tX_{args.dataset}.npy")
    tY_path = os.path.join(datafiles_dir, f"tY_{args.dataset}.npy")
    F_path = os.path.join(datafiles_dir, f"F_{args.dataset}.npy")

    if os.path.exists(X_path):
        print(f"Loading cached data for {args.dataset}...")
        X = np.load(X_path)
        Y = np.load(Y_path)
        tX = np.load(tX_path)
        tY = np.load(tY_path)
        flattened = np.load(F_path)
    else:
        print(f"Processing data for {args.dataset}...")
        X_train_raw, X_test_raw = load_and_split_csv(filename, train_end_date)

        # MODIFIED: Window size is now 2 for 2-day aggregation
        window_size = 2
        X, Y = create_windows(X_train_raw, window_size)
        tX, tY = create_windows(X_test_raw, window_size)

        # The 'flattened' data for the encoder is simply the training input X
        flattened = X

        # Save processed data to cache
        os.makedirs(datafiles_dir, exist_ok=True)
        np.save(X_path, X)
        np.save(Y_path, Y)
        np.save(tX_path, tX)
        np.save(tY_path, tY)
        np.save(F_path, flattened)
        print("Cached data saved.")

    return X, Y, tX, tY, flattened