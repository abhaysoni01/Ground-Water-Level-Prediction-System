import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSEMetric
import joblib
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def forecast_groundwater(dist, horizon, seq_length=12):
    df = pd.read_csv(os.path.join(project_root, 'data', 'processed_data.csv'))
    features = [col for col in df.columns if col not in ['state', 'district', 'year', 'month', 'avg_groundwater_level']]
    target = 'avg_groundwater_level'
    data = df[df['district'] == dist].sort_values(['year', 'month']).reset_index(drop=True)
    
    scaler = joblib.load(os.path.join(project_root, 'models', f'{dist}_scaler.pkl'))
    
    # Load model with custom objects for compatibility
    custom_objects = {
        'mse': MeanSquaredError(),
        'MeanSquaredError': MeanSquaredError,
        'MSEMetric': MSEMetric
    }
    try:
        model = load_model(os.path.join(project_root, 'models', f'{dist}_model.h5'), custom_objects=custom_objects)
    except:
        # Fallback: try loading without custom objects
        model = load_model(os.path.join(project_root, 'models', f'{dist}_model.h5'), compile=False)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    
    # Last seq_length data
    last_data = data.tail(seq_length)[features + [target]]
    last_scaled = scaler.transform(last_data)
    
    predictions = []
    current_seq = last_scaled[:, :-1]  # features
    
    for _ in range(horizon):
        pred_scaled = model.predict(current_seq.reshape(1, seq_length, -1), verbose=0)
        pred_inv = scaler.inverse_transform(np.concatenate([current_seq[-1:], pred_scaled], axis=1))[:, -1]
        predictions.append(pred_inv[0])
        
        # Update sequence approximately
        new_row = current_seq[-1:].copy()
        # Shift lags roughly
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = new_row[0]
        # Update the target lag
        current_seq[-1, features.index('groundwater_lag_1')] = pred_scaled[0,0]
    
    return predictions

def get_test_predictions(dist, seq_length=12):
    df = pd.read_csv(os.path.join(project_root, 'data', 'processed_data.csv'))
    features = [col for col in df.columns if col not in ['state', 'district', 'year', 'month', 'avg_groundwater_level']]
    target = 'avg_groundwater_level'
    data = df[df['district'] == dist].sort_values(['year', 'month']).reset_index(drop=True)
    
    scaler = joblib.load(os.path.join(project_root, 'models', f'{dist}_scaler.pkl'))
    
    # Load model with custom objects for compatibility
    custom_objects = {
        'mse': MeanSquaredError(),
        'MeanSquaredError': MeanSquaredError,
        'MSEMetric': MSEMetric
    }
    try:
        model = load_model(os.path.join(project_root, 'models', f'{dist}_model.h5'), custom_objects=custom_objects)
    except:
        # Fallback: try loading without custom objects
        model = load_model(os.path.join(project_root, 'models', f'{dist}_model.h5'), compile=False)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    
    # Last seq_length data
    last_data = data.tail(seq_length)[features + [target]]
    last_scaled = scaler.transform(last_data)
    
    predictions = []
    current_seq = last_scaled[:, :-1]  # features
    
    for _ in range(horizon):
        pred_scaled = model.predict(current_seq.reshape(1, seq_length, -1), verbose=0)
        pred_inv = scaler.inverse_transform(np.concatenate([current_seq[-1:], pred_scaled], axis=1))[:, -1]
        predictions.append(pred_inv[0])
        
        # Update sequence approximately
        new_row = current_seq[-1:].copy()
        # Shift lags roughly
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = new_row[0]
        # Update the target lag
        current_seq[-1, features.index('groundwater_lag_1')] = pred_scaled[0,0]
    
    return predictions

def get_test_predictions(dist, seq_length=12):
    df = pd.read_csv(os.path.join(project_root, 'data', 'processed_data.csv'))
    features = [col for col in df.columns if col not in ['state', 'district', 'year', 'month', 'avg_groundwater_level']]
    target = 'avg_groundwater_level'
    data = df[df['district'] == dist].sort_values(['year', 'month']).reset_index(drop=True)
    
    scaler = joblib.load(os.path.join(project_root, 'models', f'{dist}_scaler.pkl'))
    
    # Load model with custom objects for compatibility
    custom_objects = {
        'mse': MeanSquaredError(),
        'MeanSquaredError': MeanSquaredError,
        'MSEMetric': MSEMetric
    }
    try:
        model = load_model(os.path.join(project_root, 'models', f'{dist}_model.h5'), custom_objects=custom_objects)
    except:
        # Fallback: try loading without custom objects
        model = load_model(os.path.join(project_root, 'models', f'{dist}_model.h5'), compile=False)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    
    data_scaled = scaler.transform(data[features + [target]])
    data_scaled = pd.DataFrame(data_scaled, columns=features + [target])
    
    X = []
    y = []
    indices = []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled.iloc[i:i+seq_length][features].values)
        y.append(data_scaled.iloc[i+seq_length][target])
        indices.append(i + seq_length)
    X = np.array(X)
    y = np.array(y)
    
    train_size = int(0.8 * len(X))
    X_test = X[train_size:]
    y_test = y[train_size:]
    test_indices = indices[train_size:]
    
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred_scaled), len(features))), y_pred_scaled], axis=1))[:, -1]
    y_test_inv = scaler.inverse_transform(np.concatenate([np.zeros((len(y_test), len(features))), y_test.reshape(-1,1)], axis=1))[:, -1]
    
    return data.iloc[test_indices]['year'] + data.iloc[test_indices]['month']/12, y_test_inv, y_pred