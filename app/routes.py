from flask import Flask, jsonify, Blueprint, render_template, request, session, redirect, url_for
from flask_session import Session
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from numpy.polynomial import Chebyshev, Polynomial
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

# from .functions import construct_smooth
import zstandard as zstd
import matplotlib.dates as mdates
import pandas as pd
import plotly.express as px
import numpy as np
import json, uuid, time, threading, requests, traceback
import os
import zstandard
import pickle
import zlib
import re

app = Flask(__name__)
app.secret_key = 'aksldfj124nsafja'

app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

app.config['SECRET_KEY'] = 'aksldfj124nsafja'
app.config['SESSION_TYPE'] = 'filesystem'  # Or 'redis' for production
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# Inside the except block:
traceback.print_exc()

# Create a dictionary to store cached dataframes
cached_data = {}

main = Blueprint('main', __name__)

# Add prediction range constraint
class SmartConstrainedRegressor:
    def __init__(self, base_model, train_mean):
        self.base_model = base_model
        self.train_mean = train_mean
        
    def predict(self, X):
        y_pred = self.base_model.predict(X)
        # Apply smoothing towards training mean
        return 0.9 * y_pred + 0.1 * self.train_mean


@main.route('/test')
def test():
    return 'Test route works!'

def fetch_crypto_data_polygon(symbol):
    url = f'https://api.polygon.io/v2/aggs/ticker/X:{symbol}USD/range/1/day/2000-01-01/2023-01-09?apiKey={api_key_polygon}'
    response = requests.get(url)
    try:
        data = response.json()
        if 'results' in data:
            df = pd.DataFrame(data['results'])
            df['t'] = pd.to_datetime(df['t'], unit='ms')  # Convert timestamp to datetime
            df.set_index('t', inplace=True)
            # Rename columns
            df.rename(columns={'o': 'Open (USD)', 'h': 'High (USD)', 'l': 'Low (USD)', 'c': 'Close (USD)', 'v': 'Volume'}, inplace=True)
            return df
        else:
            print("No 'results' in response:", data)
            return None
    except ValueError as e:
        print("Error decoding JSON:", e, response.text)
        return None


def fetch_shares_outstanding(symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key_alpha}'
    response = requests.get(url).json()
    if 'SharesOutstanding' in response:
        return float(response['SharesOutstanding'])
    else:
        print(f"No shares outstanding data found for {symbol}.")
        return None  # Handle missing data gracefully


def fetch_stock_data_alpha_vantage(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key_alpha}&outputsize=full'
    response = requests.get(url).json()
    if 'Time Series (Daily)' in response:
        data = pd.DataFrame(response['Time Series (Daily)']).T
        data.index = pd.to_datetime(data.index)
        print(f"Stock data for {symbol} fetched successfully.")
        return data
    else:
        print(f"No data found for {symbol} from Alpha Vantage.")
        return None


def fetch_data(symbol):
    """Fetches data for a given symbol and returns the DataFrame or None.

    Args:
        symbol (str): The symbol of the asset to fetch data for.

    Returns:
        pandas.DataFrame: The DataFrame containing the fetched data, 
                          or None if data is not available.
    """

    data = fetch_crypto_data_polygon(symbol)
    is_stock = False
    if data is not None:
        print(f"Crypto data for {symbol} fetched successfully.")
        return data, is_stock

    data = fetch_stock_data_alpha_vantage(symbol)
    if data is not None:
        is_stock = True
        print(f"Stock data for {symbol} fetched successfully.")
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df.index)
        shares_outstanding = fetch_shares_outstanding(symbol)
        if shares_outstanding is not None:
            df['MarketCap'] = df['5. adjusted close'].astype(float) * shares_outstanding
        else:
            print(f"No shares outstanding data available for {symbol}.")

    return df, is_stock








# PIPELINE CODE
def best_fit_pipeline(X, y, degrees=[1, 2, 3, 4], alphas=np.logspace(-10, 2, 100), n_splits=10):
    """
    Finds the best regression model for time series data considering various transformations and models.
    
    Parameters:
    X (array-like): Normalized time values (shape: [n_samples, 1])
    y (array-like): Target values (asset prices)
    degrees (list): Polynomial degrees to try
    alphas (array-like): Regularization strengths for Ridge/Lasso
    n_splits (int): Number of time series cross-validation splits
    
    Returns:
    tuple: (best_model, best_config) containing the best model and its configuration
    """
    warnings.filterwarnings('ignore', category=UserWarning)  # Suppress scikit-learn warnings
    
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    best_mse = np.inf
    best_model = None
    best_config = {}

    linear_models = ['Ridge', 'Lasso', 'ElasticNet']  # Ridge, Lasso, ElasticNet
    tree_models = ['GBM']    # GBM, RandomForest, XGBoost, etc.
    
    gbm_params = {
        'max_iter': 500,
        'max_depth': 3,
        'min_samples_leaf': 5,
        'learning_rate': 0.1,
        'early_stopping': True
    }

    # Define model configurations
    models = []
    for degree in degrees:
        models += [
            (f'Ridge_degree{degree}', Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)),
                ('scaler', StandardScaler()),
                ('model', RidgeCV(alphas=alphas))
            ])),
            
            (f'Lasso_degree{degree}', Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)),
                ('scaler', StandardScaler()),
                ('model', LassoCV(alphas=alphas, max_iter=10000))
            ])),
            
            (f'OLS_degree{degree}', Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)),
                ('model', LinearRegression())
            ])),

            (f'ElasticNet_degree{degree}', Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)),
                ('scaler', StandardScaler()),
                ('model', ElasticNetCV(alphas=alphas, l1_ratio=[.1, .5, .9], max_iter=10000))
            ])),

            ('GBM', Pipeline([
                ('poly', PolynomialFeatures(degree=1)),
                ('model', HistGradientBoostingRegressor(**gbm_params))
            ]))
        ]

    # Apply to models
    constrained_models = []
    for name, model in models:
        constrained_models.append(
            (f'Constrained_{name}', Pipeline([
                ('model', model),
                ('constraint', SmartConstrainedRegressor(None, y.mean()))  # Wrapper
            ]))
        )
    models += constrained_models

    # Define transformations
    transformations = [
        ('none', None, None),
        ('log', np.log, np.exp),
        ('boxcox', PowerTransformer(method='box-cox', standardize=False), None)
    ]
    
    ts_cv = TimeSeriesSplit(
        n_splits=5,
        test_size=7,  # 7-period forecast
        gap=0
    )   

     # Add debug statistics upfront
    print("\n=== Input Data Statistics ===")
    print(f"X range: {X.min():.2f} to {X.max():.2f}")
    print(f"y range: {y.min():.2f} to {y.max():.2f} (mean {y.mean():.2f})")
    
    for trans_name, trans_func, inv_func in transformations:
        # Show transformation results
        try:
            if trans_name == 'log':
                y_trans = trans_func(y)
                print(f"\nLog transform stats: {y_trans.min():.2f} to {y_trans.max():.2f}")
            elif trans_name == 'boxcox':
                pt = PowerTransformer(method='box-cox')
                y_trans = pt.fit_transform(y.reshape(-1, 1)).flatten()
                print(f"BoxCox transform stats: {y_trans.min():.2f} to {y_trans.max():.2f}")
        except:
            pass
    
    for trans_name, trans_func, inv_func in transformations:
        for model_name, model in models:
            try:
                # # Handle transformations using TransformedTargetRegressor
                # if trans_name == 'boxcox' and 'GBM' in model_name:
                #     continue
                if trans_name == 'boxcox':
                    # Check for constant values in training fold
                    if len(np.unique(y_train)) == 1:
                        continue  # Skip this fold
                    tt_model.transformer_.fit(y_train.reshape(-1, 1))
                    check_is_fitted(tt_model.transformer_)
                else:
                    tt_model = TransformedTargetRegressor(
                        regressor=model,
                        func=trans_func,
                        inverse_func=inv_func
                    )
                
                mse_scores = []
                for train_idx, test_idx in ts_cv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    tt_model.fit(X_train, y_train)
                    y_pred = tt_model.predict(X_test)
                    mse_scores.append(mean_squared_error(y_test, y_pred))
                    
                    # Add debug prints for first fold
                    if len(mse_scores) == 0:
                        print(f"\n=== First Fold Debug: {trans_name}/{model_name} ===")
                        print(f"Training data range: {X_train.min():.2f}-{X_train.max():.2f}")
                        print(f"Training target range: {y_train.min():.2f}-{y_train.max():.2f}")
                        
                        # Check polynomial feature generation
                        if 'poly' in model.named_steps:
                            poly = model.named_steps['poly']
                            X_poly = poly.fit_transform(X_train)
                            print(f"Poly features (first sample): {X_poly[0]}")
                            
                        # Check model coefficients during fitting
                        if 'model' in model.named_steps:
                            model.named_steps['model'].fit(X_train, y_train)
                            if hasattr(model.named_steps['model'], 'coef_'):
                                coefs = model.named_steps['model'].coef_
                                print(f"Model coefficients: {coefs}")
                                print(f"Coef magnitude: {np.linalg.norm(coefs):.2f}")
                
                avg_mse = np.mean(mse_scores)
                
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_model = tt_model
                    best_config = {
                        'transformation': trans_name,
                        'model': model_name,
                        'mse': avg_mse
                    }
                    
            except Exception as e:
                print(f"Error with {trans_name}/{model_name}: {str(e)}")
                continue
    
    # After selecting best model
    print("\n=== Best Model Analysis ===")
    print(f"Configuration: {best_config}")
    
    # Check training data predictions
    y_pred_train = best_model.predict(X)
    print("\nTraining Actual vs Predicted:")
    for true, pred in zip(y[:5], y_pred_train[:5]):
        print(f"{true:.2f} -> {pred:.2f}")

    # Check feature importance
    if hasattr(best_model.regressor_, 'coef_'):
        coefs = best_model.regressor_.coef_
        print(f"\nFeature coefficients: {coefs}")
        print(f"L2 Norm: {np.linalg.norm(coefs):.2f}")


    # Refit best model on entire dataset
    if best_model is not None:
        best_model.fit(X, y)
        print(f"Best model: {best_config['model']} with {best_config['transformation']} transform (MSE: {best_config['mse']:.2e})")
    else:
        raise ValueError("No valid model could be fit")
    
    return best_model, best_config
# PIPELINE CODE






# POLYNOMIAL COMPRESSION FUNCTIONS
# Compression Functions

DEBUG = False  # Set to False to disable logging

# POLYNOMIAL COMPRESSION FUNCTIONS WITH DEBUGGING
def save_compressed_poly(file_path, coefficients, precision=12, metadata=None):
    assert len(coefficients.shape) == 1, "Coefficients must be 1D array"
    
    # Original coefficients logging
    if DEBUG:
        print("\n=== SAVE PROCESS ===")
        print(f"Original coefficients (first 3): {coefficients[:3]}...")
        print(f"Data type: {coefficients.dtype}")

    # Scaling and quantization
    scale_factor = 10**precision
    scaled_coeffs = np.round(coefficients * scale_factor).astype(np.int64)
    
    if DEBUG:
        print("\n=== SCALING STAGE ===")
        print(f"Scale factor: {scale_factor}")
        print(f"Example scaled values (first 3): {scaled_coeffs[:3]}...")
        print(f"Min/Max scaled: {scaled_coeffs.min()}, {scaled_coeffs.max()}")

    # LEB128 Encoding
    encoded_bytes = bytearray()
    if DEBUG:
        print("\n=== ENCODING STAGE ===")
        
    for i, coeff in enumerate(scaled_coeffs):
        original = coeff
        bytes_list = _int_to_sleb128(coeff)
        encoded_bytes.extend(bytes_list)
        
        if DEBUG and i < 3:  # Print first 3 coefficients' encoding
            print(f"Coefficient {i}:")
            print(f"  Original scaled: {original}")
            print(f"  Encoded bytes: {bytes_list}")
            print(f"  Hex representation: {bytes_list.hex()}")

    if DEBUG:
        print(f"\nTotal encoded bytes before compression: {len(encoded_bytes)}")

    # Zstandard Compression
    cctx = zstd.ZstdCompressor(level=22)
    compressed_data = cctx.compress(encoded_bytes)
    
    if DEBUG:
        print("\n=== COMPRESSION STAGE ===")
        print(f"Compressed size: {len(compressed_data)} bytes")
        print(f"Compression ratio: {len(compressed_data)/len(encoded_bytes):.1%}")

    # Metadata
    metadata = metadata or {}
    metadata.update({
        "precision": precision,
        "original_dtype": str(coefficients.dtype),
        "coeff_count": len(coefficients),
        "format": "scaled+leb128+zstd"
    })

    # Write to file
    with open(file_path, 'wb') as f:
        meta_json = json.dumps(metadata).encode('utf-8')
        f.write(len(meta_json).to_bytes(4, 'big'))
        f.write(meta_json)
        f.write(compressed_data)

def load_compressed_poly(file_path):
    with open(file_path, 'rb') as f:
        # Read metadata
        meta_len = int.from_bytes(f.read(4), 'big')
        metadata = json.loads(f.read(meta_len))
        
        if DEBUG:
            print("\n=== LOAD PROCESS ===")
            print(f"Metadata: {metadata}")

        # Decompress
        dctx = zstd.ZstdDecompressor()
        decoded = dctx.decompress(f.read())
        
        if DEBUG:
            print("\n=== DECOMPRESSION STAGE ===")
            print(f"Decompressed bytes length: {len(decoded)}")
            if len(decoded) < 100:
                print(f"First 20 bytes hex: {decoded[:20].hex()}")

        # LEB128 Decoding
        scaled_coeffs = _bytes_to_sleb128(decoded)
        
        if DEBUG:
            print("\n=== DECODING STAGE ===")
            print(f"Decoded coefficients count: {len(scaled_coeffs)}")
            print(f"First 3 decoded values: {scaled_coeffs[:3]}...")
            print(f"Min/Max decoded: {scaled_coeffs.min()}, {scaled_coeffs.max()}")

        # Validation
        if len(scaled_coeffs) != metadata['coeff_count']:
            raise ValueError(f"Mismatch: Expected {metadata['coeff_count']}, got {len(scaled_coeffs)}")

        # Reconstruction
        precision = metadata['precision']
        reconstructed = scaled_coeffs.astype(np.float64) / (10**precision)
        
        if DEBUG:
            print("\n=== RECONSTRUCTION ===")
            print(f"Using precision: {precision}")
            print(f"First 3 reconstructed values: {reconstructed[:3]}...")
            print(f"Reconstructed dtype: {reconstructed.dtype}")

        return reconstructed, metadata

# DEBUG-ENHANCED LEB128 FUNCTIONS
def _int_to_sleb128(value):
    bytes_list = []
    original = value
    if DEBUG and abs(value) > 2**32:
        print(f"  Encoding large value: {value}")
        
    while True:
        byte = value & 0x7F
        value >>= 7
        sign_bit = byte & 0x40
        
        if (value != -1 or not sign_bit) and (value != 0 or sign_bit):
            bytes_list.append(byte | 0x80)
        else:
            bytes_list.append(byte)
            break

    if DEBUG and abs(original) > 1000:
        print(f"  Value {original} encoded as {len(bytes_list)} bytes")
        
    return bytes(bytes_list)


def _bytes_to_sleb128(byte_data):
    values = []
    stream = list(byte_data)
    
    if DEBUG:
        print("\n=== BYTE DECODING ===")
        print(f"Total bytes to decode: {len(stream)}")
    
    while stream:
        result = 0
        shift = 0
        byte = 0x80
        bytes_read = []
        
        while byte & 0x80:
            byte = stream.pop(0)
            bytes_read.append(byte)
            result |= (byte & 0x7F) << shift
            shift += 7

        if bytes_read and (bytes_read[-1] & 0x40):
            result |= -(1 << shift)
            
        values.append(result)
        
        if DEBUG and len(values) <= 3:
            print(f"Decoded value {len(values)-1}:")
            print(f"  Bytes: {bytes_read}")
            print(f"  Raw result: {result}")
            print(f"  Hex: {hex(result)}")

    return np.array(values, dtype=np.int64)
# END POLYNOMIAL COMPRESSION FUNCTIONS





# # COMPRESSION FUNCTIONS
def encode_split(values, decimals=8):
    """Split values into high/low 4-decimal components"""
    scaled = np.round(values * 10**decimals).astype(np.int64)
    split_factor = 10**(decimals//2)
    return (scaled // split_factor).astype(np.int32), (scaled % split_factor).astype(np.int32)

def save_dual_compressed(file_path, values, decimals=8):
    # Split into high/low components
    high, low = encode_split(values, decimals)
    print("The first 10 values of high are:", high[:10])
    print("The first 10 values of low are:", low[:10])
    
    # Create compression pipeline for both components
    def compress_component(arr):
        deltas = np.empty_like(arr)
        deltas[0] = arr[0]
        deltas[1:] = arr[1:] - arr[:-1]
        
        encoded = bytearray()
        for delta in deltas:
            encoded.extend(_int_to_sleb128_reg(delta))
        return zstd.ZstdCompressor(level=22).compress(encoded)
    
    compressed_high = compress_component(high)
    compressed_low = compress_component(low)
    
    # Package metadata
    metadata = {
        "original_length": len(values),
        "decimals": decimals,
        "format": "dual-stream-delta+leb128+zstd"
    }
    
    # Write to single file
    with open(file_path, 'wb') as f:
        # Write metadata header
        meta_json = json.dumps(metadata).encode('utf-8')
        f.write(len(meta_json).to_bytes(4, 'big'))
        f.write(meta_json)
        
        # Write compressed streams with length prefixes
        for data in [compressed_high, compressed_low]:
            f.write(len(data).to_bytes(4, 'big'))
            f.write(data)

def load_dual_compressed(file_path):
    with open(file_path, 'rb') as f:
        # Read metadata
        meta_len = int.from_bytes(f.read(4), 'big')
        metadata = json.loads(f.read(meta_len))
        original_length = metadata["original_length"]
        decimals = metadata.get("decimals", 8)
        split_factor = 10**(decimals//2)
        
        # Read and decompress both components
        dctx = zstd.ZstdDecompressor()
        
        def decompress_component():
            data_len = int.from_bytes(f.read(4), 'big')
            return np.cumsum(_bytes_to_sleb128_reg(dctx.decompress(f.read(data_len)))).astype(np.int64)
        
        high = decompress_component()
        low = decompress_component()

        print("The first 10 values of low after decompression are:", low[:10])
        
        # Reconstruct original values
        combined = (high * split_factor + low).astype(np.float64) / 10**decimals
        return combined[:original_length]

# LEB128 helper functions (same as original implementation)
def _int_to_sleb128_reg(value):
    bytes_list = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if (value == 0 and (byte & 0x40) == 0) or (value == -1 and (byte & 0x40)):
            bytes_list.append(byte)
            break
        bytes_list.append(byte | 0x80)
    return bytes(bytes_list)

def _bytes_to_sleb128_reg(byte_data):
    values = []
    stream = list(byte_data)
    while stream:
        result = 0
        shift = 0
        while True:
            byte = stream.pop(0)
            result |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                if (byte & 0x40) and (shift < 64):  # Handle 64-bit sign extension
                    result |= -(1 << shift)
                break
        values.append(result)
    return np.array(values, dtype=np.int64)


def save_compressed(file_path, normalized_y_values, total_investment, smooth_value):
    # Step 1: Scale to 8 decimal points and split into two parts
    scaled_8 = np.round(normalized_y_values * 1e8).astype(np.int64)
    first_part = (scaled_8 // 10000).astype(np.int64)  # First 4 decimals as integer
    second_part = (scaled_8 % 10000).astype(np.int64)  # Last 4 decimals as integer
    
    # Step 2: Delta encoding for both parts
    deltas1 = np.empty_like(first_part)
    deltas1[0] = first_part[0]
    deltas1[1:] = first_part[1:] - first_part[:-1]
    
    deltas2 = np.empty_like(second_part)
    deltas2[0] = second_part[0]
    deltas2[1:] = second_part[1:] - second_part[:-1]
    
    # Step 3: Signed LEB128 encoding for both deltas
    encoded_bytes1 = bytearray()
    for delta in deltas1:
        encoded_bytes1.extend(_int_to_sleb128_reg(delta))
    
    encoded_bytes2 = bytearray()
    for delta in deltas2:
        encoded_bytes2.extend(_int_to_sleb128_reg(delta))
    
    # Step 4: Zstandard compression
    cctx = zstd.ZstdCompressor(level=22)
    compressed1 = cctx.compress(encoded_bytes1)
    compressed2 = cctx.compress(encoded_bytes2)
    
    # Update metadata
    metadata = {
        "total_investment": total_investment,
        "smooth_value": smooth_value,
        "n": len(scaled_8),
        "format": "delta+leb128+zstd+split8"
    }
    
    # Write to file with metadata and both compressed parts
    with open(file_path, 'wb') as f:
        # Write metadata
        meta_json = json.dumps(metadata).encode('utf-8')
        f.write(len(meta_json).to_bytes(4, 'big'))
        f.write(meta_json)
        
        # Write first compressed part with length prefix
        f.write(len(compressed1).to_bytes(4, 'big'))
        f.write(compressed1)
        
        # Write second compressed part with length prefix
        f.write(len(compressed2).to_bytes(4, 'big'))
        f.write(compressed2)


def load_compressed(file_path):
    with open(file_path, 'rb') as f:
        # Read metadata
        meta_len = int.from_bytes(f.read(4), 'big')
        metadata = json.loads(f.read(meta_len))
        
        # Read first compressed part
        compressed1_len = int.from_bytes(f.read(4), 'big')
        compressed1 = f.read(compressed1_len)
        
        # Read second compressed part
        compressed2_len = int.from_bytes(f.read(4), 'big')
        compressed2 = f.read(compressed2_len)
        
        # Decompress both parts
        dctx = zstd.ZstdDecompressor()
        decoded1 = dctx.decompress(compressed1)
        decoded2 = dctx.decompress(compressed2)
        
        # Decode LEB128 for both parts
        deltas1 = _bytes_to_sleb128_reg(decoded1)
        deltas2 = _bytes_to_sleb128_reg(decoded2)
        
        # Reconstruct original parts
        first_part = np.cumsum(deltas1).astype(np.int64)
        second_part = np.cumsum(deltas2).astype(np.int64)
        
        # Combine parts to restore 8-decimal values
        scaled_8 = first_part * 10000 + second_part
        return scaled_8 / 1e8, metadata

# # END COMPRESSION FUNCTIONS






# COMPUTE THE BEST SMOOTH VALUE: 
def compute_lowest_mse_smooth(data, degree):
    arr = [1, 13, 27, 39, 50]
    
    # Time the computation
    timer = time.time()
    for i in arr:


        downsampled, total_investment, _ = construct_smooth(data, i)

        if (i == arr[0]):
            len_original = len(downsampled)

        downsampled = downsampled + total_investment

        # Convert DatetimeIndex to numeric
        x_numeric = mdates.date2num(downsampled.index)

        # Normalize x-values to mean=0, std=1
        x_mean = x_numeric.mean()
        x_std = x_numeric.std()
        x_normalized = (x_numeric - x_mean) / x_std  # Now in ~[-1, 1] range

        n = len(x_normalized)

        print("The length of the mse test at degree", i, "is:", n)
        
        # threshold1 = int(n * 0.90)
        threshold2 = int(n * 0.95)  # 90% of the data

        # Assign weights
        weights = np.ones(n)
        # weights[threshold1:] = 10
        weights[threshold2:] = 5  # Assign weight of 10 to the last 10%

        downsampled_log = np.log(downsampled)

        # # ORIGINAL FUNCTION:
        coefficients = np.polyfit(x_normalized, downsampled_log.values, degree, w=weights)
        f = np.poly1d(coefficients)

        # Generate x-values for evaluation (numeric)
        x_numeric_eval = np.linspace(x_numeric.min(), x_numeric.max(), len(downsampled_log))

        # Normalize evaluation points using SAME mean/std as training data
        x_eval_normalized = (x_numeric_eval - x_mean) / x_std

        # Evaluate the polynomial (on log-transformed data)
        y_values_numeric_log = f(x_eval_normalized)

        #mse = mean_squared_error(downsampled_log.values, y_values_numeric_log)

        # weight_factor = 1.001 ** i
        # weight_factor = int(len_original / n)

        # print("The weight factor is:", weight_factor)

        # Calculate weighted MSE
        weighted_mse = np.mean(weights * (downsampled_log.values - y_values_numeric_log) ** 2)
        print("the weighted mse is:", weighted_mse, "for a smooth value of:", i)

        if (i == arr[0]):
            lowest_mse = weighted_mse
            lowest_smooth = i
        elif (weighted_mse < lowest_mse):
            lowest_mse = weighted_mse
            lowest_smooth = i

    timer = time.time() - timer
    print("the time taken is:", timer)
    print("the lowest mse is:", lowest_mse, "for a smooth value of:", lowest_smooth)

    return lowest_mse, lowest_smooth


def downsample_data(data, n):
    """
    Downsamples a pandas Series by selecting every nth element, 
    but includes global min/max within each interval if they 
    are the global min/max seen thus far.

    Args:
        data: A pandas Series.
        n: The interval for downsampling.

    Returns:
        A pandas Series with downsampled data.
    """

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if n >= len(data):
        return data  # Return original if n is too large.

    downsampled_data = []
    downsampled_indices = []

    min_val = data.iloc[0]
    max_val = data.iloc[0]

    min_val_interval = data.iloc[0]
    max_val_interval = data.iloc[0]

    for i in range(len(data)):
        if data.iloc[i] < min_val:
            min_val = data.iloc[i]
            min_val_interval = min_val
        if data.iloc[i] > max_val:
            max_val = data.iloc[i]
            max_val_interval = max_val

        if i % n == 0:
            if max_val_interval == max_val:
                downsampled_data.append(max_val_interval)
                downsampled_indices.append(data.index[i])
            elif min_val_interval == min_val:
                downsampled_data.append(min_val_interval)
                downsampled_indices.append(data.index[i])
            else:
                downsampled_data.append(data.iloc[i])
                downsampled_indices.append(data.index[i])

            if i < (len(data) - 1):
                min_val_interval = data.iloc[i + 1]
                max_val_interval = data.iloc[i + 1]

    return pd.Series(downsampled_data, index=downsampled_indices)


def construct_smooth(data, smooth_value):
    
    print(":smooth_value:", smooth_value)
    slider_values = data.get('sliderValues', {})

    cache_id = data.get('cache_id')

    if not cache_id:
        return jsonify({'error': 'Missing cache ID'}), 400

    cache_entry = DATA_CACHE.get(cache_id)
    if not cache_entry:
        return jsonify({'error': 'Invalid or expired cache ID'}), 400

    cached_data = cache_entry['cached_data']
    print(f"Processing cache ID: {cache_id}")

    cached_assets = cache_entry['assets']
    print("cached_assets are:", cached_assets)


    # Validate data structure
    if not all('dates' in d and 'investment' in d for d in cached_data.values()):
        raise ValueError("Invalid cached data structure")

    # Create a DataFrame with all dates
    all_dates = sorted(set().union(*[d['dates'] for d in cached_data.values()]))
    # num_points = len(all_dates)
    # print("THE number of points is:", num_points)
    df = pd.DataFrame(index=all_dates)
    total_investment = 0

    
    investments = ''
    investment_amounts = []

    # Add weighted investments
    for symbol, weight_pct in slider_values.items():
        if symbol in cached_data:
            weight = weight_pct / 100.0
            # print("GETTING HERE: weight is:", weight, "for symbol:", symbol)
            symbol_data = cached_data[symbol]
            temp_df = pd.DataFrame({
                'investment': symbol_data['investment'],
                'date': symbol_data['dates'],
                'weight': symbol_data['weight']
            }).set_index('date')
            df[symbol] = temp_df['investment'] * weight
            
            investment_amounts.append(symbol_data['weight'] * weight)
            total_investment += symbol_data['weight'] * weight

    print("investment_amounts are:", investment_amounts)

    for i, (symbol, _, _, is_reg) in enumerate(cached_assets):
        # updated_amount = amount / investment_amounts[i]
        if (is_reg):
            # print("The value of investment_amounts[i] is:", updated_amount, "and the value of i is:", i, "and amount is:", amount)
            investments += symbol.upper() + str(int(investment_amounts[i])) + "," 
        else: 
            investments += symbol.upper() + str(int(investment_amounts[i])) + "',"

    investments = investments[:-1]
    print("investments are:", investments)

    # Sum across all symbols and handle NaN
    combined = df.sum(axis=1).fillna(0)

    # Downsample the data
    try:
        downsampled = downsample_data(combined, smooth_value)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return downsampled, total_investment, investments


def print_polynomials(coefficients):
    # coefficients = f.coefficients.tolist()
    degree = len(coefficients) - 1

    polynomial_str = ""

    for i, coef in enumerate(coefficients):
        power = degree - i

        if coef != 0:  # Skip terms with zero coefficients
            if coef > 0 and polynomial_str:  # Add '+' if not the first term
                polynomial_str += " + "
            elif coef < 0:
                polynomial_str += " - "
                coef = abs(coef)  # Make coefficient positive for display

            if power == 0:  # Constant term
                polynomial_str += f"{coef:.10f}"
            elif power == 1:  # Linear term
                if coef != 1:  # Don't show '1' if the coefficient is 1
                    polynomial_str += f"{coef:.10f}x"
                else:
                    polynomial_str += "x"
            else:  # Higher-degree terms
                if coef != 1:
                    polynomial_str += f"{coef:.10f}x^{power}"
                else:
                    polynomial_str += f"x^{power}"

    print(f"Polynomial Function: {polynomial_str}")


def find_optimal_degree(x_normalized, downsampled_log):
    lowest_mse = 0
    lowest_degree_mse = 0

    n = len(x_normalized)
    threshold = int(n * 0.95)  # 90% of the data

    # Assign weights
    weights = np.ones(n)
    weights[threshold:] = 10

    for i in range(1, 30):

        # Fit Chebyshev polynomial
        coefficients = np.polyfit(x_normalized, downsampled_log.values, i, w=weights)
        f = np.poly1d(coefficients)
      
        # Generate x-values for evaluation (numeric)
        x_numeric_eval = np.linspace(x_normalized.min(), x_normalized.max(), len(downsampled_log))

        # Evaluate the polynomial (on log-transformed data)
        y_values_numeric_log = f(x_numeric_eval)

        
        # THIS MAY FAIL IF USING THE MEAN SQUARED ERROR FUNCTION
        mse = mean_squared_error(downsampled_log.values, y_values_numeric_log)
        
        if (i == 1):
            lowest_mse = mse
            lowest_degree_mse = i
        elif (mse < lowest_mse):
            lowest_mse = mse
            lowest_degree_mse = i

    print(f"Lowest Mean Squared Error: {lowest_mse} at degree {lowest_degree_mse}")
    return lowest_mse, lowest_degree_mse


def construct_derivative_function(f, x_eval_normalized, x_std, y_values_numeric):
    # Create derivative of the polynomial
    f_derivative = f.deriv()

    # Evaluate the derivative polynomial
    y_values_derivative_log = f_derivative(x_eval_normalized) / x_std

    y_values_derivative_log = y_values_derivative_log.tolist()

    # Evaluate derivative (original space, if needed)
    y_derivative = y_values_numeric * y_values_derivative_log  # Step 3 (optional)
    y_derivative_reg = y_derivative.tolist()

    return y_values_derivative_log, y_derivative_reg, f_derivative


def print_rmse(y_values_numeric_log, y_values_numeric, x_numeric_eval, y_values_derivative_log, y_derivative_reg):
    # Compute finite differences of log(y) w.r.t. x
    delta_x = np.diff(x_numeric_eval)  # x spacing (should be uniform due to linspace)
    dfdx_finite_diff = np.gradient(y_values_numeric_log, x_numeric_eval)

    # Compute finite differences of y w.r.t. x
    dydx_finite_diff = np.gradient(y_values_numeric, x_numeric_eval)

    # LOG SPACE DERIVATIVE RMSE
    rmse_log = np.sqrt(np.mean((y_values_derivative_log - dfdx_finite_diff)**2))
    print(f"Log-space derivative RMSE: {rmse_log}")

    # ORIGINAL SPACE DERIVATIVE RMSE
    rmse_original = np.sqrt(np.mean((y_derivative_reg - dydx_finite_diff)**2))
    print(f"Original-space derivative RMSE: {rmse_original}")

    # RELATIVE ERROR:
    relative_error = rmse_original / np.mean(np.abs(y_derivative_reg))
    print(f"Relative error: {relative_error * 100:.2f}%")


@main.route('/')
def index():
    regular_count = 1  # Provide default values for testing
    inverse_count = 1
    return render_template('index.html', regular_count=regular_count, inverse_count=inverse_count)

@main.route('/set_screen_width', methods=['POST'])
def set_screen_width():
    screen_width = request.json.get('width')
    session['screen_width'] = screen_width
    return '', 204


# PLOT TAKES AS INPUT TWO CHARTS AND PLOTS THEIR RELATIVE CHART
@main.route('/plot', methods=['POST'])
def plot():
    chart1 = request.form.get('simple_chart1')
    chart2 = request.form.get('simple_chart2')
    screen_width = session.get('screen_width', 1200)  # Default width if not set

    def fetch_data(symbol):
        # First try Polygon for crypto data
        data = fetch_crypto_data_polygon(symbol)
        isStock = False
        if data is not None:
            print(f"Crypto data for {symbol} fetched successfully.")
            return data, isStock  # Return if crypto data is found

        # Fall back to Alpha Vantage for stock data
        data = fetch_stock_data_alpha_vantage(symbol)
        if data is not None:
            isStock = True
            print(f"Stock data for {symbol} fetched successfully.")
            return data, isStock

        print(f"No data found for {symbol} from both APIs.")
        return None, isStock

    # Fetch data for both charts
    data1, isStock1 = fetch_data(chart1)
    data2, isStock2 = fetch_data(chart2)

    if data1 is None:
        print(f"No data available for {chart1}.")
        return "No data available for Chart 1"
    if data2 is None:
        print(f"No data available for {chart2}.")
        return "No data available for Chart 2"

    # Use price data for crypto and market cap for stocks
    if isStock1 and '5. adjusted close' in data1.columns:
        shares_outstanding1 = fetch_shares_outstanding(chart1)
        if shares_outstanding1 is not None:
            data1['MarketCap'] = data1['5. adjusted close'].astype(float) * shares_outstanding1
        else:
            return f"No shares outstanding data available for {chart1}"
    elif 'Close (USD)' in data1.columns:
        data1['MarketCap'] = data1['Close (USD)'].astype(float)
    else:
        return f"MarketCap data not available for {chart1}"

    if isStock2 and '5. adjusted close' in data2.columns:
        shares_outstanding2 = fetch_shares_outstanding(chart2)
        if shares_outstanding2 is not None:
            data2['MarketCap'] = data2['5. adjusted close'].astype(float) * shares_outstanding2
        else:
            return f"No shares outstanding data available for {chart2}"
    elif 'Close (USD)' in data2.columns:
        data2['MarketCap'] = data2['Close (USD)'].astype(float)
    else:
        return f"MarketCap data not available for {chart2}"

    print("Data1 Columns:", data1.columns)
    print("Data2 Columns:", data2.columns)
    print("Data1 Sample:", data1.head())
    print("Data2 Sample:", data2.head())

    # Interpolate and synchronize data
    data1 = data1.reindex(pd.date_range(data1.index.min(), data1.index.max())).interpolate(method='time')
    data2 = data2.reindex(pd.date_range(data2.index.min(), data2.index.max())).interpolate(method='time')

    print("Data1 After Interpolation:", data1.head())
    print("Data2 After Interpolation:", data2.head())

    # Merging and plotting (the same as before)
    merged_data = pd.merge(data1, data2, left_index=True, right_index=True, suffixes=(f'_{chart1}', f'_{chart2}'))
    print("Merged Data Columns:", merged_data.columns)
    print("Merged Data Sample:", merged_data.head())

    # Check if the 'MarketCap' columns exist after merge
    # if f'MarketCap_{chart1}' not in merged_data.columns or f'MarketCap_{chart2}' not in merged_data.columns:
    #     return "No Market Cap data to merge"

    merged_data['RelativeMarketCap'] = merged_data[f'MarketCap_{chart1}'] / merged_data[f'MarketCap_{chart2}']
    initial_point = merged_data['RelativeMarketCap'].iloc[0]
    final_point = merged_data['RelativeMarketCap'].iloc[-1]
    percent_change = ((final_point - initial_point) / initial_point) * 100

    if (percent_change > 0):
        percent_change = "+ " + str(round(percent_change, 2)) + "%"
    else:
        percent_change = "- " + str(abs(round(percent_change, 2))) + "%"

    # Plot using Plotly
    fig_linear = px.line(merged_data, y='RelativeMarketCap', title=f'{chart1} Relative to {chart2} ({percent_change}) on a Linear Scale')
    fig_linear.update_layout(width=screen_width)  # Set the width dynamically
    fig_log = px.line(merged_data, y='RelativeMarketCap', title=f'{chart1} Relative to {chart2} ({percent_change}) on a Logarithmic Scale')
    fig_log.update_layout(yaxis_type="log", width=screen_width)  # Set the width dynamically

    graph_html_linear = fig_linear.to_html(full_html=False)
    graph_html_log = fig_log.to_html(full_html=False)

    print("Getting here before error:")

    plots = {'linear': graph_html_linear, 'logarithmic': graph_html_log}

    return render_template('_plot_container.html', graph_html_linear=graph_html_linear, graph_html_log=graph_html_log)



# PLOT COMPLEX IS A MORE COMPLEX VERSION OF PLOT
@main.route('/plot_complex', methods=['POST'])
def plot_complex():

    def fetch_data(symbol):
        # First try Polygon for crypto data
        data = fetch_crypto_data_polygon(symbol)
        isStock = False
        if data is not None:
            print(f"Crypto data for {symbol} fetched successfully.")
            return data, isStock  # Return if crypto data is found

        # Fall back to Alpha Vantage for stock data
        data = fetch_stock_data_alpha_vantage(symbol)
        if data is not None:
            isStock = True
            print(f"Stock data for {symbol} fetched successfully.")

            # Convert the data to a DataFrame if it's not already
            df = pd.DataFrame(data)

            # Ensure the date column is in datetime format
            df['date'] = pd.to_datetime(df.index)

            # Calculate market cap
            shares_outstanding = fetch_shares_outstanding(symbol)
            if shares_outstanding is not None:
                df['MarketCap'] = df['5. adjusted close'].astype(float) * shares_outstanding
            else:
                print(f"No shares outstanding data available for {symbol}.")
                return None, isStock

            return df, isStock

        print(f"No data found for {symbol} from both APIs.")
        return None, isStock

    num_assets = []
    den_assets = []

    for key, value in request.form.items():
        print(f'Form key: {key}, value: {value}')
        if key.startswith('num_chart'):
            index = key.replace('num_chart', '')
            amount_key = f'num_amount{index}'
            amount_value = float(request.form.get(amount_key, 1))
            num_assets.append((value, amount_value))
            print(f'Numerator: {value} with amount {amount_value}')
        elif key.startswith('den_chart'):
            index = key.replace('den_chart', '')
            amount_key = f'den_amount{index}'
            amount_value = float(request.form.get(amount_key, 1))
            den_assets.append((value, amount_value))
            print(f'Denominator: {value} with amount {amount_value}')

    def fetch_bundle_data(assets, invert=False):
        bundle_data = pd.DataFrame()
        earliest_date = None
        for symbol, amount in assets:
            print(f"Fetching data for symbol: {symbol} with amount {amount}")
            data, isStock = fetch_data(symbol)
            if data is None:
                print(f"No data available for {symbol}.")
                continue

            # Calculate percentage gain/loss from IPO date
            initial_market_cap = data['MarketCap'].iloc[-1]
            data['RelativeMarketCap'] = (data['MarketCap'] / initial_market_cap + 0) * amount

            # Determine the earliest date
            if earliest_date is None or data.index.min() < earliest_date:
                earliest_date = data.index.min()

            # Invert values if it's a short position
            if invert:
                data['RelativeMarketCap'] *= -1

            # Fill pre-IPO dates with zero
            ipo_date = data.index.min()
            pre_ipo_dates = pd.date_range(earliest_date, ipo_date - pd.Timedelta(days=1))
            pre_ipo_data = pd.DataFrame(index=pre_ipo_dates, columns=['RelativeMarketCap'])
            pre_ipo_data['RelativeMarketCap'] = 0

            # Combine pre-IPO data with actual data
            data = pd.concat([pre_ipo_data, data])

            if bundle_data.empty:
                bundle_data = data[['RelativeMarketCap']]
            else:
                bundle_data = bundle_data.add(data[['RelativeMarketCap']], fill_value=0)

            print(f'Adding data for {symbol} with amount {amount}')

        return bundle_data

    # Fetch and combine data for numerator and denominator bundles
    numerator_data = fetch_bundle_data(num_assets)
    denominator_data = fetch_bundle_data(den_assets, invert=True)  # Invert for short positions

    if numerator_data is None or denominator_data is None:
        print("No data available for one or more assets in the bundles.")
        return "No data available for one or more assets in the bundles."

    # Combine numerator and denominator bundles
    merged_data = pd.merge(numerator_data, denominator_data, left_index=True, right_index=True, suffixes=('_num', '_den'))
    merged_data['CombinedRelativeMarketCap'] = merged_data['RelativeMarketCap_num'] + merged_data['RelativeMarketCap_den']

    print("Merged Data Sample:", merged_data.head())

    # Get screen width from session
    screen_width = session.get('screen_width', 1200)  # Default width if not set

    # Plot using Plotly
    fig_linear = px.line(merged_data, y='CombinedRelativeMarketCap', title='Combined Relative Market Cap (Linear Scale)')
    fig_linear.update_layout(width=screen_width)  # Set the width dynamically

    fig_log = px.line(merged_data, y='CombinedRelativeMarketCap', title='Combined Relative Market Cap (Logarithmic Scale)')
    fig_log.update_layout(yaxis_type="log", width=screen_width)  # Set the width dynamically

    graph_html_linear = fig_linear.to_html(full_html=False)
    graph_html_log = fig_log.to_html(full_html=False)

    return render_template('plot.html', graph_html_linear=graph_html_linear, graph_html_log=graph_html_log)


@main.route('/update_slider', methods=['POST'])
def update_slider():
    try:
        data = request.get_json()
        slider_values = data.get('sliderValues', {})
        cache_id = data.get('cache_id')

        if not cache_id:
            return jsonify({'error': 'Missing cache ID'}), 400

        cache_entry = DATA_CACHE.get(cache_id)
        if not cache_entry:
            return jsonify({'error': 'Invalid or expired cache ID'}), 400

        cached_data = cache_entry['cached_data']
        print(f"Processing cache ID: {cache_id}")

        # Validate data structure
        if not all('dates' in d and 'investment' in d for d in cached_data.values()):
            raise ValueError("Invalid cached data structure")

        # Create a DataFrame with all dates
        all_dates = sorted(set().union(*[d['dates'] for d in cached_data.values()]))
        df = pd.DataFrame(index=all_dates)
        total_investment = 0

        # Add weighted investments
        for symbol, weight_pct in slider_values.items():
            if symbol in cached_data:
                weight = weight_pct / 100.0
                # print("GETTING HERE: weight is:", weight, "for symbol:", symbol)
                symbol_data = cached_data[symbol]
                temp_df = pd.DataFrame({
                    'investment': symbol_data['investment'],
                    'date': symbol_data['dates'],
                    'weight': symbol_data['weight']
                }).set_index('date')
                df[symbol] = temp_df['investment'] * weight
                total_investment += symbol_data['weight'] * weight

        # Sum across all symbols and handle NaN
        combined = df.sum(axis=1).fillna(0)
        combined += total_investment

        return jsonify({
            'success': True,
            'plotData': {
                'x': combined.index.tolist(),
                'y': combined.round(2).tolist()
            },
            'totalInvestment': total_investment
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        traceback.print_exc()  # Add this for detailed error logging
        return jsonify({'error': f"Processing error: {str(e)}"}), 500



# THIS IS THE STARTING POINT OF THE APPLICATION WHICH RUNS THE MAIN ROUTE PLOT_TEST
@main.route('/plot_test', methods=['GET', 'POST'])
def plot_test():
    all_assets = []
    plots = None  # Initialize plots outside the conditional block
    cached_data = {}  # Initialize cached_data here

    if request.method == 'POST':
        regular_count = int(request.form.get('regular_count', 0)) # Default 0 if missing
        inverse_count = int(request.form.get('inverse_count', 0)) # Default 0 if missing

        regular_assets = []
        inverse_assets = []
        initial_all_equal = []
        total_weight = 0
        inverse_saved_assets = []
        regular_saved_assets = []

        # Collect asset data with weights and slider values
        for i in range(regular_count):
            symbol = request.form.get(f'regular_test_chart{i}')
            weight = float(request.form.get(f'regular_test_weight{i}', 1))
            slider_value = float(session.get(f'regular_test_chart{i}_slider_value', 100))
            if symbol: # Only add if symbol is not empty
                regular_assets.append((symbol, weight, slider_value))
                total_weight += weight

        for i in range(inverse_count):
            symbol = request.form.get(f'inverse_test_chart{i}')
            weight = float(request.form.get(f'inverse_test_weight{i}', 1))
            slider_value = float(session.get(f'inverse_test_chart{i}_slider_value', 100))
            if symbol: # Only add if symbol is not empty
                inverse_assets.append((symbol, weight, slider_value))
                total_weight += weight

        if not regular_assets and not inverse_assets:
            return "No data available for the specified symbols."

        total_weight = sum(weight for _, weight, _ in regular_assets + inverse_assets)

        all_assets = []
        for symbol, weight, slider_value in regular_assets:
            all_assets.append((symbol, weight, slider_value, True))  # True for regular asset

        for symbol, weight, slider_value in inverse_assets:
            all_assets.append((symbol, weight, slider_value, False)) # False for inverse asset

        # Process slider values from POST request
        if 'slider_values' in request.form:
            slider_values = request.form.getlist('slider_values[]')
            for i, (symbol, _, _, _) in enumerate(all_assets):
                asset_id = f"{symbol}_test_chart"
                slider_value = float(slider_values[i]) / 100
                all_assets[i] = (symbol, slider_value * total_weight, slider_value, all_assets[i][3]) # Keep the boolean

        # **Crucial: Recalculate total_weight based on updated weights**
        total_weight = sum(weight for symbol, weight, _, _ in all_assets)

       # Construct the combined dataframe
        combined_df = None
        symbol_dataframes = {}  # Store dataframes by symbol and type

        for symbol, weight, slider_value, is_regular in all_assets:
            data, is_stock = fetch_data(symbol)

            if data is None:
                print(f"No data available for {symbol}")
                continue

            df = data.copy()
            if is_stock:
                df['date'] = pd.to_datetime(df.index)
                shares_outstanding = fetch_shares_outstanding(symbol)
                if shares_outstanding is not None:
                    df['MarketCap'] = df['5. adjusted close'].astype(float) * shares_outstanding
                else:
                    print(f"No shares outstanding data available for {symbol}.")
                    continue
                price_column = 'MarketCap'
            else:
                df['date'] = df.index
                price_column = 'Close (USD)'

            # Calculate investment regardless of cached_data
            if is_regular:
                df['investment'] = (((df[price_column] / df[price_column].iloc[-1]) + 0) * weight) - weight
            else:
                df['investment'] = (((df[price_column].iloc[-1] / df[price_column]) + 0) * weight) - weight

            if symbol not in cached_data:
                cached_data[symbol] = {'df': df, 'is_stock': is_stock, 'weight': weight}

            df = df[['date', 'investment']].set_index('date')

            # Store dataframe by symbol and type
            if symbol not in symbol_dataframes:
                symbol_dataframes[symbol] = {'regular': None, 'inverse': None}

            if is_regular:
                symbol_dataframes[symbol]['regular'] = df
            else:
                symbol_dataframes[symbol]['inverse'] = df

        # Combine dataframes
        for symbol, dfs in symbol_dataframes.items():
            regular_df = dfs['regular']
            inverse_df = dfs['inverse']

            if regular_df is not None and inverse_df is not None:
                # Combine regular and inverse, subtracting inverse from regular
                combined_symbol_df = regular_df.join(inverse_df, how="outer", lsuffix="_regular", rsuffix="_inverse")
                combined_symbol_df.fillna(0, inplace=True)
                combined_symbol_df["investment"] = combined_symbol_df["investment_regular"] - combined_symbol_df["investment_inverse"]
                combined_symbol_df = combined_symbol_df[["investment"]]
            elif regular_df is not None:
                combined_symbol_df = regular_df
            elif inverse_df is not None:
                combined_symbol_df = inverse_df
            else:
                continue

            if combined_df is None:
                combined_df = combined_symbol_df
            else:
                combined_df = combined_df.join(combined_symbol_df, how="outer", lsuffix="_left", rsuffix="_right")
                combined_df.fillna(0, inplace=True)
                combined_df["investment"] = combined_df["investment_left"] + combined_df["investment_right"]
                combined_df = combined_df[["investment"]]

        if combined_df is None:
            return "No valid data could be processed."

        combined_df['total_investment'] = total_weight + combined_df['investment']
        title_parts = [f"{symbol} ({weight})" for symbol, weight, _, _ in all_assets] # Added _
        title = "Total Investment: " + ", ".join(title_parts)

        fig_linear = px.line(combined_df.reset_index(), x='date', y='total_investment', title=f'Total Investment (Linear Scale) - {title}')
        fig_log = px.line(combined_df.reset_index(), x='date', y='total_investment', title=f'Total Investment (Logarithmic Scale) - {title}')
        fig_log.update_layout(yaxis_type="log")

        plots = {'linear': fig_linear.to_html(full_html=False), 'logarithmic': fig_log.to_html(full_html=False)}

        # Convert and store in session
        cached_data_for_js = {}
        for symbol, data in cached_data.items():
            cached_data_for_js[symbol] = {
                'investment': data['df']['investment'].tolist(),
                'dates': data['df'].index.strftime('%Y-%m-%d').tolist(),
                'weight': data['weight']
            }

        # Generate unique cache ID
        cache_id = str(uuid.uuid4())

        # Store processed data in cache
        DATA_CACHE[cache_id] = {
            'cached_data': cached_data_for_js,
            'timestamp': time.time(),
            'assets': all_assets
        }

        session.clear()  # Clear existing session data
        session.modified = True

        # Store only cache ID in session
        session['current_cache_id'] = cache_id

        return render_template('plot.html',
                            plots=plots,
                            all_assets=all_assets,  # Ensure this is passed
                            cache_id=cache_id)

    return render_template('_plot_container.html', plots=plots)
      

@main.route('/update_smooth', methods=['POST'])
def update_smooth():
    try:
        # Grab the data from the request
        data = request.get_json()
        smooth_value = data.get('smoothValue', 1)
        smooth_value = int(smooth_value)  # Convert to integer

        downsampled, total_investment = construct_smooth(data, smooth_value)
        downsampled = downsampled + total_investment

        return jsonify({
            'success': True,
            'plotData': {
                'x': downsampled.index.tolist(),
                'y': downsampled.values.tolist(),
            },
            'totalInvestment': total_investment
        })
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        traceback.print_exc()  # Add this for detailed error logging
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

def compress_model(model):
    dumped = pickle.dumps(model)
    return zlib.compress(dumped, level=9)


def simple_fit_predict(x, y, degree=1, show_plot=True):
    """
    Simplified modeling pipeline with debug outputs
    Returns model, predictions, and debug info
    """
    # Convert to numpy arrays
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    
    # Log transform target (asset prices)
    y_log = np.log(y)
    print(f"\n=== Data Summary ===")
    print(f"X range: {x.min():.2f} to {x.max():.2f}")
    print(f"Y original range: {y.min():.2f} to {y.max():.2f}")
    print(f"Y log range: {y_log.min():.4f} to {y_log.max():.4f}")
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(x)
    print(f"\n=== Polynomial Features ===")
    print(f"Degree: {degree}")
    print(f"First sample features: {X_poly[0]}")

    val = [x for x in np.arange(0.0, 1.01, 0.01)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # # Fit model
    # for i in val:
    #     # model = Ridge(alpha=i, fit_intercept=False)  # Bias already in polynomial
    #     # model = Lasso(alpha=i, fit_intercept=False)

    #     model = SVR(kernel='rbf', C=100, gamma=i)  # Adjust C and gamma as needed

    #     model.fit(X_poly, y_log)

        
    #     # Get predictions
    #     y_pred_log = model.predict(X_poly)
    #     y_pred = np.exp(y_pred_log)
    #     curr_mse = mean_squared_error(y, y_pred)

    #     if (i == 0):
    #         min_mse = curr_mse
    #         min_alpha = i
    #     elif (curr_mse < min_mse):
    #         min_mse = curr_mse
    #         min_alpha = i

    # Generate example data (X_poly) for demonstration purposes
    n_samples = len(X_scaled) # Number of samples (change this to match your data)

    # Define the coefficient for the parabola
    a = 0.01

    # Generate the x values (0 to n_samples - 1)
    x_values = np.arange(n_samples)

    # Calculate the midpoint
    midpoint = n_samples // 2

    # Generate the parabolic weights
    weights = -a * (x_values - midpoint) ** 2 + midpoint

    # Ensure the weights are positive and normalize them
    weights = np.maximum(weights, 0)
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1

    # Print the weights
    print(weights)

    model = SVR(kernel='rbf', C=1000, gamma=1)  # Bias already in polynomial
    model.fit(X_poly, y_log)

        
    # Get predictions
    y_pred_log = model.predict(X_poly)
    y_pred = np.exp(y_pred_log)
    # print(f"\n=== Model Summary ===")
    # print(f"Lowest MSE: {min_mse} at alpha {min_alpha}")
    # print(f"\n=== END OF MINIMUM Model Summary ===")

    # print(f"\n=== Model Summary ===")
    # print(f"Coefficients: {model.coef_}")
    # print(f"Intercept: {model.intercept_}")
    # print(f"MSE: {mean_squared_error(y, y_pred):.4f}")

    # Get size metrics
    compressed_size = len(compress_model(model))  # Typically 1-10KB
    print(f"Model size: {compressed_size/1024:.1f}KB")

    # Debug predictions
    print("\n=== Prediction Samples ===")
    print("Index | Actual | Predicted | Error")
    for i in range(5):
        print(f"{i:5} | {y[i]:6.2f} | {y_pred[i]:8.2f} | {y[i]-y_pred[i]:6.2f}")
    
    # Visual check
    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(y, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f"Degree {degree} Polynomial Fit")
        plt.legend()
        plt.show()
    
    # return model, y_pred, {
    #     'coefficients': model.coef_,
    #     'mse': mean_squared_error(y, y_pred),
    #     'poly_features': X_poly
    # }

    return y_pred





## MAIN FUNCTION TO RUN THE CONSTRUCT FUNCTION CODE

def compute_scale_invariant(poly, x_range=(0, 1), n_points=100, eps=1e-8):
    """Compute scale-invariant signature for a polynomial"""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = poly(x)
    dy = poly.deriv(1)(x)
    ddy = poly.deriv(2)(x)
    
    # Handle division stability
    denominator = ddy * y
    denominator += eps * np.sign(denominator) + eps
    return (dy ** 2) / denominator

def find_best_match(snippet_poly, target_poly, target_domain, min_length=3, resolution=100):
    """Find best matching interval in target polynomial"""
    # Compute signatures
    snippet_sig = compute_scale_invariant(snippet_poly)
    target_x = np.linspace(target_domain[0], target_domain[1], resolution)
    target_sig = compute_scale_invariant(target_poly, target_domain, resolution)
    
    best_score = -np.inf
    best_match = (0, 0)
    
    # Generate candidate windows using FFT acceleration
    for length in range(min_length, resolution//2):
        # Create normalized window template
        window_x = np.linspace(0, 1, length)
        interp_fn = interp1d(np.linspace(0, 1, len(snippet_sig)), snippet_sig, 
                            kind='cubic', fill_value='extrapolate')
        template = interp_fn(window_x)
        
        # Normalize template
        template = (template - template.mean()) / (template.std() + 1e-8)
        
        # Compute cross-correlation using FFT
        target_norm = (target_sig - np.mean(target_sig)) / (np.std(target_sig) + 1e-8)
        corr = fftconvolve(target_norm, template[::-1], mode='valid')
        
        # Find peak correlation
        max_idx = np.argmax(corr)
        max_score = corr[max_idx]
        
        if max_score > best_score:
            best_score = max_score
            start = int(max_idx / resolution * len(target_x))
            end = int((max_idx + length) / resolution * len(target_x))
            best_match = (target_x[start], target_x[end])
    
    return best_match, best_score

## MAIN FUNCTION TO RUN THE CONSTRUCT FUNCTION CODE







@main.route('/construct_function', methods=['POST'])
def construct_function():
    try:
        # Grab the data from the request
        data = request.get_json()
        smooth_value = data.get('smoothValue', 1)
        smooth_value = int(smooth_value)  # Convert to integer
        lowest_smooth = smooth_value
       
        degree_value = data.get('degreeValue', 1)
        degree_value = int(degree_value)

        # if (smooth_value == 50):
        #     lowest_mse_smooth, lowest_smooth = compute_lowest_mse_smooth(data, 26)
          
        downsampled, total_investment, investments_str = construct_smooth(data, smooth_value)
        downsampled = downsampled + total_investment     

        downsampled_log = np.log(downsampled)


        # if (data.get('saveSmall')):
        #     downsampled_log = downsampled_log[:(smooth_value*10)]

        # Convert DatetimeIndex to numeric
        x_numeric = mdates.date2num(downsampled_log.index)

        # Normalize x-values to mean=0, std=1
        x_mean = x_numeric.mean()
        x_std = x_numeric.std()
        x_normalized = (x_numeric - x_mean) / x_std  # Now in ~[-1, 1] range
        n = len(downsampled_log)


        # CALCULATE WEIGHTS SECTION
        # Calculate weights
        n = len(x_normalized)
        section1_end = int(n / 6)
        section2_end = int(n / 6) + 1

        
        n = len(x_normalized)
        threshold1 = int(n * 0.95)
        threshold2 = int(n * 0.98)

        # Assign weights
        weights = np.ones(n)
        weights[threshold1:threshold2] = 4
        weights[threshold2:] = 20  # Assign weight of 50 to the last 2%


        lowest_mse, lowest_mse_degree = find_optimal_degree(x_normalized, downsampled_log)
        degree = lowest_mse_degree


        # # ORIGINAL FUNCTION:
        coefficients = np.polyfit(x_normalized, downsampled_log.values, lowest_mse_degree, w=weights)
        f = np.poly1d(coefficients)

        # For predictions
        x_test_normalized = (mdates.date2num(x_normalized) - x_mean) / x_std

        # Generate x-values for evaluation (numeric)
        x_numeric_eval = np.linspace(x_numeric.min(), x_numeric.max(), len(downsampled_log))

        # Normalize evaluation points using SAME mean/std as training data
        x_eval_normalized = (x_numeric_eval - x_mean) / x_std
        
        # Evaluate the polynomial (on log-transformed data)
        y_values_numeric_log = f(x_eval_normalized)

        # Inverse log transform for plotting
        y_values_numeric = np.exp(y_values_numeric_log)

        y_values_derivative_log, y_derivative_reg, _ = construct_derivative_function(f, x_eval_normalized, x_std, y_values_numeric)

        # Convert numeric x-values back to datetimes
        x_datetime_eval = mdates.num2date(x_numeric_eval)

        # normalized_y_values = y_values_numeric / total_investment
    
        y_values = y_values_numeric.tolist()

      
        
        saving = data.get('saveSmall')
        retrieving = False

    
        relative_folder = 'app/save_files'
        # file_path = os.path.join(base_path, f"{investments_str}Tot:{total_investment}S:{smooth_value}LEN:{n}.txt")

        # Capture the first ticker (all letters before the first number)
        first_ticker = re.search(r'[A-Z]+', investments_str).group(0)
        all_tickers = re.findall(r'[A-Z]+', investments_str)

        if (len(all_tickers) > 1):
            relative_folder += '/bundled'

        relative_folder += '/' + investments_str[0] + '/' + investments_str[:2] + '/' + str(first_ticker)

        # Ensure the relative directory exists
        if not os.path.exists(relative_folder):
            os.makedirs(relative_folder)
            
        file_path = os.path.join(relative_folder, f"{investments_str}(S{smooth_value}).dsp")


        ''' SAVE THE FILE PATH FOR THE DUAL COMPRESSION
        '''
        #log_normalized_y_values = np.log(normalized_y_values)

        if (saving):
            save_compressed(file_path, y_downsampled_normalized_log, total_investment, smooth_value)
            file_path_poly = os.path.join(relative_folder, f"f(x):{investments_str}(S{smooth_value}).dsp")
            save_compressed_poly(file_path_poly, coefficients)
        elif (retrieving):
            ''' RETRIEVE THE FILE PATH FOR THE DUAL COMPRESSION
            '''
            #print("The last 10 values of log_normalized_y_values are:", log_normalized_y_values[-10:])
            file_path_retrieve = os.path.join(relative_folder, "TSLA20(S1).dsp")
            data_retrieved, metadata = load_compressed(file_path_retrieve)
            data_retrieved = np.exp(data_retrieved)


            y_data_retrieved = (np.array(data_retrieved) * total_investment) - total_investment
            y_data_retrieved = y_data_retrieved.tolist()
           
            print("The file_path_retrieve is:", file_path_retrieve)
            print("The first 10 values of the data retrieved are:", data_retrieved[-10:], "and the length is:", len(data_retrieved), "and the last values is:", last_log_data, "and the first values is:", first_log_data)

            # file_path_retrieve = os.path.join(relative_folder, "PLTR200,BTBT400,MCD100,MSFT200'(S50).dsp")
            # loaded_coeffs, metadata = load_compressed_poly(file_path_retrieve)
            # print_polynomials(coefficients)
            file_path_retrieve = os.path.join(relative_folder, "f(x):MU20(S1).dsp")
            loaded_coeffs, metadata = load_compressed_poly(file_path_retrieve)
            fun_loaded = np.poly1d(loaded_coeffs)

            y_values_numeric_loaded = np.exp(fun_loaded(x_eval_normalized))
            y_values_numeric_loaded = y_values_numeric_loaded.tolist()


        # Compute similar sections to functions:
        # # ORIGINAL FUNCTION:
        # x_newmeric = mdates.date2num(downsampled_log.index[-100:])
        # new_coefficients = np.polyfit(x_normalized[-100:], downsampled_log.values[-100:], lowest_mse_degree, w=weights)
        # f_new = np.poly1d(coefficients)


        # file_path_retrieve = os.path.join(relative_folder, "f(x):MU20(S1).dsp")
        # loaded_coeffs, metadata = load_compressed_poly(file_path_retrieve)
        # fun_loaded = np.poly1d(loaded_coeffs)

        # match_range, confidence = find_best_match(f_new, fun_loaded, (x_newmeric.min(), x_newmeric.max()), min_length=10)
        # print(f"Best match between {match_range[0]:.1f} and {match_range[1]:.1f}")
        # print(f"Confidence: {confidence:.3f}")

        return jsonify({
            'success': True,
            'functionData': {
                'x': x_datetime_eval,
                'y': y_values,
            },
            'derivativeData': {
                'x': x_datetime_eval,
                'y': y_values_derivative_log,
            },
            'plotData': {
                'x': downsampled.index.tolist(),
                'y': downsampled.values.tolist(),
            },
            'totalInvestment': total_investment
        })
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        traceback.print_exc()  # Add this for detailed error logging
        return jsonify({'error': f"Processing error: {str(e)}"}), 500



app.register_blueprint(main, url_prefix='/main')

if __name__ == '__main__':
    app.run(debug=True)


DATA_CACHE = {}
CACHE_TIMEOUT = 3600

def cache_cleanup():
    while True:
        time.sleep(300)
        now = time.time()
        expired = [k for k, v in DATA_CACHE.items() if now - v['timestamp'] > CACHE_TIMEOUT]
        for k in expired:
            del DATA_CACHE[k]

# Start thread only once
if not hasattr(main, 'cleanup_thread_started'):
    cleanup_thread = threading.Thread(target=cache_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    main.cleanup_thread_started = True