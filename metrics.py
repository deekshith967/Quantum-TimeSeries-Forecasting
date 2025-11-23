# metrics.py (No changes needed)

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Add a small epsilon to avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return np.min(drawdown) if len(drawdown) > 0 else 0

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    std_dev = np.std(excess_returns)
    return np.mean(excess_returns) / (std_dev + 1e-8)

def sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    negative_returns = returns[returns < risk_free_rate]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-8
    return np.mean(excess_returns) / (downside_std + 1e-8)

def metrics(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()

    # Standard metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    # Financial metrics (calculated on predicted values)
    returns = np.diff(y_pred) / (y_pred[:-1] + 1e-8)
    ret_pct = np.sum(returns) * 100
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    max_dd = max_drawdown(returns)

    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape_val:.2f}%")
    print("-" * 20)
    print(f"Total Return (based on predictions): {ret_pct:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Sortino Ratio: {sortino:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")