import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
from hmmlearn.hmm import GaussianHMM
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Set date range
start_date = '2005-01-01'
end_date = '2024-12-31'

# Download bond ETF (LQD), S&P 500 (SPY), and 10Y yield proxy (^TNX)
tickers = ['LQD', 'SPY', '^TNX']
data_yf = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

# Flatten column MultiIndex
data_yf.columns = [' '.join(col).strip() for col in data_yf.columns.values]

# Download 10-Year Treasury Constant Maturity Rate from FRED
try:
    dgs10 = pdr.DataReader('DGS10', 'fred', start_date, end_date)
    dgs10.rename(columns={'DGS10': 'DGS10 Close'}, inplace=True)
    dgs10['DGS10 Close'].fillna(method='ffill', inplace=True)
    print("Successfully downloaded DGS10 from FRED.")
except Exception as e:
    print(f"Failed to download DGS10 from FRED: {e}")
    dgs10 = pd.DataFrame()

# Merge all data into one DataFrame
combined = pd.concat([data_yf, dgs10], axis=1)
combined.dropna(inplace=True)

# DEBUG: Print after flattening and merging
print("\nData columns after flattening:")
print(combined.columns)

print("\nSample rows from merged dataset:")
print(combined.head())

# Calculate daily returns and use as input for HMM
combined['LQD Returns'] = combined['LQD Close'].pct_change()
combined['SPY Returns'] = combined['SPY Close'].pct_change()
combined['10Y Yield Change'] = combined['DGS10 Close'].pct_change()

# Drop NaNs created by percent change
combined.dropna(inplace=True)

# Define feature set (can try different combinations)
X = combined[['LQD Returns', 'SPY Returns', '10Y Yield Change']].values

# DEBUG: Check X shape
print(f"\nShape of input X for HMM: {X.shape}")
print("First 5 rows of X:\n", X[:5])

# Train HMM
model = GaussianHMM(n_components=2, covariance_type='full', n_iter=1000, random_state=42)
model.fit(X)

# Predict hidden states
hidden_states = model.predict(X)
combined['Regime'] = hidden_states

# Visualize regimes
plt.figure(figsize=(14, 6))
for i in range(model.n_components):
    state_data = combined[combined['Regime'] == i]
    plt.plot(state_data.index, state_data['LQD Close'], '.', label=f'Regime {i}')
plt.legend()
plt.title("Regime Classification based on HMM")
plt.xlabel("Date")
plt.ylabel("LQD Price")
plt.show()