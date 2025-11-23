# import pandas as pd
# import os

# # Paths
# BASE_DIR = "./QEncoder_SP500_prediction/datasets/"
# stocks_path = os.path.join(BASE_DIR, "stocks.csv")
# sp_path = os.path.join(BASE_DIR, "sp.csv")
# combined_path = os.path.join(BASE_DIR, "combined_dataset.csv")

# # Load both CSVs
# stocks_df = pd.read_csv(stocks_path)
# sp_df = pd.read_csv(sp_path)

# # Remove any unwanted columns from sp_df (columns with missing data)
# sp_df = sp_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Keep only the relevant columns

# # Add 'Name' column to sp_df (set it as 'S&P500' for all rows)
# sp_df['Name'] = 'S&P500'

# # Concatenate the two datasets (train and test)
# combined_df = pd.concat([stocks_df, sp_df], ignore_index=True)

# # Optional: Drop any duplicates and sort by 'Name' and 'Date'
# combined_df = combined_df.drop_duplicates()
# combined_df = combined_df.sort_values(by=['Name', 'Date']).reset_index(drop=True)

# # Save the merged dataset to a new CSV file
# combined_df.to_csv(combined_path, index=False)

# print(f"Combined dataset saved to: {combined_path}")

# import yfinance as yf
# import pandas as pd

# # Step 1: Download data for WTI (W&T Offshore, Inc.)
# ticker = 'WTI'
# data = yf.download(ticker, start='2005-01-28', end='2025-05-13', progress=False)

# # Step 2: Reset index to move Date from index to column
# data = data.reset_index()

# # Step 3: Keep only the required columns and reorder them
# data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# # Step 4: Drop rows with any missing values (NaNs)
# data = data.dropna()

# # Step 5 (Optional): Sort by date (ascending)
# data = data.sort_values('Date')

# # Step 6: Save to CSV
# data.to_csv("WTI_Offshore_Cleaned_Data.csv", index=False)

# # Step 7: Display a preview
# print(data.head())


# import yfinance as yf
# import pandas as pd

# # Step 1: Download NIFTY 50 index data (^NSEI)
# ticker = "^NSEI"
# data = yf.download(ticker, start="2007-09-17", end="2025-05-13", progress=False)

# # Step 2: Reset index to get 'Date' as a column
# data = data.reset_index()

# # Step 3: Keep only required columns and reorder them
# data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# # Step 4: Drop rows with any missing values (especially Volume, which can be NaN for indexes)
# data = data.dropna()

# # Step 5: Sort by date (ascending)
# data = data.sort_values('Date')

# # Step 6: Save cleaned dataset to CSV
# data.to_csv("NIFTY50_Cleaned_Data.csv", index=False)

# # Step 7: Show preview
# print(data.head())


import numpy as np
import os

BASE_DIR = './QEncoder_SP500_prediction'
datafiles_dir = os.path.join(BASE_DIR,'processed_data/')
x = np.load(os.path.join(datafiles_dir,'X_wti.npy'))
print(x[0:])