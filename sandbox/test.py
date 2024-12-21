from crypto_volatility_lab.data_construction import CryptoScraper
from crypto_volatility_lab.data_construction.timeSeriesCreator import TimeSeriesCreator

scraper = CryptoScraper(start_date="2023-06-01")
df = scraper.get_data_for_currency("BTC-USD")

ts_creator = TimeSeriesCreator(df, date_column_name="Date", value_column_name="Close")
log_returns = ts_creator.create_log_return_time_series()
volatility = ts_creator.create_volatility_time_series()

df["Log Returns"] = log_returns
df["Volatility"] = volatility
print(df.head())
