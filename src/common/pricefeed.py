import yfinance as yf
from datetime import datetime, time
import pytz


class CommodityPriceFetcher:
    def __init__(self):
        self.gold_ticker = yf.Ticker("GC=F")  # Gold futures
        self.oil_ticker = yf.Ticker("CL=F")  # Crude oil futures
        self.nyse_tz = pytz.timezone("America/New_York")

    def is_trading_hours(self, timestamp):
        # Convert timestamp to NYSE timezone
        nyse_time = timestamp.astimezone(self.nyse_tz)

        # Define trading hours (assuming 24/5 trading for futures)
        trading_start = time(18, 0)  # 6 PM previous day
        trading_end = time(17, 0)  # 5 PM current day

        # Check if it's a weekday and within trading hours
        is_weekday = nyse_time.weekday() < 5
        if trading_start <= trading_end:
            is_trading_time = trading_start <= nyse_time.time() <= trading_end
        else:  # Handles overnight trading
            is_trading_time = (
                nyse_time.time() >= trading_start or nyse_time.time() <= trading_end
            )

        return is_weekday and is_trading_time

    def get_price(self, commodity, timestamp):
        if not self.is_trading_hours(timestamp):
            return None

        ticker = self.gold_ticker if commodity.lower() == "gold" else self.oil_ticker

        # Fetch historical data
        data = ticker.history(start=timestamp, end=timestamp, interval="1m")

        if data.empty:
            return None

        return data["Close"].iloc[0]


# Usage example
if __name__ == "__main__":
    fetcher = CommodityPriceFetcher()

    # Example timestamps
    timestamps = [
        datetime(2023, 6, 15, 10, 30, tzinfo=pytz.UTC),  # During trading hours
        datetime(2023, 6, 15, 22, 30, tzinfo=pytz.UTC),  # Outside trading hours
        datetime(2023, 6, 17, 10, 30, tzinfo=pytz.UTC),  # Saturday
    ]

    for timestamp in timestamps:
        gold_price = fetcher.get_price("gold", timestamp)
        oil_price = fetcher.get_price("oil", timestamp)

        print(f"Timestamp: {timestamp}")
        print(f"Gold price: {gold_price}")
        print(f"Oil price: {oil_price}")
        print()
