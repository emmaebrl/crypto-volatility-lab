import datetime
import os
import time
import requests
from bs4 import BeautifulSoup
from typing import Optional, List
import pandas as pd

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
}


class CryptoScraper:
    """A web scraper for fetching historical cryptocurrency data from Yahoo Finance."""

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: str = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime(
            "%Y-%m-%d"
        ),
        headers=DEFAULT_HEADERS,
    ):
        self.start_date = self._parse_date(start_date) if start_date else None
        self.end_date = self._parse_date(end_date)
        self.headers = headers

    def _parse_date(self, date_str: str) -> List[int]:
        """Parses a date string into a list of integers [year, month, day]."""
        try:
            return list(map(int, date_str.split("-")))
        except ValueError:
            raise ValueError(
                f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD."
            )

    def _get_unix_time(self, date_parts: List[int]) -> int:
        """Converts a date list [year, month, day] to a Unix timestamp."""
        year, month, day = date_parts
        date = datetime.datetime(year, month, day)
        return int(time.mktime(date.timetuple()))

    def _build_url(self, currency: str) -> str:
        """Builds the URL for fetching historical data for a given currency."""
        start_unix = self._get_unix_time(self.start_date) if self.start_date else 0
        end_unix = self._get_unix_time(self.end_date)
        return (
            f"https://finance.yahoo.com/quote/{currency}/history/?"
            f"period1={start_unix}&period2={end_unix}"
        )

    def get_data_for_currency(self, currency) -> pd.DataFrame:
        url = self._build_url(currency)
        response = None
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if not table:
                print("No table found in the HTML content.")
                return pd.DataFrame()

            data_headers = [th.text for th in table.find("thead").find_all("th")]  # type: ignore
            data_headers = [header.split()[0].strip() for header in data_headers]
            rows = table.find("tbody").find_all("tr")  # type: ignore
            data = []
            for row in rows:
                data.append([td.text for td in row.find_all("td")])

            return pd.DataFrame(data, columns=data_headers)
        except requests.exceptions.RequestException as e:
            print(
                f"Error fetching data from {url}: {e} (Status code: {response.status_code if response else 'N/A'})"
            )
            return pd.DataFrame()

    def get_data(
        self,
        currencies: List[str],
        output_folder: Optional[str] = "data",
    ):
        output_folder = output_folder or "data"
        os.makedirs(output_folder, exist_ok=True)
        for currency in currencies:
            df = self.get_data_for_currency(currency)
            if isinstance(df, pd.DataFrame) and not df.empty:
                output_path = os.path.join(output_folder, f"{currency}.csv")
                df.to_csv(output_path, index=False)
                print(f"Data for {currency} saved to {output_path}.")
            else:
                print(f"No data found for {currency}.")
