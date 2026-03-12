"""
Stock market data collection for US, India, and China markets.

Uses yfinance for historical price data across all three markets.
Handles index data, sector ETFs, and individual stocks.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from src.utils.config import get_config, DATA_DIR


class StockDataFetcher:
    """Fetches and manages historical stock price data."""

    def __init__(self):
        self.config = get_config()
        self.raw_dir = DATA_DIR / "raw" / "stocks"
        self.processed_dir = DATA_DIR / "processed" / "stocks"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_symbol(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'TCS.NS', '^GSPC')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1wk', '1mo')

        Returns:
            DataFrame with OHLCV data
        """
        start = start or self.config.time_range["start"]
        end = end or self.config.time_range["end"]

        logger.info(f"Fetching {symbol} from {start} to {end}")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Standardize column names
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df.index.name = "date"

            # Save raw data
            safe_symbol = symbol.replace("^", "IDX_").replace(".", "_")
            df.to_csv(self.raw_dir / f"{safe_symbol}.csv")

            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()

    def fetch_market(self, market_id: str) -> dict[str, pd.DataFrame]:
        """
        Fetch all symbols for a given market (US, India, China).

        Returns:
            Dict mapping symbol to its DataFrame
        """
        symbols = self.config.get_all_symbols(market_id)
        logger.info(f"Fetching {len(symbols)} symbols for {market_id}")

        data = {}
        for symbol in symbols:
            df = self.fetch_symbol(symbol)
            if not df.empty:
                data[symbol] = df

        return data

    def fetch_all_markets(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Fetch data for all configured markets."""
        all_data = {}
        for market_id in ["US", "India", "China"]:
            all_data[market_id] = self.fetch_market(market_id)
        return all_data

    def get_index_prices(self) -> dict[str, pd.Series]:
        """
        Get closing prices for main indices of each market.
        Useful for quick cross-market comparisons.
        """
        indices = {
            "US_SP500": "^GSPC",
            "US_NASDAQ": "^IXIC",
            "India_NIFTY": "^NSEI",
            "India_SENSEX": "^BSESN",
            "China_SSE": "000001.SS",
            "China_HSI": "^HSI",
        }

        prices = {}
        for name, symbol in indices.items():
            df = self.fetch_symbol(symbol)
            if not df.empty and "close" in df.columns:
                prices[name] = df["close"]

        return prices

    def load_cached(self, symbol: str) -> pd.DataFrame | None:
        """Load previously fetched data from disk."""
        safe_symbol = symbol.replace("^", "IDX_").replace(".", "_")
        path = self.raw_dir / f"{safe_symbol}.csv"

        if path.exists():
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            return df
        return None

    def get_sector_data(self, market_id: str = "US") -> dict[str, pd.Series]:
        """Get closing prices for sector ETFs (US market)."""
        market = self.config.get_market(market_id)
        sector_etfs = market.get("sector_etfs", {})

        sector_prices = {}
        for sector, etf in sector_etfs.items():
            df = self.fetch_symbol(etf)
            if not df.empty and "close" in df.columns:
                sector_prices[sector] = df["close"]

        return sector_prices

    def compute_cross_market_returns(self) -> pd.DataFrame:
        """
        Compute aligned daily returns for main indices across all markets.
        Handles timezone differences by aligning on date.
        """
        index_prices = self.get_index_prices()

        returns = {}
        for name, prices in index_prices.items():
            ret = prices.pct_change().dropna()
            ret.index = ret.index.normalize()  # Remove time component
            returns[name] = ret

        return pd.DataFrame(returns).dropna()
