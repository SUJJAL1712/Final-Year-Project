"""
Stock market data collection for US, India, and China markets.

Uses yfinance for historical price data across all three markets.
Handles index data, sector ETFs, and individual stocks.
"""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from src.utils.config import get_config, DATA_DIR, ANALYSIS_START, ANALYSIS_END


class StockDataFetcher:
    """Fetches and manages historical stock price data."""

    def __init__(self):
        self.config = get_config()
        self.raw_dir = DATA_DIR / "raw" / "stocks"
        self.processed_dir = DATA_DIR / "processed" / "stocks"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.request_pause_seconds = 3.0
        self.max_retries = 5

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        """Convert a ticker into a filesystem-safe filename stem."""
        return symbol.replace("^", "IDX_").replace(".", "_")

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
        start = start or ANALYSIS_START
        end = end or ANALYSIS_END

        logger.info(f"Fetching {symbol} from {start} to {end}")

        cached = self.load_cached(symbol)
        if cached is not None and not cached.empty:
            try:
                cache_start = cached.index.min().tz_localize(None) if hasattr(cached.index.min(), 'tz') and cached.index.min().tz else cached.index.min()
                cache_end = cached.index.max().tz_localize(None) if hasattr(cached.index.max(), 'tz') and cached.index.max().tz else cached.index.max()
                req_start = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tz else pd.Timestamp(start)
                req_end = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tz else pd.Timestamp(end)
                # Accept cache if within 5 days of requested start/end (weekends/holidays)
                if cache_start <= req_start + pd.Timedelta(days=5) and cache_end >= req_end - pd.Timedelta(days=5):
                    logger.info(f"Using cached stock data for {symbol}")
                    return cached
            except Exception as e:
                logger.warning(f"Cache check failed for {symbol}: {e}")

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Pause before each request to avoid rate limiting
                if attempt > 1:
                    wait_seconds = min(15 * (attempt - 1), 60)
                    logger.info(f"Waiting {wait_seconds}s before retry {attempt} for {symbol}")
                    time.sleep(wait_seconds)

                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    break

                # Standardize column names
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                df.index.name = "date"
                # Strip timezone to avoid tz mismatch issues downstream
                if hasattr(df.index, "tz") and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Save raw data
                df.to_csv(self.raw_dir / f"{self._safe_symbol(symbol)}.csv")

                logger.info(f"Fetched {len(df)} rows for {symbol}")
                time.sleep(self.request_pause_seconds)
                return df

            except Exception as e:
                last_error = e
                message = str(e).lower()
                is_rate_limited = "too many requests" in message or "rate limited" in message

                if attempt < self.max_retries and is_rate_limited:
                    logger.warning(
                        f"Rate limited fetching {symbol} "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    continue

                logger.error(f"Failed to fetch {symbol}: {e}")
                break

        if cached is not None and not cached.empty:
            logger.warning(f"Using stale cached stock data for {symbol} after fetch failure")
            return cached

        if last_error:
            logger.error(f"No cached fallback for {symbol} after fetch failure")
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
        path = self.raw_dir / f"{self._safe_symbol(symbol)}.csv"

        if path.exists():
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            # Ensure tz-naive index (handles tz-aware, mixed-tz, and object dtype)
            try:
                if hasattr(df.index, "tz") and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                elif df.index.dtype == object or str(df.index.dtype).startswith("datetime64[ns,"):
                    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            except Exception:
                df.index = pd.to_datetime(df.index.astype(str).str.replace(r"[+-]\d{2}:\d{2}$", "", regex=True))
            # Normalize to midnight (drop time component from date index)
            df.index = df.index.normalize()
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
