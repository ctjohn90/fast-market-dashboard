"""Yahoo Finance data fetcher for market indicators."""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.config import Settings


logger = logging.getLogger(__name__)


# Yahoo Finance tickers to fetch
YAHOO_TICKERS: dict[str, str] = {
    # VIX Term Structure & Volatility
    "^VIX": "VIX Spot",
    "^VIX3M": "VIX 3-Month",
    "^VVIX": "VVIX (Volatility of VIX)",
    # Equity indices
    "SPY": "S&P 500 ETF",
    "IWM": "Russell 2000 ETF",
    "QQQ": "Nasdaq 100 ETF",
    # Credit ETFs (direct bond market stress)
    "HYG": "High Yield Bond ETF",
    "JNK": "SPDR High Yield Bond ETF",
    "LQD": "Investment Grade Bond ETF",
    # Sector ETFs (for correlation)
    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLU": "Utilities",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    # Safe haven & bonds
    "GLD": "Gold ETF",
    "TLT": "20+ Year Treasury ETF",
    "UUP": "US Dollar ETF",
    # International (risk appetite)
    "EEM": "Emerging Markets ETF",
    "FXI": "China Large Cap ETF",
    "EFA": "Developed Markets ETF",
    # Currency (carry trade)
    "FXY": "Japanese Yen ETF",
    # Commodities
    "USO": "US Oil ETF",
    # Volatility
    "^SKEW": "SKEW Index",
}


class YahooFetcher:
    """Fetches market data from Yahoo Finance using batch downloads."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.cache = DataCache(self.settings.db_path)

    def _normalize_ticker(self, ticker: str) -> str:
        """Convert Yahoo ticker to cache-friendly ID."""
        return "YF_" + ticker.replace("^", "").replace("-", "_")

    def fetch_all(self, force_full: bool = False, period: str = "5y") -> dict[str, pd.DataFrame]:
        """
        Fetch all tickers in a single batch request.
        
        Args:
            force_full: If True, refetch all history
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            
        Returns:
            Dict mapping ticker to DataFrame
        """
        tickers = list(YAHOO_TICKERS.keys())
        logger.info(f"Batch downloading {len(tickers)} tickers...")

        try:
            # Single batch request - much less likely to be rate limited
            data = yf.download(
                tickers,
                period=period,
                progress=True,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )

            if data.empty:
                logger.error("No data returned from Yahoo Finance")
                return {}

            results = {}
            fetched_at = datetime.now()

            for ticker in tickers:
                series_id = self._normalize_ticker(ticker)
                
                try:
                    # Handle both single and multi-ticker response formats
                    if len(tickers) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]

                    if ticker_data.empty:
                        logger.warning(f"  {ticker}: No data")
                        continue

                    # Extract Close prices
                    close = ticker_data["Close"].dropna()
                    if close.empty:
                        continue

                    df = pd.DataFrame({"value": close})
                    df.index = pd.to_datetime(df.index)

                    # Store in cache
                    rows = self.cache.store_observations(series_id, df, fetched_at)
                    logger.info(f"  {ticker} -> {series_id}: {rows} observations")

                    results[ticker] = self.cache.get_series(series_id)

                except Exception as e:
                    logger.warning(f"  {ticker}: Error processing - {e}")

            return results

        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            return {}

    def get_status(self) -> dict[str, dict]:
        """Get cache status for Yahoo Finance series."""
        cache_status = self.cache.get_cache_status()

        status = {}
        for ticker, name in YAHOO_TICKERS.items():
            series_id = self._normalize_ticker(ticker)
            if series_id in cache_status:
                info = cache_status[series_id]
                status[ticker] = {
                    "name": name,
                    "series_id": series_id,
                    "observation_count": info["observation_count"],
                    "first_date": info["first_date"],
                    "last_date": info["last_date"],
                }
            else:
                status[ticker] = {
                    "name": name,
                    "series_id": series_id,
                    "observation_count": 0,
                    "first_date": None,
                    "last_date": None,
                }

        return status

    def calculate_derived_indicators(self) -> dict[str, pd.DataFrame]:
        """
        Calculate derived indicators from cached Yahoo data.
        
        Returns:
            Dict of derived indicator DataFrames
        """
        derived = {}
        fetched_at = datetime.now()

        # VIX Term Structure (VIX / VIX3M)
        # < 1 = backwardation = PANIC
        # > 1 = contango = normal
        vix = self.cache.get_series("YF_VIX")
        vix3m = self.cache.get_series("YF_VIX3M")
        if not vix.empty and not vix3m.empty:
            aligned = pd.concat([vix["value"], vix3m["value"]], axis=1, keys=["vix", "vix3m"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ts = aligned["vix"] / aligned["vix3m"]
                df = pd.DataFrame({"value": ts})
                self.cache.store_observations("VIX_TERM_STRUCTURE", df, fetched_at)
                derived["VIX_TERM_STRUCTURE"] = df
                logger.info(f"  VIX Term Structure: {len(ts)} obs, current={ts.iloc[-1]:.3f}")

        # Risk Appetite (IWM / SPY)
        # High = risk-on, Low = risk-off
        iwm = self.cache.get_series("YF_IWM")
        spy = self.cache.get_series("YF_SPY")
        if not iwm.empty and not spy.empty:
            aligned = pd.concat([iwm["value"], spy["value"]], axis=1, keys=["iwm", "spy"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ratio = aligned["iwm"] / aligned["spy"]
                df = pd.DataFrame({"value": ratio})
                self.cache.store_observations("RISK_APPETITE", df, fetched_at)
                derived["RISK_APPETITE"] = df
                logger.info(f"  Risk Appetite (IWM/SPY): {len(ratio)} obs")

        # Safe Haven Demand (GLD / SPY)
        # High = fear, Low = greed
        gld = self.cache.get_series("YF_GLD")
        if not gld.empty and not spy.empty:
            aligned = pd.concat([gld["value"], spy["value"]], axis=1, keys=["gld", "spy"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ratio = aligned["gld"] / aligned["spy"]
                df = pd.DataFrame({"value": ratio})
                self.cache.store_observations("SAFE_HAVEN", df, fetched_at)
                derived["SAFE_HAVEN"] = df
                logger.info(f"  Safe Haven (GLD/SPY): {len(ratio)} obs")

        # Defensive Rotation (XLU / XLY)
        # High = defensive positioning, Low = aggressive
        xlu = self.cache.get_series("YF_XLU")
        xly = self.cache.get_series("YF_XLY")
        if not xlu.empty and not xly.empty:
            aligned = pd.concat([xlu["value"], xly["value"]], axis=1, keys=["xlu", "xly"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ratio = aligned["xlu"] / aligned["xly"]
                df = pd.DataFrame({"value": ratio})
                self.cache.store_observations("DEFENSIVE_ROTATION", df, fetched_at)
                derived["DEFENSIVE_ROTATION"] = df
                logger.info(f"  Defensive Rotation (XLU/XLY): {len(ratio)} obs")

        # Sector Correlation (average pairwise correlation of sector ETFs)
        # High correlation = stress (everything moving together)
        sectors = ["XLF", "XLK", "XLE", "XLV", "XLU", "XLY", "XLP"]
        sector_data = {}
        for sector in sectors:
            df = self.cache.get_series(f"YF_{sector}")
            if not df.empty:
                sector_data[sector] = df["value"].pct_change()

        if len(sector_data) >= 4:
            sector_df = pd.DataFrame(sector_data).dropna()
            if len(sector_df) > 60:
                # 60-day rolling average pairwise correlation
                avg_corr = []
                dates = []

                for i in range(60, len(sector_df)):
                    window = sector_df.iloc[i - 60 : i]
                    corr_matrix = window.corr().values
                    # Upper triangle only (exclude diagonal)
                    upper = corr_matrix[np.triu_indices(len(corr_matrix), k=1)]
                    avg_corr.append(upper.mean())
                    dates.append(sector_df.index[i])

                if avg_corr:
                    df = pd.DataFrame({"value": avg_corr}, index=dates)
                    self.cache.store_observations("SECTOR_CORRELATION", df, fetched_at)
                    derived["SECTOR_CORRELATION"] = df
                    logger.info(f"  Sector Correlation: {len(avg_corr)} obs, current={avg_corr[-1]:.3f}")

        # HY Bond Stress (HYG / LQD ratio)
        # Low ratio = flight to quality (stress)
        hyg = self.cache.get_series("YF_HYG")
        lqd = self.cache.get_series("YF_LQD")
        if not hyg.empty and not lqd.empty:
            aligned = pd.concat([hyg["value"], lqd["value"]], axis=1, keys=["hyg", "lqd"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ratio = aligned["hyg"] / aligned["lqd"]
                df = pd.DataFrame({"value": ratio})
                self.cache.store_observations("HY_BOND_STRESS", df, fetched_at)
                derived["HY_BOND_STRESS"] = df
                logger.info(f"  HY Bond Stress (HYG/LQD): {len(ratio)} obs")

        # EM Risk (EEM / SPY ratio)
        # Low ratio = EM underperformance = risk-off
        eem = self.cache.get_series("YF_EEM")
        if not eem.empty and not spy.empty:
            aligned = pd.concat([eem["value"], spy["value"]], axis=1, keys=["eem", "spy"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ratio = aligned["eem"] / aligned["spy"]
                df = pd.DataFrame({"value": ratio})
                self.cache.store_observations("EM_RISK", df, fetched_at)
                derived["EM_RISK"] = df
                logger.info(f"  EM Risk (EEM/SPY): {len(ratio)} obs")

        # Flight to Quality (TLT / HYG ratio)
        # High ratio = flight to treasuries = stress
        tlt = self.cache.get_series("YF_TLT")
        if not tlt.empty and not hyg.empty:
            aligned = pd.concat([tlt["value"], hyg["value"]], axis=1, keys=["tlt", "hyg"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ratio = aligned["tlt"] / aligned["hyg"]
                df = pd.DataFrame({"value": ratio})
                self.cache.store_observations("FLIGHT_TO_QUALITY", df, fetched_at)
                derived["FLIGHT_TO_QUALITY"] = df
                logger.info(f"  Flight to Quality (TLT/HYG): {len(ratio)} obs")

        # Tech vs Staples (XLK / XLP ratio)
        # Low ratio = defensive rotation
        xlk = self.cache.get_series("YF_XLK")
        xlp = self.cache.get_series("YF_XLP")
        if not xlk.empty and not xlp.empty:
            aligned = pd.concat([xlk["value"], xlp["value"]], axis=1, keys=["xlk", "xlp"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ratio = aligned["xlk"] / aligned["xlp"]
                df = pd.DataFrame({"value": ratio})
                self.cache.store_observations("TECH_VS_STAPLES", df, fetched_at)
                derived["TECH_VS_STAPLES"] = df
                logger.info(f"  Tech vs Staples (XLK/XLP): {len(ratio)} obs")

        # ========== NEW FORECAST FEATURES ==========

        # VVIX (Volatility of VIX) - leading indicator of big moves
        vvix = self.cache.get_series("YF_VVIX")
        if not vvix.empty:
            df = pd.DataFrame({"value": vvix["value"]})
            self.cache.store_observations("VVIX_LEVEL", df, fetched_at)
            derived["VVIX_LEVEL"] = df
            logger.info(f"  VVIX Level: {len(df)} obs, current={df['value'].iloc[-1]:.1f}")

        # TLT Realized Volatility (MOVE Index proxy)
        # 20-day annualized volatility of long treasury ETF
        tlt = self.cache.get_series("YF_TLT")
        if not tlt.empty and len(tlt) > 20:
            tlt_returns = tlt["value"].pct_change()
            tlt_vol = tlt_returns.rolling(20).std() * np.sqrt(252) * 100
            tlt_vol = tlt_vol.dropna()
            if not tlt_vol.empty:
                df = pd.DataFrame({"value": tlt_vol})
                self.cache.store_observations("BOND_VOLATILITY", df, fetched_at)
                derived["BOND_VOLATILITY"] = df
                logger.info(f"  Bond Volatility (TLT): {len(df)} obs, current={tlt_vol.iloc[-1]:.1f}%")

        # SPY Relative Volume (liquidity/conviction signal)
        # Current volume vs 20-day average
        spy = self.cache.get_series("YF_SPY")
        if not spy.empty:
            try:
                spy_data = yf.download("SPY", period="2y", progress=False)
                if not spy_data.empty and "Volume" in spy_data.columns:
                    vol = spy_data["Volume"]
                    vol_avg = vol.rolling(20).mean()
                    rel_vol = vol / vol_avg
                    rel_vol = rel_vol.dropna()
                    if not rel_vol.empty:
                        df = pd.DataFrame({"value": rel_vol})
                        self.cache.store_observations("SPY_REL_VOLUME", df, fetched_at)
                        derived["SPY_REL_VOLUME"] = df
                        logger.info(f"  SPY Relative Volume: {len(df)} obs, current={rel_vol.iloc[-1]:.2f}x")
            except Exception as e:
                logger.warning(f"  SPY Volume fetch failed: {e}")

        # Credit Risk Appetite (HYG/LQD momentum)
        # 5-day change in junk vs investment grade ratio
        if not hyg.empty and not lqd.empty:
            aligned = pd.concat([hyg["value"], lqd["value"]], axis=1, keys=["hyg", "lqd"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ratio = aligned["hyg"] / aligned["lqd"]
                ratio_momentum = ratio.pct_change(5) * 100  # 5-day % change
                ratio_momentum = ratio_momentum.dropna()
                if not ratio_momentum.empty:
                    df = pd.DataFrame({"value": ratio_momentum})
                    self.cache.store_observations("CREDIT_APPETITE_MOM", df, fetched_at)
                    derived["CREDIT_APPETITE_MOM"] = df
                    logger.info(f"  Credit Appetite Momentum: {len(df)} obs")

        # Small Cap Risk Appetite Momentum (IWM/SPY momentum)
        iwm = self.cache.get_series("YF_IWM")
        if not iwm.empty and not spy.empty:
            aligned = pd.concat([iwm["value"], spy["value"]], axis=1, keys=["iwm", "spy"])
            aligned = aligned.dropna()
            if not aligned.empty:
                ratio = aligned["iwm"] / aligned["spy"]
                ratio_momentum = ratio.pct_change(5) * 100
                ratio_momentum = ratio_momentum.dropna()
                if not ratio_momentum.empty:
                    df = pd.DataFrame({"value": ratio_momentum})
                    self.cache.store_observations("SMALLCAP_APPETITE_MOM", df, fetched_at)
                    derived["SMALLCAP_APPETITE_MOM"] = df
                    logger.info(f"  Small Cap Appetite Momentum: {len(df)} obs")

        # Yen Carry Signal (FXY change)
        # Rising yen = carry trade unwinding = risk-off
        fxy = self.cache.get_series("YF_FXY")
        if not fxy.empty:
            fxy_change = fxy["value"].pct_change(5) * 100  # 5-day change
            fxy_change = fxy_change.dropna()
            if not fxy_change.empty:
                df = pd.DataFrame({"value": fxy_change})
                self.cache.store_observations("YEN_CARRY_SIGNAL", df, fetched_at)
                derived["YEN_CARRY_SIGNAL"] = df
                logger.info(f"  Yen Carry Signal: {len(df)} obs")

        # Oil Stress (USO 5-day return)
        # Large moves in oil often precede equity volatility
        uso = self.cache.get_series("YF_USO")
        if not uso.empty:
            uso_ret = uso["value"].pct_change(5) * 100
            uso_ret = uso_ret.dropna()
            if not uso_ret.empty:
                df = pd.DataFrame({"value": uso_ret})
                self.cache.store_observations("OIL_STRESS", df, fetched_at)
                derived["OIL_STRESS"] = df
                logger.info(f"  Oil Stress (USO 5d return): {len(df)} obs")

        # VIX Rate of Change (momentum)
        if not vix.empty:
            vix_roc = vix["value"].pct_change(5) * 100
            vix_roc = vix_roc.dropna()
            if not vix_roc.empty:
                df = pd.DataFrame({"value": vix_roc})
                self.cache.store_observations("VIX_MOMENTUM", df, fetched_at)
                derived["VIX_MOMENTUM"] = df
                logger.info(f"  VIX Momentum (5d ROC): {len(df)} obs")

        # Spread Acceleration (2nd derivative of HY spread proxy)
        # Uses HYG inverse as spread proxy
        if not hyg.empty:
            spread_proxy = 1 / hyg["value"]  # Inverse of price = proxy for spread
            spread_change = spread_proxy.pct_change(5)
            spread_accel = spread_change.diff(5)  # 2nd derivative
            spread_accel = spread_accel.dropna()
            if not spread_accel.empty:
                df = pd.DataFrame({"value": spread_accel})
                self.cache.store_observations("SPREAD_ACCELERATION", df, fetched_at)
                derived["SPREAD_ACCELERATION"] = df
                logger.info(f"  Spread Acceleration: {len(df)} obs")

        return derived


def main() -> None:
    """CLI entry point for Yahoo Finance fetching."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance market data")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and exit",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max",
    )
    args = parser.parse_args()

    fetcher = YahooFetcher()

    if args.status:
        status = fetcher.get_status()
        print("\nYahoo Finance Cache Status:")
        print("-" * 80)
        for ticker, info in sorted(status.items()):
            count = info["observation_count"]
            last = info["last_date"] or "N/A"
            print(f"{ticker:12} | {info['name']:25} | {count:6} obs | Last: {last}")
        return

    # Batch fetch all tickers
    fetcher.fetch_all(period=args.period)

    # Calculate derived indicators
    print("\nCalculating derived indicators...")
    fetcher.calculate_derived_indicators()

    print("\nDone.")


if __name__ == "__main__":
    main()
