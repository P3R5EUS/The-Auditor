
"""
financials.py — Pull live financial metrics from Yahoo Finance via yfinance.

This complements the filing analysis with current market data.
Falls back gracefully if the ticker isn't found or network is unavailable.
"""

import yfinance as yf


def get_financial_snapshot(ticker: str) -> dict | None:
    """
    Fetch key financial metrics for a ticker.

    Returns:
        Dict with market_cap, pe_ratio, revenue, eps — all formatted strings.
        Returns None if data unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        def fmt_large(val) -> str:
            """Format large numbers as $XB or $XM."""
            if val is None:
                return "N/A"
            val = float(val)
            if val >= 1e12:
                return f"${val/1e12:.2f}T"
            elif val >= 1e9:
                return f"${val/1e9:.2f}B"
            elif val >= 1e6:
                return f"${val/1e6:.2f}M"
            return f"${val:,.0f}"

        def fmt_ratio(val, decimals=2) -> str:
            if val is None:
                return "N/A"
            return f"{float(val):.{decimals}f}x"

        def fmt_eps(val) -> str:
            if val is None:
                return "N/A"
            return f"${float(val):.2f}"

        return {
            "market_cap": fmt_large(info.get("marketCap")),
            "pe_ratio": fmt_ratio(info.get("trailingPE")),
            "revenue": fmt_large(info.get("totalRevenue")),
            "eps": fmt_eps(info.get("trailingEps")),
            "price": f"${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}",
            "52w_high": f"${info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52w_low": f"${info.get('fiftyTwoWeekLow', 'N/A')}",
        }

    except Exception as e:
        print(f"[financials] Could not fetch yfinance data for {ticker}: {e}")
        return None