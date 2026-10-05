# Sample market data

`sample_market_data.csv` is **synthetic data for development and testing only**.
The symbols (ABC, XYZ, LMN) are not real CSE stocks and the prices, volumes and
values traded are randomly generated, not historical.

- 3 symbols x 25 trading days (2026-01-02 to 2026-02-05, weekdays)
- ABC: liquid, low volatility (~Rs. 100)
- XYZ: moderately liquid, medium volatility (~Rs. 45)
- LMN: illiquid, high volatility (~Rs. 175-250), includes zero-volume days

Known quirk: 5 LMN rows have zero volume but a non-zero price range, which real
data would not show. The data-source audit flags these as
`ZERO_VOLUME_WITH_PRICE_RANGE` warnings. The file is kept unchanged because the
Phase 1 tests depend on it.
