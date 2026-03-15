"""
Tests for the GDELT historical event fetcher.

Tests cover:
- URL construction for daily aggregate files
- CAMEO code classification (tariff, conflict, unrelated)
- Goldstein-to-severity mapping
- Market inference from actor countries
- Processing pipeline (raw -> classified events)
- Deduplication logic
- Date range clamping and edge cases
"""

import io
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data_collection.gdelt_historical import (
    GDELTHistoricalFetcher,
    GDELT_DAILY_BASE,
    GDELT_DAILY_START,
    GDELT_DAILY_COLUMNS,
    USE_COLS,
    USE_COL_INDICES,
    TARGET_COUNTRIES,
    COUNTRY_MAP,
    MARKET_IMPACT,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def fetcher(tmp_path):
    """Create a GDELTHistoricalFetcher with a temp cache directory."""
    return GDELTHistoricalFetcher(cache_dir=tmp_path / "gdelt_cache")


def _make_raw_row(**overrides):
    """Create a raw GDELT-style row dict with sensible defaults."""
    base = {
        "GlobalEventID": "123456",
        "Day": "20230615",
        "Actor1Name": "UNITED STATES",
        "Actor1CountryCode": "USA",
        "Actor2Name": "CHINA",
        "Actor2CountryCode": "CHN",
        "IsRootEvent": "1",
        "EventCode": "162",
        "EventBaseCode": "162",
        "EventRootCode": "16",
        "QuadClass": "4",
        "GoldsteinScale": -7.0,
        "NumMentions": 10,
        "NumSources": 5,
        "NumArticles": 3,
        "AvgTone": -3.5,
        "SOURCEURL": "http://example.com/article",
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ #
# URL construction
# ------------------------------------------------------------------ #

class TestBuildDailyURL:
    def test_basic_url(self):
        dt = datetime(2023, 6, 15)
        url = GDELTHistoricalFetcher.build_daily_url(dt)
        assert url == f"{GDELT_DAILY_BASE}20230615.export.CSV.zip"

    def test_url_with_leading_zeros(self):
        dt = datetime(2015, 1, 5)
        url = GDELTHistoricalFetcher.build_daily_url(dt)
        assert url == f"{GDELT_DAILY_BASE}20150105.export.CSV.zip"

    def test_url_format_matches_gdelt(self):
        """URL should match the exact GDELT daily file pattern."""
        dt = datetime(2020, 12, 31)
        url = GDELTHistoricalFetcher.build_daily_url(dt)
        assert url.startswith("http://data.gdeltproject.org/events/")
        assert url.endswith(".export.CSV.zip")
        assert "20201231" in url


# ------------------------------------------------------------------ #
# Column definitions
# ------------------------------------------------------------------ #

class TestColumnDefinitions:
    def test_daily_columns_count(self):
        """GDELT 1.0 daily format has exactly 58 columns."""
        assert len(GDELT_DAILY_COLUMNS) == 58

    def test_use_cols_are_subset(self):
        """All USE_COLS must exist in GDELT_DAILY_COLUMNS."""
        for col in USE_COLS:
            assert col in GDELT_DAILY_COLUMNS, f"{col} not in GDELT_DAILY_COLUMNS"

    def test_use_col_indices_correct(self):
        """USE_COL_INDICES should map to the right columns."""
        for i, idx in enumerate(USE_COL_INDICES):
            assert GDELT_DAILY_COLUMNS[idx] == USE_COLS[i]

    def test_core_columns_present(self):
        """Key analysis columns must be in USE_COLS."""
        required = [
            "EventCode", "EventBaseCode", "EventRootCode",
            "GoldsteinScale", "NumMentions", "AvgTone",
            "Actor1CountryCode", "Actor2CountryCode",
        ]
        for col in required:
            assert col in USE_COLS


# ------------------------------------------------------------------ #
# CAMEO classification
# ------------------------------------------------------------------ #

class TestClassifyEvent:
    def test_tariff_full_code(self, fetcher):
        etype, cat = fetcher._classify_event("162", "16")
        assert etype == "tariff"
        assert cat == "tariff_imposition"

    def test_tariff_4digit_code(self, fetcher):
        etype, cat = fetcher._classify_event("1611", "16")
        assert etype == "tariff"
        assert cat == "tariff_increase"

    def test_tariff_base_fallback(self, fetcher):
        """When 4-digit code not in map, 3-digit base should match."""
        # 1611 is a 4-digit code; base "161" maps to "tariff_increase"
        etype, cat = fetcher._classify_event("1611", "16")
        assert etype == "tariff"
        assert cat == "tariff_increase"

    def test_removed_routine_diplomacy(self, fetcher):
        """031* (routine diplomacy) was intentionally removed — should not classify."""
        etype, cat = fetcher._classify_event("0311", "03")
        assert etype == ""
        assert cat == ""

    def test_conflict_full_code(self, fetcher):
        etype, cat = fetcher._classify_event("190", "19")
        assert etype == "conflict"
        assert cat == "military_action"

    def test_conflict_root_fallback(self, fetcher):
        """Code not in specific maps but root code matches."""
        etype, cat = fetcher._classify_event("209", "20")
        assert etype == "conflict"
        assert cat == "conflict_outbreak"

    def test_unrelated_event(self, fetcher):
        """Events outside our CAMEO maps should return empty strings."""
        etype, cat = fetcher._classify_event("010", "01")
        assert etype == ""
        assert cat == ""

    def test_sanctions_imposed_tariff(self, fetcher):
        etype, cat = fetcher._classify_event("164", "16")
        assert etype == "tariff"
        assert cat == "sanctions_imposed"

    def test_export_ban(self, fetcher):
        etype, cat = fetcher._classify_event("163", "16")
        assert etype == "tariff"
        assert cat == "export_ban"

    def test_military_buildup(self, fetcher):
        etype, cat = fetcher._classify_event("150", "15")
        assert etype == "conflict"
        assert cat == "military_buildup"

    def test_shipping_disruption(self, fetcher):
        etype, cat = fetcher._classify_event("191", "19")
        assert etype == "conflict"
        assert cat == "shipping_disruption"


# ------------------------------------------------------------------ #
# Severity mapping
# ------------------------------------------------------------------ #

class TestGoldsteinToSeverity:
    def test_conflict_most_severe(self):
        assert GDELTHistoricalFetcher._goldstein_to_severity(-10.0, "conflict") == 10

    def test_conflict_moderate(self):
        assert GDELTHistoricalFetcher._goldstein_to_severity(-4.0, "conflict") == 7

    def test_conflict_mild(self):
        assert GDELTHistoricalFetcher._goldstein_to_severity(0.0, "conflict") == 5

    def test_conflict_positive(self):
        assert GDELTHistoricalFetcher._goldstein_to_severity(5.0, "conflict") == 4

    def test_tariff_high_magnitude(self):
        assert GDELTHistoricalFetcher._goldstein_to_severity(-9.5, "tariff") == 10

    def test_tariff_moderate_negative(self):
        assert GDELTHistoricalFetcher._goldstein_to_severity(-5.0, "tariff") == 7

    def test_tariff_positive_high(self):
        """Positive Goldstein for tariff (e.g., trade deal) should still map by magnitude."""
        assert GDELTHistoricalFetcher._goldstein_to_severity(7.5, "tariff") == 8

    def test_nan_returns_default(self):
        assert GDELTHistoricalFetcher._goldstein_to_severity(float("nan"), "conflict") == 5

    def test_severity_range(self):
        """All severities should be between 1 and 10."""
        for g in range(-10, 11):
            for etype in ["conflict", "tariff"]:
                sev = GDELTHistoricalFetcher._goldstein_to_severity(float(g), etype)
                assert 1 <= sev <= 10, f"Severity {sev} out of range for g={g}, type={etype}"


# ------------------------------------------------------------------ #
# Market inference
# ------------------------------------------------------------------ #

class TestInferMarkets:
    def test_us_china_bilateral(self, fetcher):
        markets = fetcher._infer_markets("US", "China")
        assert "US" in markets
        assert "China" in markets

    def test_india_direct(self, fetcher):
        markets = fetcher._infer_markets("India", "US")
        assert "India" in markets
        assert "US" in markets

    def test_russia_spillover(self, fetcher):
        """Russia affects all 3 markets."""
        markets = fetcher._infer_markets("Russia", "Ukraine")
        assert "US" in markets
        assert "India" in markets
        assert "China" in markets

    def test_taiwan_affects_us_china(self, fetcher):
        markets = fetcher._infer_markets("Taiwan", "China")
        assert "US" in markets
        assert "China" in markets

    def test_unknown_defaults_to_all(self, fetcher):
        """Unknown countries should default to all 3 markets."""
        markets = fetcher._infer_markets("Brazil", "Argentina")
        assert set(markets) == {"US", "India", "China"}

    def test_pakistan_affects_india(self, fetcher):
        markets = fetcher._infer_markets("Pakistan", "India")
        assert "India" in markets


# ------------------------------------------------------------------ #
# Processing pipeline
# ------------------------------------------------------------------ #

class TestProcessRaw:
    def test_basic_processing(self, fetcher):
        raw = pd.DataFrame([_make_raw_row()])
        result = fetcher._process_raw(raw)
        assert len(result) == 1
        assert result.iloc[0]["event_type"] == "tariff"
        assert result.iloc[0]["category"] == "tariff_imposition"
        assert result.iloc[0]["data_source"] == "gdelt_daily"

    def test_filter_by_event_type(self, fetcher):
        raw = pd.DataFrame([
            _make_raw_row(EventCode="162", EventRootCode="16"),  # tariff
            _make_raw_row(EventCode="190", EventRootCode="19"),  # conflict
        ])
        tariff_only = fetcher._process_raw(raw, "tariff")
        assert len(tariff_only) == 1
        assert tariff_only.iloc[0]["event_type"] == "tariff"

        conflict_only = fetcher._process_raw(raw, "conflict")
        assert len(conflict_only) == 1
        assert conflict_only.iloc[0]["event_type"] == "conflict"

    def test_unrelated_events_filtered(self, fetcher):
        raw = pd.DataFrame([_make_raw_row(EventCode="010", EventRootCode="01")])
        result = fetcher._process_raw(raw)
        assert len(result) == 0

    def test_country_mapping(self, fetcher):
        raw = pd.DataFrame([_make_raw_row(
            Actor1CountryCode="USA",
            Actor2CountryCode="IND",
        )])
        result = fetcher._process_raw(raw)
        assert result.iloc[0]["source_country"] == "US"
        assert result.iloc[0]["target_country"] == "India"

    def test_markets_affected_populated(self, fetcher):
        raw = pd.DataFrame([_make_raw_row()])
        result = fetcher._process_raw(raw)
        markets = result.iloc[0]["markets_affected"]
        assert isinstance(markets, list)
        assert len(markets) > 0

    def test_date_parsing(self, fetcher):
        raw = pd.DataFrame([_make_raw_row(Day="20230615")])
        result = fetcher._process_raw(raw)
        assert result.iloc[0]["date"] == pd.Timestamp("2023-06-15")

    def test_invalid_date_dropped(self, fetcher):
        raw = pd.DataFrame([_make_raw_row(Day="invalid")])
        result = fetcher._process_raw(raw)
        assert len(result) == 0


# ------------------------------------------------------------------ #
# Deduplication
# ------------------------------------------------------------------ #

class TestDeduplicate:
    def test_keeps_highest_severity(self, fetcher):
        events = pd.DataFrame([
            {
                "date": pd.Timestamp("2023-06-15"),
                "event_type": "tariff",
                "category": "tariff_imposition",
                "source_country": "US",
                "target_country": "China",
                "severity": 7,
            },
            {
                "date": pd.Timestamp("2023-06-15"),
                "event_type": "tariff",
                "category": "tariff_imposition",
                "source_country": "US",
                "target_country": "China",
                "severity": 9,
            },
        ])
        result = fetcher._deduplicate(events)
        assert len(result) == 1
        assert result.iloc[0]["severity"] == 9

    def test_different_categories_kept(self, fetcher):
        events = pd.DataFrame([
            {
                "date": pd.Timestamp("2023-06-15"),
                "event_type": "tariff",
                "category": "tariff_imposition",
                "source_country": "US",
                "target_country": "China",
                "severity": 7,
            },
            {
                "date": pd.Timestamp("2023-06-15"),
                "event_type": "tariff",
                "category": "export_ban",
                "source_country": "US",
                "target_country": "China",
                "severity": 8,
            },
        ])
        result = fetcher._deduplicate(events)
        assert len(result) == 2

    def test_different_country_pairs_kept(self, fetcher):
        events = pd.DataFrame([
            {
                "date": pd.Timestamp("2023-06-15"),
                "event_type": "tariff",
                "category": "tariff_imposition",
                "source_country": "US",
                "target_country": "China",
                "severity": 7,
            },
            {
                "date": pd.Timestamp("2023-06-15"),
                "event_type": "tariff",
                "category": "tariff_imposition",
                "source_country": "US",
                "target_country": "India",
                "severity": 6,
            },
        ])
        result = fetcher._deduplicate(events)
        assert len(result) == 2

    def test_empty_dataframe(self, fetcher):
        result = fetcher._deduplicate(pd.DataFrame())
        assert result.empty

    def test_2day_window_merges(self, fetcher):
        """Events in the same 2-day window with same key are merged."""
        # Pick two dates that land in the same 2-day period:
        # Period = (date - 2015-01-01).days // 2
        # 2023-06-14 = 3086 days from epoch => period 1543
        # 2023-06-15 = 3087 days from epoch => period 1543
        events = pd.DataFrame([
            {
                "date": pd.Timestamp("2023-06-14"),
                "event_type": "tariff",
                "category": "tariff_imposition",
                "source_country": "US",
                "target_country": "China",
                "severity": 7,
            },
            {
                "date": pd.Timestamp("2023-06-15"),
                "event_type": "tariff",
                "category": "tariff_imposition",
                "source_country": "US",
                "target_country": "China",
                "severity": 5,
            },
        ])
        result = fetcher._deduplicate(events)
        assert len(result) == 1
        assert result.iloc[0]["severity"] == 7


# ------------------------------------------------------------------ #
# Date range clamping
# ------------------------------------------------------------------ #

class TestDateRangeClamping:
    @patch.object(GDELTHistoricalFetcher, "_download_day", return_value=None)
    def test_clamps_to_gdelt_start(self, mock_dl, fetcher):
        """Dates before GDELT_DAILY_START should be clamped."""
        fetcher.fetch_historical_events(
            start_date="2010-01-01",
            end_date="2013-04-03",
        )
        # Should only download from Apr 1 2013 onward
        called_dates = [call.args[0] for call in mock_dl.call_args_list]
        for d in called_dates:
            assert d >= GDELT_DAILY_START

    @patch.object(GDELTHistoricalFetcher, "_download_day", return_value=None)
    def test_does_not_download_future(self, mock_dl, fetcher):
        """Should not try to download dates after today."""
        fetcher.fetch_historical_events(
            start_date="2023-01-01",
            end_date="2099-12-31",
        )
        called_dates = [call.args[0] for call in mock_dl.call_args_list]
        today = datetime.now()
        for d in called_dates:
            assert d <= today


# ------------------------------------------------------------------ #
# Download day (with mocked HTTP)
# ------------------------------------------------------------------ #

class TestDownloadDay:
    def _make_fake_zip(self, tsv_content: str) -> bytes:
        """Create a fake ZIP file containing a TSV."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("20230615.export.CSV", tsv_content)
        return buf.getvalue()

    def _make_tsv_row(self, overrides=None) -> str:
        """Build a single TSV row with 58 tab-separated fields.

        overrides: dict mapping column index (int) -> value (str)
        Column indices match GDELT_DAILY_COLUMNS:
          0=GlobalEventID, 1=Day, 5=Actor1Code, 6=Actor1Name,
          7=Actor1CountryCode, 15=Actor2Code, 16=Actor2Name,
          17=Actor2CountryCode, 25=IsRootEvent, 26=EventCode,
          27=EventBaseCode, 28=EventRootCode, 29=QuadClass,
          30=GoldsteinScale, 31=NumMentions, 32=NumSources,
          33=NumArticles, 34=AvgTone, 57=SOURCEURL
        """
        # Start with 58 empty fields
        fields = [""] * 58
        defaults = {
            0: "123456",          # GlobalEventID
            1: "20230615",        # Day
            2: "202306",          # MonthYear
            3: "2023",            # Year
            4: "2023.4548",       # FractionDate
            5: "USA",             # Actor1Code
            6: "UNITED STATES",   # Actor1Name
            7: "USA",             # Actor1CountryCode
            15: "CHN",            # Actor2Code
            16: "CHINA",          # Actor2Name
            17: "CHN",            # Actor2CountryCode
            25: "1",              # IsRootEvent
            26: "162",            # EventCode
            27: "162",            # EventBaseCode
            28: "16",             # EventRootCode
            29: "4",              # QuadClass
            30: "-7.0",           # GoldsteinScale
            31: "10",             # NumMentions
            32: "5",              # NumSources
            33: "3",              # NumArticles
            34: "-3.5",           # AvgTone
            57: "http://example.com",  # SOURCEURL
        }
        for idx, val in defaults.items():
            fields[idx] = val
        if overrides:
            for idx, val in overrides.items():
                fields[idx] = str(val)
        return "\t".join(fields)

    @patch("src.data_collection.gdelt_historical.requests.get")
    def test_successful_download(self, mock_get, fetcher):
        tsv = self._make_tsv_row()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = self._make_fake_zip(tsv)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetcher._download_day(datetime(2023, 6, 15))
        assert result is not None
        assert not result.empty

    @patch("src.data_collection.gdelt_historical.requests.get")
    def test_404_returns_none(self, mock_get, fetcher):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = fetcher._download_day(datetime(2023, 6, 15))
        assert result is None

    @patch("src.data_collection.gdelt_historical.requests.get")
    def test_filters_non_root_events(self, mock_get, fetcher):
        """Non-root events (IsRootEvent != 1) should be filtered out."""
        rows = [
            self._make_tsv_row(),                     # root event
            self._make_tsv_row(overrides={25: "0"}),  # non-root (IsRootEvent=0)
        ]
        tsv = "\n".join(rows)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = self._make_fake_zip(tsv)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetcher._download_day(datetime(2023, 6, 15))
        assert result is not None
        assert len(result) == 1

    @patch("src.data_collection.gdelt_historical.requests.get")
    def test_filters_low_mention_events(self, mock_get, fetcher):
        """Events with fewer than 3 mentions should be filtered out."""
        rows = [
            self._make_tsv_row(),                      # 10 mentions
            self._make_tsv_row(overrides={31: "1"}),   # 1 mention (NumMentions=1)
        ]
        tsv = "\n".join(rows)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = self._make_fake_zip(tsv)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetcher._download_day(datetime(2023, 6, 15))
        assert result is not None
        assert len(result) == 1

    def test_cached_result_returned(self, fetcher):
        """If a cached CSV exists, no HTTP call should be made."""
        cache_path = fetcher._download_cache / "20230615.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a valid cached CSV
        df = pd.DataFrame([_make_raw_row()])
        df.to_csv(cache_path, index=False)

        with patch("src.data_collection.gdelt_historical.requests.get") as mock_get:
            result = fetcher._download_day(datetime(2023, 6, 15))
            mock_get.assert_not_called()
            assert result is not None
            assert not result.empty


# ------------------------------------------------------------------ #
# Constants validation
# ------------------------------------------------------------------ #

class TestConstants:
    def test_gdelt_daily_start(self):
        """GDELT daily files start April 1, 2013."""
        assert GDELT_DAILY_START == datetime(2013, 4, 1)

    def test_target_countries_valid(self):
        """All target countries should be 3-letter codes."""
        for code in TARGET_COUNTRIES:
            assert len(code) == 3
            assert code.isupper()

    def test_country_map_covers_targets(self):
        """All target countries should have a mapping."""
        for code in TARGET_COUNTRIES:
            assert code in COUNTRY_MAP, f"{code} not in COUNTRY_MAP"

    def test_gdelt_daily_base_url(self):
        assert GDELT_DAILY_BASE == "http://data.gdeltproject.org/events/"
