import pandas as pd
import numpy as np
import pytest
from tide.plumbing import (
    _get_pipe_from_proc_list,
    _get_column_wise_transformer,
    get_pipeline_from_dict,
    Plumber,
)

import plotly.io as pio

pio.renderers.default = "browser"


@pytest.fixture
def time_index():
    """Create a standard time index for test data."""
    return pd.date_range("2009-01-01", freq="h", periods=24, tz="UTC")


@pytest.fixture
def basic_data(time_index):
    """Create basic test data with various units and tags."""
    return pd.DataFrame(
        {
            "Tin__°C__building": np.random.randn(24) * 5 + 20,
            "Text__°C__outdoor": np.random.randn(24) * 3 + 10,
            "radiation__W/m2__outdoor": np.abs(np.random.randn(24)) * 100,
            "Humidity__%HR": np.random.randn(24) * 5 + 50,
            "Humidity__%HR__room1": np.random.randn(24) * 5 + 45,
            "Humidity_2": np.random.randn(24) * 5 + 55,
            "light__DIMENSIONLESS__building": np.abs(np.random.randn(24)) * 200,
            "mass_flwr__m3/h__hvac": np.abs(np.random.randn(24)) * 400 + 500,
        },
        index=time_index,
    )


@pytest.fixture
def gapped_data(time_index):
    """Create test data with specific gaps for testing gap-related functionality."""
    data = pd.DataFrame(
        {
            "a__°C__zone_1": np.random.randn(24),
            "b__°C__zone_1": np.random.randn(24),
            "c__Wh__zone_2": np.abs(np.random.randn(24) * 100),
        },
        index=time_index,
    )

    # Add cumulative sum to energy data
    data["c__Wh__zone_2"] = data["c__Wh__zone_2"].cumsum()

    # Add specific gaps
    data.loc["2009-01-01 05:00":"2009-01-01 09:00", "a__°C__zone_1"] = np.nan  # 5h gap
    data.loc["2009-01-01 15:00", "b__°C__zone_1"] = np.nan  # 1h gap
    data.loc["2009-01-01 17:00", "b__°C__zone_1"] = np.nan  # 1h gap
    data.loc["2009-01-01 20:00", "c__Wh__zone_2"] = np.nan  # 1h gap

    return data


@pytest.fixture
def pipe_dict():
    """Create a standard pipeline dictionary for testing."""
    return {
        "pre_processing": {
            "°C": [["ReplaceThreshold", {"upper": 25}]],
            "W/m2__outdoor": [["DropTimeGradient", {"upper_rate": -100}]],
        },
        "common": [["Interpolate", ["linear"]], ["Ffill"], ["Bfill", {"limit": 3}]],
        "resampling": [["Resample", ["3h", "mean", {"W/m2": "sum"}]]],
        "compute_energy": [
            [
                "ExpressionCombine",
                [
                    {
                        "T1": "Tin__°C__building",
                        "T2": "Text__°C__outdoor",
                        "m": "mass_flwr__m3/h__hvac",
                    },
                    "(T1 - T2) * m * 1004 * 1.204",
                    "Air_flow_energy__hvac__J",
                    True,
                ],
            ]
        ],
    }


class TestPipelineComponents:
    """Tests for individual pipeline components and transformers."""

    def test_pipe_from_proc_list(self, pipe_dict):
        """Test creation and application of processing pipeline from list."""
        test_df = pd.DataFrame(
            {
                "temp__°C__building": [10.0, np.nan, 20.0, 30.0],
                "humid__%HR__building": [50.0, 60.0, np.nan, 80.0],
            },
            index=pd.date_range("2009", freq="h", periods=4, tz="UTC"),
        )

        pipe = _get_pipe_from_proc_list(test_df.columns, pipe_dict["common"], tz="UTC")
        result = pipe.fit_transform(test_df)

        # Check that gaps were filled with interpolation
        assert not result.isna().any().any()
        # For temp: 10 -> [15] -> 20 -> 30 (linear interpolation)
        assert result.iloc[1]["temp__°C__building"] == pytest.approx(15.0)
        # For humid: 50 -> 60 -> [70] -> 80 (linear interpolation)
        assert result.iloc[2]["humid__%HR__building"] == pytest.approx(70.0)

        # Check that non-gap values remain unchanged
        assert result.iloc[0]["temp__°C__building"] == 10.0
        assert result.iloc[3]["temp__°C__building"] == 30.0
        assert result.iloc[0]["humid__%HR__building"] == 50.0
        assert result.iloc[1]["humid__%HR__building"] == 60.0

    def test_column_wise_transformer(self, pipe_dict):
        """Test column-wise transformer creation and application."""
        # Create controlled test data with known values
        test_df = pd.DataFrame(
            {
                "temp1__°C__zone1": [24.0, 26.0, np.nan, 28.0],
                # Two values above threshold
                "temp2__°C__zone2": [23.0, 25.0, 27.0, np.nan],
                # One value above threshold
                "radiation__W/m2__outdoor": [100, 200, 50, 150],  # For gradient test
                "humid__%HR__zone1": [50.0, 60.0, 70.0, 80.0],  # Should be unaffected
            },
            index=pd.date_range("2009", freq="h", periods=4, tz="UTC"),
        )

        # Test with all columns
        transformer = _get_column_wise_transformer(
            proc_dict=pipe_dict["pre_processing"],
            data_columns=test_df.columns,
            tz="UTC",
            process_name="test",
        )
        result = transformer.fit_transform(test_df.copy())

        # Check temperature threshold applied (excluding NaN)
        temp1_mask = ~pd.isna(result["temp1__°C__zone1"])
        temp2_mask = ~pd.isna(result["temp2__°C__zone2"])
        assert (result["temp1__°C__zone1"][temp1_mask] <= 25).all()
        assert (result["temp2__°C__zone2"][temp2_mask] <= 25).all()

        # Verify specific values
        assert result.iloc[0]["temp1__°C__zone1"] == 24.0  # Unchanged
        assert result.iloc[1]["temp2__°C__zone2"] == 25.0  # Capped
        assert pd.isna(result.iloc[2]["temp1__°C__zone1"])  # NaN preserved
        assert pd.isna(result.iloc[3]["temp1__°C__zone1"])  # Capped

        # Check radiation gradient (should drop when rate < -100)
        assert pd.isna(
            result.iloc[2]["radiation__W/m2__outdoor"]
        )  # Dropped due to steep negative gradient

        # Check humidity unaffected
        pd.testing.assert_series_equal(
            result["humid__%HR__zone1"], test_df["humid__%HR__zone1"]
        )

        # Test with subset of columns (temperature only)
        temp_cols = [col for col in test_df.columns if "°C" in col]
        transformer = _get_column_wise_transformer(
            proc_dict=pipe_dict["pre_processing"],
            data_columns=temp_cols,
            tz="UTC",
            process_name="test",
        )
        assert len(transformer.transformers_) == 1  # Only temperature transformer

        # Test with no matching columns
        humidity_cols = [col for col in test_df.columns if "%HR" in col]
        transformer = _get_column_wise_transformer(
            proc_dict=pipe_dict["pre_processing"],
            data_columns=humidity_cols,
            tz="UTC",
            process_name="test",
        )
        assert transformer is None  # No transformers needed

    def test_pipeline_from_dict(self, gapped_data):
        """Test creation of full pipeline from dictionary configuration."""
        pipe_dict = {
            "fill_1": {"a__°C__zone_1": [["Interpolate"]]},
            "combine": [
                [
                    "ExpressionCombine",
                    [
                        {
                            "T1": "a__°C__zone_1",
                            "T2": "b__°C__zone_1",
                        },
                        "T1 * T2",
                        "new_unit__°C²__zone_1",
                        True,
                    ],
                ]
            ],
            "fill_final": [["Interpolate"]],
        }

        pipe = get_pipeline_from_dict(gapped_data.columns, pipe_dict, verbose=True)
        result = pipe.fit_transform(gapped_data.copy())

        # Check new column created
        assert "new_unit__°C²__zone_1" in result.columns

        # Check gaps filled
        assert not result.isna().any().any()


class TestPlumber:
    """Tests for the Plumber class functionality."""

    def test_initialization(self, gapped_data, pipe_dict):
        """Test Plumber initialization and basic attributes."""
        plumber = Plumber(gapped_data, pipe_dict)
        assert plumber.data is not None
        assert plumber.root is not None
        assert plumber.pipe_dict == pipe_dict

    def test_data_selection(self, gapped_data):
        """Test data selection using tags."""
        plumber = Plumber(gapped_data)

        # Test unit selection
        temp_cols = plumber.select("°C")
        assert len(temp_cols) == 2
        assert all("°C" in col for col in temp_cols)

        # Test zone selection
        zone_1_cols = plumber.select("zone_1")
        assert len(zone_1_cols) == 2
        assert all("zone_1" in col for col in zone_1_cols)

    def test_pipeline_execution(self, basic_data, pipe_dict):
        """Test pipeline execution with different step selections."""
        plumber = Plumber(basic_data, pipe_dict)

        # Test full pipeline
        full_pipe = plumber.get_pipeline()
        assert len(full_pipe.steps) > 0

        # Test partial pipeline
        partial_pipe = plumber.get_pipeline(steps=["pre_processing"])
        assert len(partial_pipe.steps) == 1

        # Test with no pipeline
        identity_pipe = plumber.get_pipeline(steps=None)
        assert len(identity_pipe.steps) == 1
        assert identity_pipe.steps[0][0] == "Identity"

    def test_corrected_data(self, basic_data, pipe_dict):
        """Test data correction through pipeline."""
        plumber = Plumber(basic_data, pipe_dict)

        # Test with time slice
        result = plumber.get_corrected_data(
            start="2009-01-01 05:00", stop="2009-01-01 10:00"
        )
        assert len(result) == 3


class TestGapsDescription:
    """Tests for gap analysis functionality."""

    @pytest.fixture
    def gaps_data(self, time_index):
        """Create data with specific gaps for testing gap analysis."""
        data = pd.DataFrame(
            {
                "temp__°C__Building": np.ones(24),
                "humidity__%__Building": np.ones(24),
                "power__W__Building": np.ones(24),
            },
            index=time_index,
        )

        # Create gaps of different durations
        data.loc["2009-01-01 02:00":"2009-01-01 04:00", "temp__°C__Building"] = np.nan
        data.loc["2009-01-01 08:00", "temp__°C__Building"] = np.nan
        data.loc["2009-01-01 12:00":"2009-01-01 14:00", "humidity__%__Building"] = (
            np.nan
        )
        data.loc["2009-01-01 06:00":"2009-01-01 18:00", "power__W__Building"] = np.nan

        return data

    def test_basic_gaps_description(self):
        """Test basic gap analysis functionality."""
        my_df = pd.DataFrame({

            "temp__°C__Building": [np.nan, 1, np.nan, 3],
            "power__W__Building": [np.nan, 1, 2, np.nan],
        }, index=pd.date_range("2009", freq="h", periods=4, tz='UTC'))

        plumber = Plumber(my_df)
        result = plumber.get_gaps_description()

        # Check structure
        assert all(col in result.columns for col in my_df.columns)
        expected_stats = [
            "data_presence_%",
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
        assert all(stat in result.index for stat in expected_stats)

        # Check specific values
        temp_col = "temp__°C__Building"
        assert result[temp_col]["count"] == 2
        assert result[temp_col]["data_presence_%"] == pytest.approx(50., rel=1e-2)
        assert result["combination"]["max"] == pd.to_timedelta("02:00:00")

    def test_gap_thresholds(self, gaps_data):
        """Test gap analysis with duration thresholds."""
        plumber = Plumber(gaps_data)

        # Test minimum duration threshold
        result = plumber.get_gaps_description(gaps_gte="3h")
        assert result["temp__°C__Building"]["count"] == 1
        assert result["power__W__Building"]["count"] == 1

        # Test maximum duration threshold
        result = plumber.get_gaps_description(gaps_lte="2h")
        assert result["temp__°C__Building"]["count"] == 1
        assert "power__W__Building" not in result.columns

    def test_gap_analysis_edge_cases(self, time_index):
        """Test gap analysis edge cases."""
        # Test with no gaps
        clean_data = pd.DataFrame({"temp__°C__Building": np.ones(24)}, index=time_index)
        plumber = Plumber(clean_data)
        result = plumber.get_gaps_description()
        assert result.empty

        # Test with invalid selection
        result = plumber.get_gaps_description(select="nonexistent")
        assert result.empty

        # Test single point gap
        data = clean_data.copy()
        data.loc[data.index[12], "temp__°C__Building"] = np.nan
        plumber = Plumber(data)
        result = plumber.get_gaps_description()
        assert result["temp__°C__Building"]["count"] == 1
        assert pd.Timedelta(result["temp__°C__Building"]["mean"]) == pd.Timedelta("1h")


class TestPlotting:
    """Tests for plotting functionality."""

    def test_basic_plot(self, gapped_data):
        """Test basic plotting functionality."""
        plumber = Plumber(gapped_data)
        fig = plumber.plot()

        # Check figure was created
        assert fig is not None
        # Check data is present in figure
        assert len(fig.data) > 0
        # Check all columns are plotted
        assert all(
            col in [trace.name for trace in fig.data] for col in gapped_data.columns
        )
