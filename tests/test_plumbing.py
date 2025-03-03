import pandas as pd

import numpy as np

from tide.plumbing import (
    _get_pipe_from_proc_list,
    _get_column_wise_transformer,
    get_pipeline_from_dict,
    Plumber,
)

import plotly.io as pio
import pytest

pio.renderers.default = "browser"

TEST_DF = pd.DataFrame(
    {
        "Tin__°C__building": [10.0, 20.0, 30.0],
        "Text__°C__outdoor": [-1.0, 5.0, 4.0],
        "radiation__W/m2__outdoor": [50, 100, 400],
        "Humidity__%HR": [10, 15, 13],
        "Humidity__%HR__room1": [20, 30, 50],
        "Humidity_2": [10, 15, 13],
        "light__DIMENSIONLESS__building": [100, 200, 300],
        "mass_flwr__m3/h__hvac": [300, 500, 600],
    },
    index=pd.date_range("2009", freq="h", periods=3, tz="UTC"),
)

TEST_DF_2 = pd.DataFrame(
    {
        "a__°C__zone_1": np.random.randn(24),
        "b__°C__zone_1": np.random.randn(24),
        "c__Wh__zone_2": np.random.randn(24) * 100,
    },
    index=pd.date_range("2009", freq="h", periods=24, tz="UTC"),
)

TEST_DF_2["c__Wh__zone_2"] = abs(TEST_DF_2).cumsum()["c__Wh__zone_2"]

TEST_DF_2.loc["2009-01-01 05:00:00":"2009-01-01 09:00:00", "a__°C__zone_1"] = np.nan
TEST_DF_2.loc["2009-01-01 15:00:00", "b__°C__zone_1"] = np.nan
TEST_DF_2.loc["2009-01-01 17:00:00", "b__°C__zone_1"] = np.nan
TEST_DF_2.loc["2009-01-01 20:00:00", "c__Wh__zone_2"] = np.nan

PIPE_DICT = {
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


class TestPlumbing:
    def test__get_all_data_step(self):
        test_df = TEST_DF.copy()
        test_df.iloc[1, 0] = np.nan
        test_df.iloc[0, 1] = np.nan
        pipe = _get_pipe_from_proc_list(test_df.columns, PIPE_DICT["common"], tz="UTC")

        res = pipe.fit_transform(test_df)

        pd.testing.assert_series_equal(
            res["Tin__°C__building"], TEST_DF["Tin__°C__building"]
        )
        assert float(res.iloc[0, 1]) == 5.0

    def test__get_column_wise_transformer(self):
        col_trans = _get_column_wise_transformer(
            proc_dict=PIPE_DICT["pre_processing"],
            data_columns=TEST_DF.columns,
            tz="UTC",
            process_name="test",
        )

        res = col_trans.fit_transform(TEST_DF.copy())

        np.testing.assert_array_equal(res.iloc[:, 0].to_list(), [10.0, 20.0, np.nan])
        np.testing.assert_array_equal(res.iloc[:, 2].to_list(), [50.0, 100.0, np.nan])

        col_trans = _get_column_wise_transformer(
            proc_dict=PIPE_DICT["pre_processing"],
            data_columns=TEST_DF[
                [col for col in TEST_DF.columns if col != "radiation__W/m2__outdoor"]
            ].columns,
            tz="UTC",
            process_name="test",
        )

        res = col_trans.fit_transform(
            TEST_DF[
                [col for col in TEST_DF.columns if col != "radiation__W/m2__outdoor"]
            ].copy()
        )

        np.testing.assert_array_equal(res.iloc[:, 0].to_list(), [10.0, 20.0, np.nan])
        assert len(col_trans.transformers_) == 2

        cols_none = [
            "Humidity__%HR",
            "Humidity__%HR__room1",
            "Humidity_2",
            "light__DIMENSIONLESS__building",
            "mass_flwr__m3/h__hvac",
        ]

        col_trans = _get_column_wise_transformer(
            proc_dict=PIPE_DICT["pre_processing"],
            data_columns=cols_none,
            tz="UTC",
            process_name="test",
        )

        assert col_trans is None

    def test_get_pipeline_from_dict(self):
        pipe_dict = {
            "fill_1": {"a__°C__zone_1": [["Interpolate"]]},
            # "fill_2": {"b": [["Interpolate"]]},
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
            "fill_3": [["Interpolate"]],
        }

        pipe = get_pipeline_from_dict(TEST_DF_2.columns, pipe_dict, verbose=True)
        pipe.fit_transform(TEST_DF_2.copy())

        assert True

    def test_plumber(self):
        pipe = {
            "fill_1": {"a__°C__zone_1": [["Interpolate"]]},
            "fill_2": {"b": [["Interpolate"]]},
            "combine": {
                "zone_1": [
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
            },
            "fill_3": [["Interpolate"]],
        }

        plumber = Plumber()
        plumber.set_data(TEST_DF_2)
        plumber.pipe_dict = pipe
        plumber.get_pipeline()
        plumber.get_pipeline(steps=["fill_3", "combine"])
        plumber.plot()
        plumber.get_gaps_description()
        assert True


class TestGapsDescription:
    @pytest.fixture
    def sample_data(self):
        # Create sample data with known gaps
        idx = pd.date_range("2023-01-01", periods=24, freq="1h", tz="UTC")
        data = pd.DataFrame({
            "temp__°C__Building": np.ones(24),
            "humidity__%__Building": np.ones(24),
            "power__W__Building": np.ones(24)
        }, index=idx)
        
        # Create gaps of different durations
        data.loc["2023-01-01 02:00":"2023-01-01 04:00", "temp__°C__Building"] = np.nan  # 3h gap
        data.loc["2023-01-01 08:00", "temp__°C__Building"] = np.nan  # 1h gap
        data.loc["2023-01-01 12:00":"2023-01-01 14:00", "humidity__%__Building"] = np.nan  # 3h gap
        data.loc["2023-01-01 06:00":"2023-01-01 18:00", "power__W__Building"] = np.nan  # 13h gap
        
        return data

    def test_basic_gaps_description(self, sample_data):
        """Test basic functionality with default parameters"""
        plumber = Plumber(sample_data)
        result = plumber.get_gaps_description()
        
        # Check presence of all columns
        assert all(col in result.columns for col in sample_data.columns)
        
        # Check presence of all statistics
        expected_stats = ["data_presence_%", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        assert all(stat in result.index for stat in expected_stats)
        
        # Check specific values for temp column
        temp_col = "temp__°C__Building"
        assert result[temp_col]["count"] == 2  # Two gaps
        assert result[temp_col]["data_presence_%"] == pytest.approx(83.33, rel=1e-2)  # 20/24 hours present

    def test_with_duration_thresholds(self, sample_data):
        """Test with gap duration thresholds"""
        plumber = Plumber(sample_data)
        
        # Only gaps >= 3h
        result = plumber.get_gaps_description(gaps_gte="3h")
        assert result["temp__°C__Building"]["count"] == 1  # Only one 3h gap
        assert result["power__W__Building"]["count"] == 1  # One 13h gap
        
        # Only gaps <= 2h
        result = plumber.get_gaps_description(gaps_lte="2h")
        assert result["temp__°C__Building"]["count"] == 1  # Only one 1h gap
        assert "power__W__Building" not in result.columns  # No gaps <= 2h

    def test_with_data_selection(self, sample_data):
        """Test with data selection using tags"""
        plumber = Plumber(sample_data)
        
        # Select by unit
        result = plumber.get_gaps_description(select="°C")
        assert list(result.columns) == ["temp__°C__Building"]
        
        # Select by bloc
        result = plumber.get_gaps_description(select="Building")
        assert len(result.columns) == 3

    def test_empty_cases(self):
        """Test cases that should return empty DataFrame"""
        # Data with no gaps
        idx = pd.date_range("2023-01-01", periods=24, freq="1h", tz="UTC")
        clean_data = pd.DataFrame({
            "temp__°C__Building": np.ones(24)
        }, index=idx)
        plumber = Plumber(clean_data)
        
        result = plumber.get_gaps_description()
        assert result.empty
        
        # Data selection that returns no columns
        plumber = Plumber(clean_data)
        result = plumber.get_gaps_description(select="nonexistent")
        assert result.empty

    def test_combination_flag(self, sample_data):
        """Test with and without return_combination flag"""
        plumber = Plumber(sample_data)
        
        # With combination
        result = plumber.get_gaps_description(return_combination=True)
        assert "combination" in result.columns
        
        # Without combination
        result = plumber.get_gaps_description(return_combination=False)
        assert "combination" not in result.columns

    def test_single_point_gaps(self):
        """Test handling of single-point gaps"""
        idx = pd.date_range("2023-01-01", periods=24, freq="1h", tz="UTC")
        data = pd.DataFrame({
            "temp__°C__Building": np.ones(24)
        }, index=idx)
        
        # Create single point gap
        data.loc["2023-01-01 12:00", "temp__°C__Building"] = np.nan
        
        plumber = Plumber(data)
        result = plumber.get_gaps_description()
        
        assert result["temp__°C__Building"]["count"] == 1
        assert pd.Timedelta(result["temp__°C__Building"]["mean"]) == pd.Timedelta("1h")

    def test_pipeline_steps(self, sample_data):
        """Test with pipeline steps"""
        plumber = Plumber(sample_data)
        plumber.pipe_dict = {
            "step1": [["Identity"]],  # Simple identity transformation
            "step2": [["Identity"]]
        }
        
        # Test with specific steps
        result = plumber.get_gaps_description(steps=["step1"])
        assert not result.empty
        
        # Test with no steps
        result = plumber.get_gaps_description(steps=None)
        assert not result.empty
