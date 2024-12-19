import pandas as pd

import numpy as np

from tide.plumbing import (
    _get_pipe_from_proc_list,
    _get_column_wise_transformer,
    get_pipeline_from_dict,
    Plumber,
)

import plotly.io as pio

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
    index=pd.date_range("2009", freq="h", periods=3),
)

TEST_DF_2 = pd.DataFrame(
    {
        "a__°C__zone_1": np.random.randn(24),
        "b__°C__zone_1": np.random.randn(24),
        "c__Wh__zone_2": np.random.randn(24) * 100,
    },
    index=pd.date_range("2009", freq="h", periods=24),
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
        pipe = _get_pipe_from_proc_list(PIPE_DICT["common"])

        res = pipe.fit_transform(test_df)

        pd.testing.assert_series_equal(
            res["Tin__°C__building"], TEST_DF["Tin__°C__building"]
        )
        assert float(res.iloc[0, 1]) == 5.0

    def test__get_column_wise_transformer(self):
        col_trans = _get_column_wise_transformer(
            proc_dict=PIPE_DICT["pre_processing"],
            data_columns=TEST_DF.columns,
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

        plumber = Plumber()
        plumber.set_data(TEST_DF_2)
        plumber.pipe_dict = pipe

        plumber.get_pipeline()
        plumber.get_pipeline(steps=["fill_3", "combine"])

        plumber.plot()

        assert True
