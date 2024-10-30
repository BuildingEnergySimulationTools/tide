import pandas as pd

import numpy as np

from tide.plumbing import (
    _get_pipe_from_proc_list,
    _get_column_wise_transformer,
)

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

PIPE_DICT = {
    "pre_processing": {
        "°C": [["DropThreshold", {"upper": 25}]],
        "outdoor__W/m2": [["DropTimeGradient", {"upper_rate": -100}]],
    },
    "common": [["Interpolate", ["linear"]], ["Ffill"], ["Bfill", {"limit": 3}]],
    "resampling": {
        "RESAMPLER": ["3h", {"W/m2": "SUM"}],
    },
    "Compute_energy": {
        "DATACOMBINER": [
            ["T1", "T2", "M"],
            '("T1"-"T2") * M * 2001',
            "Air_flow_energy__hvac__J",
            True,
        ],
    },
}


class TestPlumbing:
    def test__get_all_data_step(self):
        test_df = TEST_DF.copy()
        test_df.iloc[1, 0] = np.nan
        test_df.iloc[0, 1] = np.nan
        pipe = _get_pipe_from_proc_list(PIPE_DICT["common"]["ALL"])

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

