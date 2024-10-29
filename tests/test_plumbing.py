import pandas as pd

from tide.plumbing import _get_pipe_from_proc_list, _get_column_wise_transformer
from tide.utils import data_columns_to_tree

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
        "°C": [["DROP_THRESHOLD", {"upper": 100}]],
        "outdoor__W/m2": [["DROP_TIME_GRADIENT", {"upper_rate": -100}]],
    },
    "common": {
        "ALL": [["INTERPOLATE", ["linear"]], ["FFILL"], ["BFILL", {"limit": 3}]]
    },
    "resampling": {
        "RESAMPLER": ["15min", {"energy": "sum"}],
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
        to_test = _get_pipe_from_proc_list(PIPE_DICT["common"]["ALL"])

        assert True

    def test__get_column_wise_transformer(self):
        col_trans = _get_column_wise_transformer(
            proc_dict=PIPE_DICT["pre_processing"],
            data_columns=TEST_DF.columns,
            process_name="test",
        )

        res = col_trans.fit_transform(TEST_DF)

        col_trans = _get_column_wise_transformer(
            proc_dict=PIPE_DICT["pre_processing"],
            data_columns=TEST_DF[
                [col for col in TEST_DF.columns if col != "radiation__W/m2__outdoor"]
            ].columns,
            process_name="test",
        )

        col_trans.fit_transform(
            TEST_DF[
                [col for col in TEST_DF.columns if col != "radiation__W/m2__outdoor"]
            ]
        )

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

        assert True
