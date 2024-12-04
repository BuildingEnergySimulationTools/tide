import datetime as dt

import pandas as pd
import numpy as np

from tide.utils import (
    get_data_blocks,
    get_outer_timestamps,
    data_columns_to_tree,
    get_data_col_names_from_root,
    get_data_level_names,
    parse_request_to_col_names,
    timedelta_to_int,
    NamedList,
    get_series_bloc,
)

DF_COLUMNS = pd.DataFrame(
    columns=[
        "name_1__°C__bloc1",
        "name_1__°C__bloc2",
        "name_2",
        "name_2__DIMENSIONLESS__bloc2",
        "name_3__kWh/m²",
        "name_5__kWh",
        "name4__DIMENSIONLESS__bloc4",
    ]
)


class TestUtils:
    def test_named_list(self):
        test = NamedList(["a", "b", "c", "d"])

        assert test["a"] == ["a"]
        assert test[["a", "d"]] == ["a", "d"]
        assert test[:"b"] == ["a", "b"]

    def test_columns_parser(self):
        root = data_columns_to_tree(DF_COLUMNS.columns)
        col_names = get_data_col_names_from_root(root)
        assert all(col in DF_COLUMNS.columns for col in col_names)

    def test_parse_request_to_col_names(self):
        res = parse_request_to_col_names(DF_COLUMNS)
        assert res == [
            "name_1__°C__bloc1",
            "name_1__°C__bloc2",
            "name_2",
            "name_2__DIMENSIONLESS__bloc2",
            "name_3__kWh/m²",
            "name_5__kWh",
            "name4__DIMENSIONLESS__bloc4",
        ]

        res = parse_request_to_col_names(DF_COLUMNS, "name_1__°C__bloc1")
        assert res == ["name_1__°C__bloc1"]

        res = parse_request_to_col_names(
            DF_COLUMNS,
            [
                "name_1__°C__bloc1",
                "name_1__°C__bloc2",
            ],
        )
        assert res == [
            "name_1__°C__bloc1",
            "name_1__°C__bloc2",
        ]

        res = parse_request_to_col_names(DF_COLUMNS, "°C")
        assert res == ["name_1__°C__bloc1", "name_1__°C__bloc2"]

        res = parse_request_to_col_names(DF_COLUMNS, "OTHER")
        assert res == ["name_2", "name_3__kWh/m²", "name_5__kWh"]

        res = parse_request_to_col_names(DF_COLUMNS, "DIMENSIONLESS__bloc2")
        assert res == ["name_2__DIMENSIONLESS__bloc2"]

        res = parse_request_to_col_names(DF_COLUMNS, "kWh")
        assert res == ["name_5__kWh"]

    def test_get_data_level_names(self):
        root = data_columns_to_tree(DF_COLUMNS.columns)
        res = get_data_level_names(root, "name")
        assert res == [
            "name_1",
            "name_1",
            "name_2",
            "name_2",
            "name_3",
            "name_5",
            "name4",
        ]

        res = get_data_level_names(root, "unit")
        assert res == ["°C", "DIMENSIONLESS", "kWh/m²", "kWh"]

        res = get_data_level_names(root, "bloc")
        assert res == ["bloc1", "bloc2", "OTHER", "bloc4"]

    def test_get_series_bloc(self):
        toy_sr = pd.Series(
            data=np.arange(24).astype(float),
            index=pd.date_range("2009", freq="h", periods=24),
            name="data_1",
        )

        toy_holes = toy_sr.copy()
        toy_holes.loc["2009-01-01 02:00:00"] = np.nan
        toy_holes.loc["2009-01-01 05:00:00":"2009-01-01 08:00:00"] = np.nan
        toy_holes.loc["2009-01-01 12:00:00":"2009-01-01 16:00:00"] = np.nan

        get_series_bloc(
            toy_holes,
            is_null=True,
            upper_td_threshold="3h",
            upper_threshold_inclusive=False,
        )

        # All data groups
        assert len(get_series_bloc(toy_holes)) == 4

        # All gaps groups
        assert len(get_series_bloc(toy_holes, is_null=True)) == 3

        # Gaps Inner bounds, one inclusive
        assert (
            len(
                get_series_bloc(
                    toy_holes,
                    is_null=True,
                    select_inner=True,
                    lower_td_threshold="1h",
                    lower_threshold_inclusive=False,
                    upper_td_threshold="4h",
                    upper_threshold_inclusive=True,
                )
            )
            == 1
        )

        # Gaps outer selection, one inclusive
        assert (
            len(
                get_series_bloc(
                    toy_holes,
                    is_null=True,
                    select_inner=False,
                    lower_td_threshold="1h",
                    lower_threshold_inclusive=False,
                    upper_td_threshold="4h",
                    upper_threshold_inclusive=True,
                )
            )
            == 2
        )

    def test_get_data_blocks(self):
        toy_df = pd.DataFrame(
            {"data_1": np.random.randn(24), "data_2": np.random.randn(24)},
            index=pd.date_range("2009-01-01", freq="h", periods=24),
        )

        toy_df.loc["2009-01-01 01:00:00", "data_1"] = np.nan
        toy_df.loc["2009-01-01 10:00:00":"2009-01-01 12:00:00", "data_1"] = np.nan
        toy_df.loc["2009-01-01 15:00:00":"2009-01-01 23:00:00", "data_2"] = np.nan

        res = get_data_blocks(
            toy_df,
            is_null=False,
            lower_td_threshold="1h30min",
            upper_td_threshold="8h",
        )
        assert len(res["data_1"]) == 1

        res = get_data_blocks(toy_df, is_null=True)
        assert len(res["combination"]) == 3
        pd.testing.assert_index_equal(
            res["data_1"][0], pd.DatetimeIndex(["2009-01-01 01:00:00"])
        )
        pd.testing.assert_index_equal(
            res["data_2"][0], pd.date_range("2009-01-01 15:00:00", freq="h", periods=9)
        )

        res = get_data_blocks(toy_df, is_null=True, lower_td_threshold="1h30min")
        assert len(res["data_1"]) == 1

        res = get_data_blocks(toy_df, return_combination=False)
        assert "combination" not in res.keys()

        # CAREFUL !!! Remove timestamps to get indexes without frequency
        toy_df.drop(
            pd.date_range("2009-01-01 02:00:00", "2009-01-01 04:00:00", freq="h"),
            axis=0,
            inplace=True,
        )

        # The gap from 01:00:00 to 04:00:00 shall be identified.
        res = get_data_blocks(toy_df, is_null=True, lower_td_threshold="3h")
        assert len(res["data_1"]) == 2

        res = get_data_blocks(
            toy_df,
            is_null=True,
            lower_td_threshold="3h",
            lower_threshold_inclusive=False,
        )
        assert len(res["data_1"]) == 1

        res = get_data_blocks(
            toy_df,
            is_null=True,
            upper_td_threshold="3h",
            upper_threshold_inclusive=False,
        )
        assert res["data_1"] == []

    def test_outer_timestamps(self):
        ref_index = pd.date_range("2009-01-01", freq="d", periods=5)
        idx = pd.date_range("2009-01-02", freq="d", periods=2)
        start, end = get_outer_timestamps(idx, ref_index)

        assert start == pd.to_datetime("2009-01-01")
        assert end == pd.to_datetime("2009-01-04")

        start, end = get_outer_timestamps(ref_index, ref_index)
        assert start == ref_index[0]
        assert end == ref_index[-1]

    def test_timedelta_to_int(self):
        X = pd.DataFrame(
            {"a": np.arange(10 * 6 * 24)},
            index=pd.date_range(dt.datetime.now(), freq="10min", periods=10 * 6 * 24),
        )

        assert timedelta_to_int("24h", X) == 144
        assert timedelta_to_int(144, X) == 144
        assert timedelta_to_int(dt.timedelta(hours=24), X) == 144
