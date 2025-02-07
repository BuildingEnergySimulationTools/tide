import datetime as dt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from tide.processing import (
    AddTimeLag,
    ApplyExpression,
    Resample,
    CombineColumns,
    ReplaceThreshold,
    DropTimeGradient,
    Dropna,
    FillNa,
    Bfill,
    Ffill,
    GaussianFilter1D,
    Identity,
    RenameColumns,
    SkTransform,
    TimeGradient,
    ReplaceDuplicated,
    STLFilter,
    FillGapsAR,
    Interpolate,
    ExpressionCombine,
    FillOikoMeteo,
    AddOikoData,
    AddSolarAngles,
    ProjectSolarRadOnSurfaces,
    FillOtherColumns,
    DropColumns,
    ReplaceTag,
)

RESOURCES_PATH = Path(__file__).parent / "resources"


def mock_get_oikolab_df(**kwargs):
    data = pd.read_csv(
        Path(RESOURCES_PATH / "oiko_mockup.csv"), index_col=0, parse_dates=True
    )
    data.index.freq = data.index.inferred_freq

    try:
        param = kwargs["param"]
    except KeyError:
        param = [
            "temperature",
            "dewpoint_temperature",
            "mean_sea_level_pressure",
            "wind_speed",
            "100m_wind_speed",
            "relative_humidity",
            "surface_solar_radiation",
            "direct_normal_solar_radiation",
            "surface_diffuse_solar_radiation",
            "surface_thermal_radiation",
            "total_cloud_cover",
            "total_precipitation",
        ]

    start = kwargs["start"].strftime("%Y-%m-%d")
    end = kwargs["end"].strftime("%Y-%m-%d")
    return data.loc[
        start:end,
        [
            "coordinates (lat,lon)",
            "model (name)",
            "model elevation (surface)",
            "utc_offset (hrs)",
        ]
        + param,
    ]


class TestCustomTransformers:
    def test_pd_identity(self):
        df = pd.DataFrame(
            {"a": [1.0]}, index=pd.date_range("2009", freq="h", periods=1, tz="UTC")
        )

        identity = Identity()
        res = identity.fit_transform(df)

        assert list(df.columns) == list(identity.get_feature_names_out())
        pd.testing.assert_frame_equal(df, res)

    def test_pd_replace_duplicated(self):
        df = pd.DataFrame(
            {"a": [1.0, 1.0, 2.0], "b": [3.0, np.nan, 3.0]},
            pd.date_range("2009-01-01", freq="h", periods=3, tz="UTC"),
        )

        res = pd.DataFrame(
            {"a": [1.0, np.nan, 2.0], "b": [3.0, np.nan, np.nan]},
            pd.date_range("2009-01-01", freq="h", periods=3, tz="UTC"),
        )

        rep_dup = ReplaceDuplicated(keep="first", value=np.nan)
        res_dup = rep_dup.fit_transform(df)

        pd.testing.assert_frame_equal(res_dup, res)

    def test_pd_dropna(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0]},
            index=pd.date_range("2009", freq="h", periods=3, tz="UTC"),
        )

        ref = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        dropper = Dropna(how="any")

        dropper.fit(df)
        assert list(df.columns) == list(dropper.get_feature_names_out())
        pd.testing.assert_frame_equal(dropper.transform(df), ref)

    def test_pd_rename_columns(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0]},
            index=pd.date_range("2009", freq="h", periods=3, tz="UTC"),
        )

        new_cols = ["c", "d"]

        renamer = RenameColumns(new_names=new_cols)

        renamer.fit(df)
        assert list(df.columns) == list(renamer.get_feature_names_in())
        assert list(renamer.get_feature_names_out()) == new_cols
        assert list(renamer.transform(df).columns) == new_cols

        new_cols_dict = {"d": "a"}
        renamer = RenameColumns(new_names=new_cols_dict)

        assert list(renamer.fit_transform(df).columns) == ["c", "a"]

        inversed = renamer.inverse_transform(
            pd.DataFrame(
                np.zeros((2, 2)), pd.date_range("2009", freq="h", periods=2, tz="UTC")
            )
        )
        assert list(inversed.columns) == ["c", "a"]

    def test_pd_sk_transformer(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        scaler = SkTransform(StandardScaler())
        to_test = scaler.fit_transform(df)

        ref = pd.DataFrame(
            {"a": [-1.0, 1.0], "b": [-1.0, 1.0]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        pd.testing.assert_frame_equal(to_test, ref)
        assert list(df.columns) == list(scaler.get_feature_names_out())

        pd.testing.assert_frame_equal(scaler.inverse_transform(to_test), df)

    def test_pd_replace_threshold(self):
        df = pd.DataFrame(
            {"col1": [1, 2, 3, np.nan, 4], "col2": [1, np.nan, np.nan, 4, 5]},
            index=pd.date_range("2009", freq="h", periods=5, tz="UTC"),
        )

        ref = pd.DataFrame(
            {"col1": [0.0, 2, 3, np.nan, 4], "col2": [0.0, np.nan, np.nan, 4, 5]},
            index=pd.date_range("2009", freq="h", periods=5, tz="UTC"),
        )

        dropper = ReplaceThreshold(lower=1.1, upper=5, value=0.0)
        dropper.fit(df)

        assert list(df.columns) == list(dropper.get_feature_names_out())

        pd.testing.assert_frame_equal(dropper.transform(df), ref)

        # check do nothing
        dropper = ReplaceThreshold()
        pd.testing.assert_frame_equal(dropper.transform(df), df)

    def test_pd_drop_time_gradient(self):
        time_index = pd.date_range("2021-01-01 00:00:00", freq="h", periods=8, tz="UTC")

        df = pd.DataFrame(
            {
                "dumb_column": [5, 5.1, 5.1, 6, 7, 22, 6, 5],
                "dumb_column2": [5, 5, 5.1, 6, 22, 6, np.nan, 6],
            },
            index=time_index,
        )

        ref = pd.DataFrame(
            {
                "dumb_column": [5.0, 5.1, np.nan, 6.0, 7.0, np.nan, 6.0, 5.0],
                "dumb_column2": [5.0, np.nan, 5.1, 6.0, np.nan, 6.0, np.nan, np.nan],
            },
            index=time_index,
        )

        dropper = DropTimeGradient(lower_rate=0, upper_rate=0.004)

        pd.testing.assert_frame_equal(ref, dropper.fit_transform(df))

        # check do nothing
        dropper = DropTimeGradient()
        pd.testing.assert_frame_equal(dropper.transform(df), df)

    def test_pd_apply_expression(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        ref = pd.DataFrame(
            {"a": [2.0, 4.0], "b": [6.0, 8.0]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        transformer = ApplyExpression("X * 2")

        pd.testing.assert_frame_equal(ref, transformer.fit_transform(df))

        df = pd.DataFrame(
            {"a__W": [1.0, 2.0], "b__W": [3.0, 4.0]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        ref = pd.DataFrame(
            {"a__kW": [0.001, 0.002], "b__kW": [0.003, 0.004]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        transformer = ApplyExpression("X / 1000", "kW")

        pd.testing.assert_frame_equal(ref, transformer.fit_transform(df))

    def test_pd_time_gradient(self):
        test = (
            pd.DataFrame(
                {"cpt1__J": [0, 1, 2, 2, 2, 3], "cpt2__J": [0, 1, 2, 2, 2, 3]},
                index=pd.date_range(
                    "2009-01-01 00:00:00", freq="10s", periods=6, tz="UTC"
                ),
            )
            * 3600
        )

        ref = pd.DataFrame(
            {
                "cpt1__W": [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0],
                "cpt2__W": [360.0, 360.0, 180.0, -5.68e-14, 180.0, 360.0],
            },
            index=pd.date_range("2009-01-01 00:00:00", freq="10s", periods=6, tz="UTC"),
        )

        derivator = TimeGradient(new_unit="W")

        pd.testing.assert_frame_equal(ref, derivator.fit_transform(test), rtol=0.01)

    def test_pd_ffill(self):
        test = pd.DataFrame(
            {
                "cpt1": [0.0, np.nan, 2.0, 2.0, np.nan, np.nan],
                "cpt2": [0.0, 1.0, 2.0, 2.0, np.nan, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )

        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 2.0, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )

        filler = Ffill()
        pd.testing.assert_frame_equal(ref, filler.fit_transform(test.copy()))

        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 0.0, 2.0, 2.0, np.nan, np.nan],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 2.0, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )

        filler = Ffill(gaps_lte="1h")
        pd.testing.assert_frame_equal(ref, filler.fit_transform(test.copy()))

    def test_pd_bfill(self):
        test = pd.DataFrame(
            {
                "cpt1": [np.nan, np.nan, 2.0, 2.0, np.nan, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, np.nan, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )

        ref = pd.DataFrame(
            {
                "cpt1": [2.0, 2.0, 2.0, 2.0, 3.0, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )

        filler = Bfill()
        pd.testing.assert_frame_equal(ref, filler.fit_transform(test.copy()))

        filler = Bfill(gaps_lte="1h")
        ref = pd.DataFrame(
            {
                "cpt1": [np.nan, np.nan, 2.0, 2.0, 3.0, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )
        pd.testing.assert_frame_equal(ref, filler.fit_transform(test.copy()))

    def test_pd_fill_na(self):
        test = pd.DataFrame(
            {
                "cpt1": [0.0, np.nan, 2.0, 2.0, np.nan, np.nan],
                "cpt2": [0.0, 1.0, 2.0, 2.0, np.nan, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )

        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 0.0, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )

        filler = FillNa(value=0.0)
        pd.testing.assert_frame_equal(ref, filler.fit_transform(test.copy()))

        filler = FillNa(value=0.0, gaps_lte="1h")
        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 0.0, 2.0, 2.0, np.nan, np.nan],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 0.0, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )
        pd.testing.assert_frame_equal(ref, filler.fit_transform(test.copy()))

        assert True

    def test_resampler(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "col0": np.arange(10) * 100,
                "col1__°C": np.arange(10),
                "col2__°C": np.random.random(10),
                "col3": np.random.random(10) * 10,
            },
            index=pd.date_range("2009-01-01", freq="h", periods=10, tz="UTC"),
        ).astype("float")

        ref = pd.DataFrame(
            {
                "col0": [400.0, 900.0],
                "col1__°C": [2.0, 7.0],
                "col2__°C": [0.56239, 0.47789],
                "col3": [9.69910, 5.24756],
            },
            index=pd.date_range("2009-01-01 00:00:00", freq="5h", periods=2, tz="UTC"),
        ).astype("float")

        column_resampler = Resample(
            rule="5h",
            method="max",
            columns_methods=[(["col2__°C"], "mean"), (["col1__°C"], "mean")],
        )

        pd.testing.assert_frame_equal(
            ref, column_resampler.fit_transform(df).astype("float"), atol=0.01
        )

        column_resampler = Resample(
            rule="5h",
            method="max",
            tide_format_methods={"°C": "mean"},
        )
        pd.testing.assert_frame_equal(
            ref, column_resampler.fit_transform(df).astype("float"), atol=0.01
        )

        column_resampler = Resample(
            rule="5h",
            method="max",
        )

        np.testing.assert_almost_equal(
            column_resampler.fit_transform(df.copy()).to_numpy(),
            np.array(
                [
                    [4.00000000e02, 4.00000000e00, 9.50714306e-01, 9.69909852e00],
                    [9.00000000e02, 9.00000000e00, 8.66176146e-01, 5.24756432e00],
                ]
            ),
            decimal=1,
        )

    def test_pd_add_time_lag(self):
        df = pd.DataFrame(
            {
                "col0": np.arange(2),
                "col1": np.arange(2) * 10,
            },
            index=pd.date_range("2009-01-01", freq="h", periods=2, tz="UTC"),
        )

        ref = pd.DataFrame(
            {
                "col0": [1.0],
                "col1": [10.0],
                "1:00:00_col0": [0.0],
                "1:00:00_col1": [0.0],
            },
            index=pd.DatetimeIndex(
                ["2009-01-01 01:00:00"], dtype="datetime64[ns, UTC]", freq="h", tz="UTC"
            ),
        )

        lager = AddTimeLag(time_lag=dt.timedelta(hours=1), drop_resulting_nan=True)
        lager.fit(df)
        lager.get_feature_names_out()
        assert list(lager.get_feature_names_out()) == [
            "col0",
            "col1",
            "1:00:00_col0",
            "1:00:00_col1",
        ]
        pd.testing.assert_frame_equal(ref, lager.fit_transform(df))

    def test_pd_gaussian_filter(self):
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            index=pd.date_range("2009", freq="h", periods=3, tz="UTC"),
        )

        gfilter = GaussianFilter1D()

        to_test = gfilter.fit_transform(df)

        np.testing.assert_almost_equal(
            gaussian_filter1d(
                df.to_numpy()[:, 0].T, sigma=5, mode="nearest", truncate=4.0
            ),
            to_test.to_numpy()[:, 0],
            decimal=5,
        )

        assert list(to_test.columns) == list(df.columns)

    def test_pd_combine_columns(self):
        x_in = pd.DataFrame(
            {"a__°C": [1, 2], "b__°C": [1, 2], "c": [1, 2]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        trans = CombineColumns(
            function=np.sum,
            columns=["a__°C", "b__°C"],
            function_kwargs={"axis": 1},
            drop_columns=True,
        )

        pd.testing.assert_frame_equal(
            trans.fit_transform(x_in.copy()),
            pd.DataFrame(
                {"c": [1, 2], "combined": [2, 4]},
                index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
            ),
        )

        assert list(trans.get_feature_names_out()) == ["c", "combined"]

        ref = x_in.copy()
        ref["combined"] = [2, 4]
        trans.set_params(drop_columns=False)

        pd.testing.assert_frame_equal(trans.fit_transform(x_in), ref)

        ref["combined_2"] = [2, 4]
        trans = CombineColumns(
            function=np.sum,
            tide_format_columns="°C",
            function_kwargs={"axis": 1},
            drop_columns=False,
            result_column_name="combined_2",
        )

        pd.testing.assert_frame_equal(trans.fit_transform(x_in), ref)

    def test_pd_stl_filter(self):
        data = pd.read_csv(
            RESOURCES_PATH / "stl_processing_data.csv", index_col=0, parse_dates=True
        )
        data = data.asfreq("15min")

        # Errors :
        # "2024-08-23 01:45:00+00:00", "Temp_1"] - 0.7
        # "2024-08-26 23:45:00+00:00 ", "Temp_1"] - 0.7
        # "2024-09-08 00:45:00+00:00", "Temp_1"] - 0.7

        # "2024-09-01 12:00:00+00:00", "Temp_2"] -= 0.7
        # "2024-09-15 12:00:00+00:00", "Temp_2"] += 0.7

        filter = STLFilter(
            period="24h",
            trend="1d",
            stl_additional_kwargs={"robust": True},
            absolute_threshold=0.5,
        )

        res = filter.fit_transform(data)

        # Check that we have the right number of holes
        pd.testing.assert_series_equal(
            res.isna().sum(), pd.Series({"Temp_1": 3, "Temp_2": 2})
        )

    def test_pd_pd_interpolate(self):
        toy_df = pd.DataFrame(
            {
                "data_1": np.arange(24).astype(float),
                "data_2": 2 * np.arange(24).astype(float),
            },
            index=pd.date_range("2009", freq="h", periods=24, tz="UTC"),
        )

        toy_holes = toy_df.copy()
        toy_holes.loc["2009-01-01 02:00:00", "data_1"] = np.nan
        toy_holes.loc["2009-01-01 05:00:00":"2009-01-01 08:00:00", "data_1"] = np.nan
        toy_holes.loc["2009-01-01 12:00:00":"2009-01-01 16:00:00", "data_1"] = np.nan
        toy_holes.loc["2009-01-01 05:00:00":"2009-01-01 08:00:00", "data_2"] = np.nan

        filler = Interpolate()
        pd.testing.assert_frame_equal(toy_df, filler.fit_transform(toy_holes.copy()))

        filler = Interpolate(gaps_lte="3h", gaps_gte="5h")
        test_df = filler.fit_transform(toy_holes.copy())

        np.testing.assert_array_equal(
            test_df.to_numpy(),
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [2.0, 4.0],
                    [3.0, 6.0],
                    [4.0, 8.0],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [9.0, 18.0],
                    [10.0, 20.0],
                    [11.0, 22.0],
                    [12.0, 24.0],
                    [13.0, 26.0],
                    [14.0, 28.0],
                    [15.0, 30.0],
                    [16.0, 32.0],
                    [17.0, 34.0],
                    [18.0, 36.0],
                    [19.0, 38.0],
                    [20.0, 40.0],
                    [21.0, 42.0],
                    [22.0, 44.0],
                    [23.0, 46.0],
                ]
            ),
        )

        filler = Interpolate(gaps_lte="4h")
        test_df = filler.fit_transform(toy_holes.copy())

        np.testing.assert_array_equal(
            test_df,
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [2.0, 4.0],
                    [3.0, 6.0],
                    [4.0, 8.0],
                    [5.0, 10.0],
                    [6.0, 12.0],
                    [7.0, 14.0],
                    [8.0, 16.0],
                    [9.0, 18.0],
                    [10.0, 20.0],
                    [11.0, 22.0],
                    [np.nan, 24.0],
                    [np.nan, 26.0],
                    [np.nan, 28.0],
                    [np.nan, 30.0],
                    [np.nan, 32.0],
                    [17.0, 34.0],
                    [18.0, 36.0],
                    [19.0, 38.0],
                    [20.0, 40.0],
                    [21.0, 42.0],
                    [22.0, 44.0],
                    [23.0, 46.0],
                ]
            ),
        )

    def test_pd_fill_gap(self):
        index = pd.date_range("2009-01-01", "2009-12-31 23:00:00", freq="h", tz="UTC")
        cumsum_second = np.arange(
            start=0, stop=(index[-1] - index[0]).total_seconds() + 1, step=3600
        )
        annual = 5 * -np.cos(
            2 * np.pi / dt.timedelta(days=360).total_seconds() * cumsum_second
        )
        daily = 5 * np.sin(
            2 * np.pi / dt.timedelta(days=1).total_seconds() * cumsum_second
        )
        toy_series = pd.Series(annual + daily + 5, index=index)

        toy_df = pd.DataFrame({"Temp_1": toy_series, "Temp_2": toy_series * 1.25 + 2})

        # Diggy diggy holes !
        holes_pairs = [
            ("2009-06-14 12:00:00", "Temp_1"),
            ("2009-05-24", "Temp_1"),
            (pd.date_range("2009-07-05", "2009-07-06", freq="h", tz="UTC"), "Temp_1"),
            (
                pd.date_range(
                    "2009-12-24 14:00:00", "2009-12-24 16:00:00", freq="h", tz="UTC"
                ),
                "Temp_1",
            ),
            ("2009-04-24", "Temp_2"),
            (pd.date_range("2009-06-05", "2009-06-06", freq="h", tz="UTC"), "Temp_2"),
            (
                pd.date_range(
                    "2009-11-24 14:00:00", "2009-11-24 16:00:00", freq="h", tz="UTC"
                ),
                "Temp_2",
            ),
        ]

        toy_df_gaps = toy_df.copy()
        for gap in holes_pairs:
            toy_df_gaps.loc[gap[0], gap[1]] = np.nan

        filler = FillGapsAR(recursive_fill=False)
        res = filler.fit_transform(toy_df_gaps.copy())

        for gap in holes_pairs[1:]:
            assert r2_score(toy_df.loc[gap[0], gap[1]], res.loc[gap[0], gap[1]]) > 0.80

        filler = FillGapsAR(model_name="STL", recursive_fill=True)
        res = filler.fit_transform(toy_df_gaps.copy())

        for gap in holes_pairs[1:]:
            assert r2_score(toy_df.loc[gap[0], gap[1]], res.loc[gap[0], gap[1]]) > 0.80

        toy_df_15min = toy_df.resample("15min").mean().interpolate()
        hole_backast = pd.date_range(
            "2009-06-05", "2009-06-06 01:15:00", freq="15min", tz="UTC"
        )
        hole_forecast = pd.date_range(
            "2009-08-05", "2009-08-06 01:45:00", freq="15min", tz="UTC"
        )
        toy_df_15min_hole = toy_df_15min.copy()
        toy_df_15min_hole.loc[hole_backast, "Temp_1"] = np.nan
        toy_df_15min_hole.loc[hole_forecast, "Temp_1"] = np.nan
        toy_df_15min_hole.iloc[:12, 0] = np.nan
        toy_df_15min_hole.iloc[-12:, 0] = np.nan

        filler = FillGapsAR(resample_at_td="1h", recursive_fill=False)
        res = filler.fit_transform(toy_df_15min_hole.copy())

        assert (
            r2_score(
                res.loc[hole_backast, "Temp_1"],
                toy_df_15min.loc[hole_backast, "Temp_1"],
            )
            > 0.80
        )
        assert (
            r2_score(
                res.loc[hole_forecast, "Temp_1"],
                toy_df_15min.loc[hole_forecast, "Temp_1"],
            )
            > 0.80
        )

        filler = FillGapsAR(model_name="STL", resample_at_td="1h", recursive_fill=True)
        res = filler.fit_transform(toy_df_15min_hole.copy())

        assert (
            r2_score(
                res.loc[hole_backast, "Temp_1"],
                toy_df_15min.loc[hole_backast, "Temp_1"],
            )
            > 0.80
        )
        assert (
            r2_score(
                res.loc[hole_forecast, "Temp_1"],
                toy_df_15min.loc[hole_forecast, "Temp_1"],
            )
            > 0.80
        )

    def test_combiner(self):
        test_df = pd.DataFrame(
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

        combiner = ExpressionCombine(
            columns_dict={
                "T1": "Tin__°C__building",
                "T2": "Text__°C__outdoor",
                "m": "mass_flwr__m3/h__hvac",
            },
            expression="(T1 - T2) * m * 1004 * 1.204",
            result_column_name="loss_ventilation__J__hvac",
        )

        res = combiner.fit_transform(test_df.copy())
        assert list(combiner.get_feature_names_out()) == [
            "Tin__°C__building",
            "Text__°C__outdoor",
            "radiation__W/m2__outdoor",
            "Humidity__%HR",
            "Humidity__%HR__room1",
            "Humidity_2",
            "light__DIMENSIONLESS__building",
            "mass_flwr__m3/h__hvac",
            "loss_ventilation__J__hvac",
        ]

        np.testing.assert_almost_equal(
            res["loss_ventilation__J__hvac"],
            [3989092.8, 9066120.0, 18857529.6],
            decimal=1,
        )

        combiner.set_params(drop_columns=True)

        res = combiner.fit_transform(test_df.copy())

        assert res.shape == (3, 6)

    @patch("tide.base.get_oikolab_df", side_effect=mock_get_oikolab_df)
    def test_fill_oiko_meteo(self, mock_get_oikolab):
        data = pd.read_csv(
            RESOURCES_PATH / "meteo_fill_df.csv", parse_dates=True, index_col=0
        )

        # dig holes
        data_gap = data.copy()
        data_gap.loc[
            "2009-07-11 02:00:00":"2009-07-11 05:00:00", "text__°C__outdoor"
        ] = np.nan
        data_gap.loc["2009-07-12 18:00:00"::, "text__°C__outdoor"] = np.nan
        data_gap.loc[
            "2009-07-11 18:00:00":"2009-07-12 07:00:00", "gh__W/m²__outdoor"
        ] = np.nan

        meteo_filler = FillOikoMeteo(
            gaps_gte="4h",
            lat=-48.87667,
            lon=-123.39333,
            columns_param_map={
                "text__°C__outdoor": "temperature",
                "gh__W/m²__outdoor": "surface_solar_radiation",
                "rh__0-1__outdoor": "relative_humidity",
            },
        )

        meteo_filler.fit_transform(data_gap)

        pd.testing.assert_series_equal(
            data["gh__W/m²__outdoor"], data_gap["gh__W/m²__outdoor"]
        )
        assert float(data_gap["text__°C__outdoor"].isnull().sum()) == 13

    @patch("tide.base.get_oikolab_df", side_effect=mock_get_oikolab_df)
    def test_add_oiko_data(self, mock_get_oikolab):
        data_idx = pd.date_range(
            start="2009-07-11 16:00:00+00:00",
            end="2009-07-12 23:15:00+00:00",
            freq="15min",
        )
        data = pd.DataFrame(
            {"tin__°C__Building": np.random.randn(len(data_idx))}, index=data_idx
        )
        add_oiko = AddOikoData(lat=-48.87667, lon=-123.39333)
        res = add_oiko.fit_transform(data)
        assert not res.isnull().any().any()
        assert res.shape == (126, 13)

    def test_add_solar_angles(self):
        df = pd.DataFrame(
            {"a": np.random.randn(24)},
            index=pd.date_range("2024-12-19", freq="h", periods=24, tz="UTC"),
        )

        sun_angle = AddSolarAngles()
        sun_angle.fit(df.copy())
        assert list(sun_angle.get_feature_names_out()) == [
            "a",
            "sun_el__angle_deg__OTHER__OTHER_SUB_BLOC",
            "sun_az__angle_deg__OTHER__OTHER_SUB_BLOC",
        ]

        res = sun_angle.transform(df.copy())
        assert res.shape == (24, 3)

    def test_processing(self):
        test_df = pd.read_csv(
            RESOURCES_PATH / "solar_projection.csv", index_col=0, parse_dates=True
        )

        test_df["GHI"] = test_df["BHI"] + test_df["DHI"]

        projector = ProjectSolarRadOnSurfaces(
            bni_column_name="BNI",
            dhi_column_name="DHI",
            ghi_column_name="GHI",
            lat=44.844,
            lon=-0.564,
            surface_azimuth_angles=[180.0, 154],
            surface_tilt_angle=[90.0, 35],
            albedo=0.25,
            surface_name=["proj_180_90", "proj_tilt_35_az_154_alb_025"],
            data_bloc="PV",
            data_sub_bloc="Pyranometer",
        )

        projector.fit(test_df)
        res = projector.transform(test_df.copy())

        assert res.shape == (24, 9)

    def test_fill_other_columns(self):
        df = pd.DataFrame(
            {
                "col_1": [np.nan, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, np.nan, np.nan],
                "col_2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "col_1_fill": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            },
            index=pd.date_range("2009", freq="h", periods=10, tz="UTC"),
        )

        col_filler = FillOtherColumns(columns_map={"col_1": "col_1_fill"})
        col_filler.fit(df)
        res = col_filler.transform(df.copy())
        assert np.all(
            res["col_1"].values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        )

        col_filler = FillOtherColumns(
            gaps_lte="1h",
            columns_map={"col_1": "col_1_fill"},
            drop_filling_columns=True,
        )
        col_filler.fit(df)
        res = col_filler.transform(df.copy())
        assert res.shape == (10, 2)
        assert np.all(
            np.isnan(res["col_1"])
            == np.isnan([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, np.nan])
        )

    def test_drop_columns(self):
        df = pd.DataFrame(
            {"a": [1, 2], "b": [1, 2], "c": [1, 2]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        col_dropper = DropColumns()
        col_dropper.fit(df)
        res = col_dropper.transform(df.copy())

        pd.testing.assert_frame_equal(df, res)

        col_dropper = DropColumns(columns="a")
        col_dropper.fit(df)
        res = col_dropper.transform(df.copy())

        pd.testing.assert_frame_equal(df[["b", "c"]], res)

        col_dropper = DropColumns(columns=["a", "b", "c"])
        col_dropper.fit(df)
        res = col_dropper.transform(df.copy())

        assert res.shape == (2, 0)

    def test_replace_tag(self):
        df = pd.DataFrame(
            {"energy_1__Wh": [1.0, 2.0], "energy_2__Whr__bloc": [3.0, 4.0]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        rep = ReplaceTag({"Whr": "Wh"})
        rep.fit_transform(df)

        assert list(rep.get_feature_names_out()) == [
            "energy_1__Wh",
            "energy_2__Wh__bloc",
        ]
        assert list(df.columns) == ["energy_1__Wh", "energy_2__Wh__bloc"]
