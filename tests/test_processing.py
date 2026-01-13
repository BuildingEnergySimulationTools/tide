import datetime as dt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

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
    KeepColumns,
    ReplaceTag,
    AddFourierPairs,
    DropQuantile,
    TrimSequence,
)

RESOURCES_PATH = Path(__file__).parent / "resources"


def check_feature_names_out(processor, res_df):
    assert processor.get_feature_names_out() == list(res_df.columns)


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

        pd.testing.assert_frame_equal(df, res)
        check_feature_names_out(identity, res)

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
        check_feature_names_out(rep_dup, res_dup)

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
        res = dropper.transform(df)
        pd.testing.assert_frame_equal(res, ref)
        check_feature_names_out(dropper, res)

    def test_pd_rename_columns(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0]},
            index=pd.date_range("2009", freq="h", periods=3, tz="UTC"),
        )

        new_cols = ["c", "d"]
        renamer = RenameColumns(new_names=new_cols)
        renamer.fit(df)
        res = renamer.fit_transform(df.copy())
        assert list(df.columns) == list(renamer.get_feature_names_in())
        check_feature_names_out(renamer, res)

        new_cols_dict = {"d": "a"}
        renamer = RenameColumns(new_names=new_cols_dict)
        res = renamer.fit_transform(res)
        assert list(res.columns) == ["c", "a"]
        check_feature_names_out(renamer, res)

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
        check_feature_names_out(scaler, ref)
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
        res = dropper.transform(df)
        pd.testing.assert_frame_equal(res, ref)
        check_feature_names_out(dropper, res)

        # check do nothing
        dropper = ReplaceThreshold()
        res = dropper.fit_transform(df)
        pd.testing.assert_frame_equal(res, ref)
        check_feature_names_out(dropper, res)

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
        res = dropper.fit_transform(df.copy())
        pd.testing.assert_frame_equal(ref, ref)
        check_feature_names_out(dropper, res)

        # check do nothing
        dropper = DropTimeGradient()
        res = dropper.fit_transform(df.copy())
        pd.testing.assert_frame_equal(res, df)
        check_feature_names_out(dropper, res)

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
        res = transformer.fit_transform(df)
        pd.testing.assert_frame_equal(ref, res)
        check_feature_names_out(transformer, res)

        df = pd.DataFrame(
            {"a__W": [1.0, 2.0], "b__W": [3.0, 4.0]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        ref = pd.DataFrame(
            {"a__kW": [0.001, 0.002], "b__kW": [0.003, 0.004]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        transformer = ApplyExpression("X / 1000", "kW")
        res = transformer.fit_transform(df)
        pd.testing.assert_frame_equal(ref, res)
        check_feature_names_out(transformer, res)

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
        res = derivator.fit_transform(test)
        pd.testing.assert_frame_equal(ref, res, rtol=0.01)
        check_feature_names_out(derivator, res)

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
        res = filler.fit_transform(test)
        pd.testing.assert_frame_equal(ref, res)
        check_feature_names_out(filler, res)

        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 0.0, 2.0, 2.0, np.nan, np.nan],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 2.0, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )

        filler = Ffill(gaps_lte="1h")
        res = filler.fit_transform(test)
        pd.testing.assert_frame_equal(ref, res)
        check_feature_names_out(filler, res)

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
        res = filler.fit_transform(test)
        pd.testing.assert_frame_equal(ref, res)
        check_feature_names_out(filler, res)

        filler = Bfill(gaps_lte="1h")
        ref = pd.DataFrame(
            {
                "cpt1": [np.nan, np.nan, 2.0, 2.0, 3.0, 3.0],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )
        res = filler.fit_transform(test)
        pd.testing.assert_frame_equal(ref, res)
        check_feature_names_out(filler, res)

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
        res = filler.fit_transform(test)
        pd.testing.assert_frame_equal(ref, res)
        check_feature_names_out(filler, res)

        filler = FillNa(value=0.0, gaps_lte="1h")
        ref = pd.DataFrame(
            {
                "cpt1": [0.0, 0.0, 2.0, 2.0, np.nan, np.nan],
                "cpt2": [0.0, 1.0, 2.0, 2.0, 0.0, 3.0],
            },
            index=pd.date_range("2009", freq="h", periods=6, tz="UTC"),
        )
        res = filler.fit_transform(test)
        pd.testing.assert_frame_equal(ref, res)
        check_feature_names_out(filler, res)

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

        res = column_resampler.fit_transform(df)
        pd.testing.assert_frame_equal(ref, res.astype("float"), atol=0.01)
        check_feature_names_out(column_resampler, res)

        column_resampler = Resample(
            rule="5h",
            method="max",
            tide_format_methods={"°C": "mean"},
        )

        res = column_resampler.fit_transform(df)
        pd.testing.assert_frame_equal(ref, res.astype("float"), atol=0.01)
        check_feature_names_out(column_resampler, res)

        column_resampler = Resample(
            rule="5h",
            method="max",
        )
        res = column_resampler.fit_transform(df)
        np.testing.assert_almost_equal(
            res.to_numpy(),
            np.array(
                [
                    [4.00000000e02, 4.00000000e00, 9.50714306e-01, 9.69909852e00],
                    [9.00000000e02, 9.00000000e00, 8.66176146e-01, 5.24756432e00],
                ]
            ),
            decimal=1,
        )
        check_feature_names_out(column_resampler, res)

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
        res = lager.fit_transform(df)
        pd.testing.assert_frame_equal(ref, res)
        check_feature_names_out(lager, res)

    def test_pd_gaussian_filter(self):
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            index=pd.date_range("2009", freq="h", periods=3, tz="UTC"),
        )

        gfilter = GaussianFilter1D()

        res = gfilter.fit_transform(df)
        check_feature_names_out(gfilter, res)
        np.testing.assert_almost_equal(
            gaussian_filter1d(
                df.to_numpy()[:, 0].T, sigma=5, mode="nearest", truncate=4.0
            ),
            res.to_numpy()[:, 0],
            decimal=5,
        )

    def test_pd_combine_columns(self):
        x_in = pd.DataFrame(
            {"a__°C": [1, 2], "b__°C": [2, 4]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        trans = CombineColumns(
            function="sum",
            drop_columns=True,
        )

        res = trans.fit_transform(x_in.copy())
        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {"combined": [3, 6]},
                index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
            ),
        )

        trans = CombineColumns(
            function="mean",
            drop_columns=True,
        )

        res = trans.fit_transform(x_in.copy())
        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {"combined": [1.5, 3]},
                index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
            ),
        )

        check_feature_names_out(trans, res)

        ref = x_in.copy()
        ref["combined"] = [1.8, 3.6]
        trans.set_params(function="average", weights=[1, 4], drop_columns=False)
        res = trans.fit_transform(x_in.copy())
        pd.testing.assert_frame_equal(res, ref)
        check_feature_names_out(trans, res)

        ref = x_in.copy()
        ref["combined_2"] = [5, 10]
        trans = CombineColumns(
            function="dot",
            weights=[1, 2],
            drop_columns=False,
            result_column_name="combined_2",
        )

        res = trans.fit_transform(x_in.copy())
        pd.testing.assert_frame_equal(res, ref)
        check_feature_names_out(trans, res)

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
        pd.testing.assert_series_equal(
            res.isna().sum(), pd.Series({"Temp_1": 3, "Temp_2": 2})
        )
        check_feature_names_out(filter, res)

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

        # filler = Interpolate()
        # pd.testing.assert_frame_equal(toy_df, filler.fit_transform(toy_holes.copy()))

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
        check_feature_names_out(filler, test_df)

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
        check_feature_names_out(filler, test_df)

        filler = Interpolate(gaps_gte="2h", gaps_lte="4h30min")
        test_df = filler.fit_transform(toy_holes.copy())
        np.testing.assert_array_equal(
            test_df,
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [np.nan, 4.0],
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
        check_feature_names_out(filler, test_df)

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
        check_feature_names_out(filler, res)

        for gap in holes_pairs[1:]:
            assert r2_score(toy_df.loc[gap[0], gap[1]], res.loc[gap[0], gap[1]]) > 0.80

        filler = FillGapsAR(model_name="STL", recursive_fill=True)
        res = filler.fit_transform(toy_df_gaps.copy())
        check_feature_names_out(filler, res)

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
        check_feature_names_out(filler, res)

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
                "Tin__°C__building__room_1": [10.0, 20.0, 30.0],
                "Tin__°C__building__room_2": [20.0, 40.0, 60.0],
                "Text__°C__outdoor__meteo": [-1.0, 5.0, 4.0],
                "radiation__W/m2__outdoor__meteo": [50, 100, 400],
                "Humidity__%HR__building__room_1": [10, 15, 13],
                "Humidity__%HR__building__room_2": [20, 30, 50],
                "Humidity__%HR__outdoor__meteo": [10, 15, 13],
                "light__DIMENSIONLESS__building__room_1": [100, 200, 300],
                "mass_flwr__m3/h__hvac__pump": [300, 500, 600],
            },
            index=pd.date_range("2009", freq="h", periods=3, tz="UTC"),
        )

        combiner = ExpressionCombine(
            columns_dict={
                "T1": "Tin__°C__building__room_1",
                "T2": "Text__°C__outdoor__meteo",
                "m": "mass_flwr__m3/h__hvac__pump",
            },
            expression="(T1 - T2) * m * 1004 * 1.204",
            result_column_name="loss_ventilation__J__building__room_1",
        )

        res = combiner.fit_transform(test_df.copy())
        check_feature_names_out(combiner, res)
        np.testing.assert_almost_equal(
            res["loss_ventilation__J__building__room_1"],
            [3989092.8, 9066120.0, 18857529.6],
            decimal=1,
        )

        combiner.set_params(drop_columns=True)
        res = combiner.fit_transform(test_df.copy())
        assert res.shape == (3, 7)
        check_feature_names_out(combiner, res)

        combiner_cond = ExpressionCombine(
            columns_dict={
                "T1": "Text__°C__outdoor__meteo",
            },
            expression="(T1 > 10) * 1",
            result_column_name="where_test_01__meteo__outdoor",
        )

        res = combiner_cond.fit_transform(test_df.copy())
        check_feature_names_out(combiner_cond, res)
        np.testing.assert_almost_equal(
            res["where_test_01__meteo__outdoor"],
            [0, 0, 0],
            decimal=1,
        )

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

        res = meteo_filler.fit_transform(data_gap)
        pd.testing.assert_series_equal(
            data["gh__W/m²__outdoor"], data_gap["gh__W/m²__outdoor"]
        )
        assert float(data_gap["text__°C__outdoor"].isnull().sum()) == 13
        check_feature_names_out(meteo_filler, res)

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
        check_feature_names_out(add_oiko, res)

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
        check_feature_names_out(sun_angle, res)

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
        check_feature_names_out(projector, res)

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
        check_feature_names_out(col_filler, res)

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
        check_feature_names_out(col_filler, res)

    def test_drop_columns(self):
        df = pd.DataFrame(
            {"a": [1, 2], "b": [1, 2], "c": [1, 2]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        col_dropper = DropColumns()
        col_dropper.fit(df)
        res = col_dropper.transform(df.copy())
        assert res.shape == (2, 0)
        check_feature_names_out(col_dropper, pd.DataFrame(index=df.index))

        col_dropper = DropColumns(columns="a")
        col_dropper.fit(df)
        res = col_dropper.transform(df.copy())
        pd.testing.assert_frame_equal(df[["b", "c"]], res)
        check_feature_names_out(col_dropper, res)

        col_dropper = DropColumns(columns=["a|b", "c"])
        col_dropper.fit(df)
        res = col_dropper.transform(df.copy())
        assert res.shape == (2, 0)
        check_feature_names_out(col_dropper, res)

    def test_keep_columns(self):
        df = pd.DataFrame(
            {"a": [1, 2], "b": [1, 2], "c": [1, 2]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        col_keeper = KeepColumns()
        col_keeper.fit(df)
        res = col_keeper.transform(df.copy())
        pd.testing.assert_frame_equal(df, res)
        check_feature_names_out(col_keeper, res)

        col_keeper = KeepColumns(columns="a")
        col_keeper.fit(df)
        res = col_keeper.transform(df.copy())
        pd.testing.assert_frame_equal(df[["a"]], res)
        check_feature_names_out(col_keeper, res)

        col_keeper = KeepColumns(columns=["a|b", "c"])
        col_keeper.fit(df)
        res = col_keeper.transform(df.copy())
        pd.testing.assert_frame_equal(df, res)
        check_feature_names_out(col_keeper, res)

    def test_replace_tag(self):
        df = pd.DataFrame(
            {"energy_1__Wh": [1.0, 2.0], "energy_2__Whr__bloc": [3.0, 4.0]},
            index=pd.date_range("2009", freq="h", periods=2, tz="UTC"),
        )

        rep = ReplaceTag({"Whr": "Wh"})
        res = rep.fit_transform(df)
        check_feature_names_out(rep, res)

    def test_add_fourier_pairs(self):
        test_df = pd.DataFrame(
            data=np.arange(24).astype("float64"),
            index=pd.date_range("2009-01-01 00:00:00", freq="h", periods=24, tz="UTC"),
            columns=["feat_1"],
        )

        signal = AddFourierPairs(period="24h", order=2)
        res = signal.fit_transform(test_df)

        ref_df = pd.DataFrame(
            data=np.array(
                [
                    [0.0, 0.000000e00, 1.000000e00, 0.000000e00, 1.000000e00],
                    [1.0, 2.588190e-01, 9.659258e-01, 5.000000e-01, 8.660254e-01],
                    [2.0, 5.000000e-01, 8.660254e-01, 8.660254e-01, 5.000000e-01],
                    [3.0, 7.071068e-01, 7.071068e-01, 1.000000e00, 6.123234e-17],
                    [4.0, 8.660254e-01, 5.000000e-01, 8.660254e-01, -5.000000e-01],
                    [5.0, 9.659258e-01, 2.588190e-01, 5.000000e-01, -8.660254e-01],
                    [6.0, 1.000000e00, 6.123234e-17, 1.224647e-16, -1.000000e00],
                    [7.0, 9.659258e-01, -2.588190e-01, -5.000000e-01, -8.660254e-01],
                    [8.0, 8.660254e-01, -5.000000e-01, -8.660254e-01, -5.000000e-01],
                    [9.0, 7.071068e-01, -7.071068e-01, -1.000000e00, -1.836970e-16],
                    [10.0, 5.000000e-01, -8.660254e-01, -8.660254e-01, 5.000000e-01],
                    [11.0, 2.588190e-01, -9.659258e-01, -5.000000e-01, 8.660254e-01],
                    [12.0, 1.224647e-16, -1.000000e00, -2.449294e-16, 1.000000e00],
                    [13.0, -2.588190e-01, -9.659258e-01, 5.000000e-01, 8.660254e-01],
                    [14.0, -5.000000e-01, -8.660254e-01, 8.660254e-01, 5.000000e-01],
                    [15.0, -7.071068e-01, -7.071068e-01, 1.000000e00, 3.061617e-16],
                    [16.0, -8.660254e-01, -5.000000e-01, 8.660254e-01, -5.000000e-01],
                    [17.0, -9.659258e-01, -2.588190e-01, 5.000000e-01, -8.660254e-01],
                    [18.0, -1.000000e00, -1.836970e-16, 3.673940e-16, -1.000000e00],
                    [19.0, -9.659258e-01, 2.588190e-01, -5.000000e-01, -8.660254e-01],
                    [20.0, -8.660254e-01, 5.000000e-01, -8.660254e-01, -5.000000e-01],
                    [21.0, -7.071068e-01, 7.071068e-01, -1.000000e00, -4.286264e-16],
                    [22.0, -5.000000e-01, 8.660254e-01, -8.660254e-01, 5.000000e-01],
                    [23.0, -2.588190e-01, 9.659258e-01, -5.000000e-01, 8.660254e-01],
                ]
            ),
            columns=[
                "feat_1",
                "1 days 00:00:00_order_1_Sine",
                "1 days 00:00:00_order_1_Cosine",
                "1 days 00:00:00_order_2_Sine",
                "1 days 00:00:00_order_2_Cosine",
            ],
            index=pd.date_range("2009-01-01 00:00:00", freq="h", periods=24, tz="UTC"),
        )

        pd.testing.assert_frame_equal(res, ref_df)

        test_df_phi = pd.DataFrame(
            data=np.arange(24),
            index=pd.date_range("2009-01-01 06:00:00", freq="h", periods=24),
            columns=["feat_1"],
        )
        test_df_phi = test_df_phi.tz_localize("UTC")
        res = signal.transform(test_df_phi)

        assert res.iloc[0, 1] == 1.0

        test_df = pd.DataFrame(
            data=np.arange(24).astype("float64"),
            index=pd.date_range("2009-01-01 00:00:00", freq="h", periods=24, tz="UTC"),
            columns=["feat_1__°C__building__room"],
        )

        signal = AddFourierPairs(period=dt.timedelta(hours=24), unit="W")
        signal.fit_transform(test_df)

        assert signal.feature_names_out_ == [
            "feat_1__°C__building__room",
            "1 day, 0:00:00_order_1_Sine__W__BLOCK__SUB_BLOCK",
            "1 day, 0:00:00_order_1_Cosine__W__BLOCK__SUB_BLOCK",
        ]

    def test_drop_quantile(self):
        index = pd.date_range(
            "2009-01-01", "2009-01-01 23:00:00", freq="15min", tz="UTC"
        )
        cumsum_second = np.arange(
            start=0, stop=(index[-1] - index[0]).total_seconds() + 1, step=15 * 60
        )

        daily = 5 * np.sin(
            2 * np.pi / dt.timedelta(days=1).total_seconds() * cumsum_second
        )

        twice_daily = 5 * np.sin(
            2 * np.pi / dt.timedelta(hours=12).total_seconds() * cumsum_second
        )

        rng = np.random.default_rng(42)

        toy_df = pd.DataFrame(
            {
                "Temp_1": daily + rng.standard_normal(daily.shape[0]),
                "Temp_2": twice_daily + 2 * rng.standard_normal(twice_daily.shape[0]),
            },
            index=index,
        )

        dropper = DropQuantile(
            upper_quantile=0.75,
            lower_quantile=0.25,
            n_iqr=1.5,
            detrend_method="Gaussian",
        )

        quant_filtered = dropper.fit_transform(toy_df)

        ref_temp_1 = pd.Series(
            [np.nan],
            pd.DatetimeIndex(
                [pd.Timestamp("2009-01-01 07:30:00+00:00", tz="UTC")], freq="15min"
            ),
            name="Temp_1",
        )

        ref_temp_2 = pd.Series(
            [np.nan],
            pd.DatetimeIndex(
                [pd.Timestamp("2009-01-01 11:30:00+0000", tz="UTC")], freq="15min"
            ),
            name="Temp_2",
        )

        pd.testing.assert_series_equal(
            quant_filtered["Temp_1"][quant_filtered["Temp_1"].isna()],
            ref_temp_1,
        )

        pd.testing.assert_series_equal(
            quant_filtered["Temp_2"][quant_filtered["Temp_2"].isna()],
            ref_temp_2,
        )

        # No filtering method
        toy_df = pd.DataFrame(
            {
                "Noise_1": rng.standard_normal(daily.shape[0]),
                "Noise_2": rng.standard_normal(twice_daily.shape[0]),
            },
            index=index,
        )

        dropper = DropQuantile(upper_quantile=0.75, lower_quantile=0.25, n_iqr=1.5)

        filtered_noise = dropper.fit_transform(toy_df)

        pd.testing.assert_index_equal(
            filtered_noise["Noise_1"][filtered_noise["Noise_1"].isna()].index,
            pd.DatetimeIndex(
                ["2009-01-01 10:00:00+00:00", "2009-01-01 15:30:00+00:00"],
                dtype="datetime64[ns, UTC]",
                freq=None,
            ),
        )

        pd.testing.assert_index_equal(
            filtered_noise["Noise_2"][filtered_noise["Noise_2"].isna()].index,
            pd.DatetimeIndex(
                ["2009-01-01 03:30:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
            ),
        )

        # Linear trend

        index = pd.date_range(
            "2009-01-01", "2009-01-01 23:00:00", freq="15min", tz="UTC"
        )

        rng = np.random.default_rng(42)

        toy_df = pd.DataFrame(
            {
                "Temp_1": np.linspace(0, 10, len(index))
                + rng.standard_normal(len(index)),
                "Temp_2": np.linspace(5, 10, len(index))
                + rng.standard_normal(len(index)),
            },
            index=index,
        )

        filter_detrend = DropQuantile(
            upper_quantile=0.75,
            lower_quantile=0.25,
            n_iqr=1.5,
            detrend_method="Detrend",
        )
        filtered = filter_detrend.fit_transform(toy_df)

        pd.testing.assert_index_equal(
            filtered["Temp_2"][filtered["Temp_2"].isna()].index,
            pd.DatetimeIndex(
                ["2009-01-01 11:30:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
            ),
        )

    def test_sequence_trim(self):
        df = pd.DataFrame(
            {
                "a": [np.nan, 1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, np.nan, 9.0],
                "b": [0.0, 10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0, 80.0, 90.0],
            },
            index=pd.date_range("2009-01-01", freq="min", periods=10, tz="UTC"),
        )

        trimer = TrimSequence()
        res = trimer.fit_transform(df)

        pd.testing.assert_frame_equal(res, df)

        trimer = TrimSequence("1min")
        res = trimer.fit_transform(df)

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    "a": [
                        np.nan,
                        np.nan,
                        2.0,
                        3.0,
                        4.0,
                        np.nan,
                        np.nan,
                        7.0,
                        np.nan,
                        np.nan,
                    ],
                    "b": [
                        np.nan,
                        10.0,
                        20.0,
                        30.0,
                        np.nan,
                        np.nan,
                        60.0,
                        70.0,
                        80.0,
                        90.0,
                    ],
                },
                index=pd.date_range("2009-01-01", freq="min", periods=10, tz="UTC"),
            ),
        )

        check_feature_names_out(trimer, res)
