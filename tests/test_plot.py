import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tide.plot import (
    plot_gaps_heatmap,
    add_multi_axis_scatter,
    get_cols_axis_maps_and_labels,
    get_gap_scatter_dict,
    get_yaxis_min_max,
)

import plotly.io as pio

pio.renderers.default = "browser"


class TestPlot:
    def test_get_cols_to_axis_maps_and_labels(self):
        columns = ["a__°C__zone1", "b__°C__zone2", "c__Wh__zone1"]
        assert get_cols_axis_maps_and_labels(columns) == get_cols_axis_maps_and_labels(
            columns, y_tag_list=["°C", "Wh"]
        )

        assert get_cols_axis_maps_and_labels(columns, "name") == (
            {
                "a__°C__zone1": {"yaxis": "y"},
                "c__Wh__zone1": {"yaxis": "y2"},
                "b__°C__zone2": {"yaxis": "y3"},
            },
            {"y": ["a__°C__zone1"], "y2": ["c__Wh__zone1"], "y3": ["b__°C__zone2"]},
            ["a", "c", "b"],
        )

        columns = ["a", "b", "c"]
        assert get_cols_axis_maps_and_labels(columns) == (
            {"a": {"yaxis": "y"}, "b": {"yaxis": "y"}, "c": {"yaxis": "y"}},
            {"y": ["a", "b", "c"]},
            ["a", "b", "c"],
        )

        columns = pd.Index(["a", "b", "c"])
        assert get_cols_axis_maps_and_labels(columns) == (
            {"a": {"yaxis": "y"}, "b": {"yaxis": "y"}, "c": {"yaxis": "y"}},
            {"y": ["a", "b", "c"]},
            ["a", "b", "c"],
        )

    def test_plot_gaps_heatmap(self):
        df = pd.DataFrame(
            {
                "a": np.random.randn(24),
                "b": np.random.randn(24),
            },
            index=pd.date_range("2009", freq="h", periods=24, tz="UTC"),
        )

        df.loc["2009-01-01 05:00:00":"2009-01-01 09:00:00", :] = np.nan
        df.loc["2009-01-01 15:00:00", "a"] = np.nan
        df.loc["2009-01-01 20:00:00", "b"] = np.nan

        plot_gaps_heatmap(df, "3h")

        assert True

    def test_add_multi_axis_scatter(self):
        df = pd.DataFrame(
            {
                "a__°C": np.random.randn(24),
                "b__°C": np.random.randn(24),
                "b__W": np.random.randn(24) * 100,
                "e__Wh": np.random.randn(24) * 100,
            },
            index=pd.date_range("2009", freq="h", periods=24, tz="UTC"),
        )
        df["e__Wh"] = abs(df).cumsum()["e__Wh"]

        fig = go.Figure()
        fig = add_multi_axis_scatter(fig, df)

        fig = go.Figure()
        fig = add_multi_axis_scatter(
            fig,
            df,
            y_axis_dict={"a__°C": "y", "b__°C": "y", "b__W": "y2", "e__Wh": "y3"},
            y_axis_labels=["y1", "y2", "y3"],
            axis_space=0.04,
            mode_dict={
                "a__°C": "markers",
                "b__°C": "markers",
                "b__W": "lines",
                "e__Wh": "lines+markers",
            },
            y_title_standoff=1,
        )

    def test_get_gaps_scatter_dict(self):
        np.random.seed(42)
        measure = pd.Series(
            np.random.randn(24),
            name="name",
            index=pd.date_range("2009", freq="h", periods=24, tz="UTC"),
        )

        measure.loc["2009-01-01 02:00:00":"2009-01-01 05:00:00"] = np.nan
        measure.loc["2009-01-01 12:00:00"] = np.nan

        col_axes_map, axes_col_map, _ = get_cols_axis_maps_and_labels(["name"])
        min_max = get_yaxis_min_max(measure)
        gap_dict = get_gap_scatter_dict(measure, min_max, col_axes_map)

        assert gap_dict == [
            {
                "x": [
                    pd.Timestamp("2009-01-01 01:00:00", tz="UTC"),
                    pd.Timestamp("2009-01-01 01:00:00", tz="UTC"),
                    pd.Timestamp("2009-01-01 06:00:00", tz="UTC"),
                    pd.Timestamp("2009-01-01 06:00:00", tz="UTC"),
                ],
                "y": [
                    -1.913280244657798,
                    1.5792128155073915,
                    1.5792128155073915,
                    -1.913280244657798,
                ],
                "mode": "none",
                "fill": "toself",
                "showlegend": False,
                "fillcolor": "rgba(102, 102, 102, 0.5)",
                "yaxis": "y",
            },
            {
                "x": [
                    pd.Timestamp("2009-01-01 11:00:00", tz="UTC"),
                    pd.Timestamp("2009-01-01 11:00:00", tz="UTC"),
                    pd.Timestamp("2009-01-01 13:00:00", tz="UTC"),
                    pd.Timestamp("2009-01-01 13:00:00", tz="UTC"),
                ],
                "y": [
                    -1.913280244657798,
                    1.5792128155073915,
                    1.5792128155073915,
                    -1.913280244657798,
                ],
                "mode": "none",
                "fill": "toself",
                "showlegend": False,
                "fillcolor": "rgba(102, 102, 102, 0.5)",
                "yaxis": "y",
            },
        ]
