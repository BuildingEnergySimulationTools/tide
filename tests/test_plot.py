import numpy as np
import pandas as pd

from tide.plot import plot_gaps_heatmap, plot

import plotly.io as pio

pio.renderers.default = "browser"


class TestPlot:
    def test_get_cols_to_axis_maps(self):
        columns = ["a__°C__zone1", "b__°C__zone2", "c__Wh__zone1"]
        assert get_cols_to_axis_maps(columns) == get_cols_to_axis_maps(
            columns, y_tag_list=["°C", "Wh"]
        )

        assert get_cols_to_axis_maps(columns, "name") == (
            {
                "a__°C__zone1": {"yaxis": "y"},
                "c__Wh__zone1": {"yaxis": "y2"},
                "b__°C__zone2": {"yaxis": "y3"},
            },
            {"y": ["a__°C__zone1"], "y2": ["c__Wh__zone1"], "y3": ["b__°C__zone2"]},
        )

        columns = ["a", "b", "c"]
        assert get_cols_to_axis_maps(columns) == (
            {"a": {"yaxis": "y"}, "b": {"yaxis": "y"}, "c": {"yaxis": "y"}},
            {"y": ["a", "b", "c"]},
        )

    def test_plot_gaps_heatmap(self):
        df = pd.DataFrame(
            {
                "a": np.random.randn(24),
                "b": np.random.randn(24),
            },
            index=pd.date_range("2009", freq="h", periods=24),
        )

        df.loc["2009-01-01 05:00:00":"2009-01-01 09:00:00", :] = np.nan
        df.loc["2009-01-01 15:00:00", "a"] = np.nan
        df.loc["2009-01-01 20:00:00", "b"] = np.nan

        fig = plot_gaps_heatmap(df, "3h")

        assert True

    def test_plot(self):
        df = pd.DataFrame(
            {
                "a__°C": np.random.randn(24),
                "b__°C": np.random.randn(24),
                "b__W": np.random.randn(24) * 100,
                "e__Wh": np.random.randn(24) * 100,
            },
            index=pd.date_range("2009", freq="h", periods=24),
        )
        df["e__Wh"] = abs(df).cumsum()["e__Wh"]

        fig = plot(df)

        fig = plot(
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

        assert True
