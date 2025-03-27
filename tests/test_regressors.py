import datetime as dt

import pandas as pd
import numpy as np
import pytest

from tide.regressors import SkSTLForecast, SkProphet


class TestRegressors:
    @pytest.fixture
    def toy_data(self):
        index = pd.date_range("2009-01-01", "2009-12-31 23:00:00", freq="h", tz="UTC")
        cumsum_second = np.arange(
            0, (index[-1] - index[0]).total_seconds() + 1, step=3600
        )

        annual = 5 * -np.cos(
            2 * np.pi / dt.timedelta(days=360).total_seconds() * cumsum_second
        )
        daily = 5 * np.sin(
            2 * np.pi / dt.timedelta(days=1).total_seconds() * cumsum_second
        )

        toy_series = pd.Series(annual + daily + 5, index=index)

        exo = 12 + 3 * np.arange(index.shape[0])

        toy_df = pd.DataFrame(
            {
                "Temp_1__°C": toy_series,
                "Temp_2__°C": toy_series * 1.25 + 2,
                "Temp_3__°C": toy_series + exo,
                "Exo": exo,
            }
        )
        return toy_df

    def test_stl_forecaster(self, toy_data):
        forecaster = SkSTLForecast(
            period="24h",
            trend="15d",
            ar_kwargs=dict(order=(1, 1, 0), trend="t"),
            backcast=False,
        )

        forecaster.fit(
            X=toy_data["2009-01-24":"2009-07-24"].index,
            y=toy_data.loc["2009-01-24":"2009-07-24", ["Temp_1__°C", "Temp_2__°C"]],
        )
        reg_score = forecaster.score(
            toy_data["2009-07-27":"2009-07-30"],
            toy_data.loc["2009-07-27":"2009-07-30", ["Temp_1__°C", "Temp_2__°C"]],
        )
        assert reg_score > 0.99

        backcaster = SkSTLForecast(backcast=True)

        backcaster.fit(
            X=toy_data.loc["2009-01-24":"2009-07-24"],
            y=toy_data.loc["2009-01-24":"2009-07-24", ["Temp_1__°C", "Temp_2__°C"]],
        )

        reg_score = backcaster.score(
            toy_data["2009-01-20":"2009-01-22"].index,
            toy_data.loc["2009-01-20":"2009-01-22", ["Temp_1__°C", "Temp_2__°C"]],
        )

        assert reg_score > 0.99

    def test_prophet_forecaster(self, toy_data):
        forecaster = SkProphet()
        forecaster.fit(
            X=toy_data["2009-01-24":"2009-07-24"].index,
            y=toy_data.loc["2009-01-24":"2009-07-24", ["Temp_1__°C", "Temp_2__°C"]],
        )

        reg_score = forecaster.score(
            toy_data["2009-07-27":"2009-07-30"].index,
            toy_data.loc["2009-07-27":"2009-07-30", ["Temp_1__°C", "Temp_2__°C"]],
        )
        assert reg_score > 0.99

        reg_score = forecaster.score(
            toy_data["2009-01-20":"2009-01-22"].index,
            toy_data.loc["2009-01-20":"2009-01-22", ["Temp_1__°C", "Temp_2__°C"]],
        )

        assert reg_score > 0.99

        forecaster = SkProphet(return_upper_lower_bounds=True)
        forecaster.fit(
            X=toy_data["2009-01-24":"2009-07-24"].index,
            y=toy_data.loc["2009-01-24":"2009-07-24", ["Temp_1__°C", "Temp_2__°C"]],
        )

        feat_out = list(forecaster.get_feature_names_out())
        predictions = forecaster.predict(toy_data["2009-07-27":"2009-07-30"].index)

        assert np.all([feat in predictions.columns for feat in feat_out])

        forecaster = SkProphet()
        forecaster.fit(
            X=toy_data.loc["2009-01-24":"2009-07-24", "Exo"],
            y=toy_data.loc["2009-01-24":"2009-07-24", "Temp_3__°C"],
        )

        reg_score = forecaster.score(
            toy_data.loc["2009-01-24":"2009-07-24", "Exo"],
            toy_data.loc["2009-01-24":"2009-07-24", "Temp_3__°C"],
        )

        assert reg_score > 0.99
        assert forecaster.get_feature_names_in() == ["Exo"]
