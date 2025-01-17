import datetime as dt

import pandas as pd
import numpy as np

from tide.regressors import SkSTLForecast


class TestRegressors:
    def test_stl_forecaster(self):
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

        forecaster = SkSTLForecast(
            period="24h",
            trend="15d",
            ar_kwargs=dict(order=(1, 1, 0), trend="t"),
            backcast=False,
        )

        forecaster.fit(toy_df["2009-01-24":"2009-07-24"])

        reg_score = forecaster.score(
            toy_df["2009-07-27":"2009-07-30"], toy_df["2009-07-27":"2009-07-30"]
        )
        assert reg_score > 0.99

        backcaster = SkSTLForecast(backcast=True)

        backcaster.fit(toy_df["2009-01-24":"2009-07-24"])

        reg_score = backcaster.score(
            toy_df["2009-01-20":"2009-01-22"], toy_df["2009-01-20":"2009-01-22"]
        )
        assert reg_score > 0.99
