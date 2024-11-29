import datetime as dt
import pandas as pd

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array
from statsmodels.tsa.seasonal import STL

from tide.utils import check_and_return_dt_index_df
from tide.base import BaseSTL


class STLEDetector(ClassifierMixin, BaseSTL):
    """
    A custom anomaly detection model based on statsmodel STL
    (Seasonal and Trend decomposition using Loess).

    The STL decomposition breaks down time series into three components: trend,
    seasonal, and residual. This class uses the residual component to detect anomalies
    based on the absolute threshold (absolute value of residual exceed threshold).

    See statsmodel doc for additional STL configuration.
    (https://www.statsmodels.org/stable/index.html)


    Parameters
    ----------
    period : int | str | dt.timedelta
        The period of the time series (e.g., daily, weekly, monthly, etc.).
        Can be an integer, string, or timedelta.
        This defines the seasonal periodicity for the STL decomposition.

    absolute_threshold : int | float
        The threshold value for residuals. Any residuals exceeding this threshold
        are considered anomalies.

    trend : int | str | dt.timedelta, optional
        The length of the trend smoother. Must be odd and larger than season
        Statsplot indicate it is usually around 150% of season.
        Strongly depends on your time series.

    seasonal : int | str | dt.timedelta, optional
        The seasonal component's smoothing parameter for STL. It defines how much
        the seasonal component is smoothed. If given as an integer,
        it must be an odd number. If None, a default value will be used.

    stl_kwargs : dict[str, float], optional
        Additional keyword arguments for the STL decomposition.
        These allow fine-tuning of the decomposition process.
        (https://www.statsmodels.org/stable/index.html)


    Attributes
    ----------

    labels_ : pd.DataFrame
        A DataFrame with binary labels (0 or 1), indicating whether an anomaly
        is detected (1) or not (0).

    stl_res : dict
        A dictionary that holds the fitted STL results for each feature in the dataset.

    Methods
    -------
    __sklearn_is_fitted__():
        Checks whether the model has been fitted and returns a boolean
        indicating the fitted status.

    fit(X: pd.Series | pd.DataFrame):
        Fits the STL model to the input time series data. Computes and stores
        residuals for each column in X.

    predict(X: pd.Series | pd.DataFrame):
        Fits the model and predicts anomalies by comparing the residuals with
        the absolute threshold. Returns a 0-1 Pandas DataFrame

    Raises
    ------
    ValueError
        If the seasonal parameter is an even number when passed as an integer.

    """

    def __init__(
        self,
        period: int | str | dt.timedelta = "24h",
        trend: int | str | dt.timedelta = "15d",
        absolute_threshold: int | float = 100,
        seasonal: int | str | dt.timedelta = None,
        stl_kwargs: dict[str, float] = None,
    ):
        super().__init__(period, trend, seasonal, stl_kwargs)
        self.absolute_threshold = absolute_threshold

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        self._pre_fit(X)
        self.stl_fit_res_ = {}
        for feat in X.columns:
            self.stl_fit_res_[feat] = STL(X[feat], **self.stl_kwargs).fit()

        return self

    def predict(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["stl_fit_res_"])
        X = check_and_return_dt_index_df(X)
        if isinstance(X, pd.Series):
            X = X.to_frame()
        check_array(X)

        res_df = pd.concat([res.resid for res in self.stl_fit_res_.values()], axis=1)
        res_df.columns = X.columns
        return (abs(res_df) > self.absolute_threshold).astype(int)
