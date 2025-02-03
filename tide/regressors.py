import datetime as dt
import warnings

import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from prophet import Prophet

from tide.base import BaseSTL, TideBaseMixin
from tide.utils import check_and_return_dt_index_df

MODEL_MAP = {"ARIMA": ARIMA}
MODEL_DEFAULT_CONF = {"ARIMA": {"order": (1, 1, 0), "trend": "t"}}


def series_to_prophet_df(x: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame()
    if isinstance(x, pd.Series):
        idx = x.index
        df["y"] = x.values
    else:
        idx = x
    df["ds"] = idx.tz_localize(None)
    return df


class SkSTLForecast(RegressorMixin, BaseSTL):
    """
    A model designed for time series forecasting or backcasting
    (predicting past values).
    It applies seasonal-trend decomposition (STL) to the training data to capture both
    trend and seasonal patterns. The model then uses ARIMA or a custom autoregressive
    model to predict these components, as well as the overall observed variable.

    Parameters
    ----------
    period : int, str, or datetime.timedelta
        The period of the time series (e.g., daily, weekly, monthly, etc.).
        Can be an integer, string, or timedelta.
        This defines the seasonal periodicity for the STL decomposition.

    trend : int, str, or datetime.timedelta
        The length of the trend smoother. If an int is specified, it must be odd and
        larger than season. Statsplot indicate it is usually around 150% of season.
        Strongly depends on your time series.

    ar_model : object, optional
        A string corresponding to the name of the Autoregressive model to be used
        to predict STL trend an periodic component.
        The name must be chosen among MODEL_MAP keys()
        If not provided, ARIMA will be used as the default model.

    seasonal : int, str, or datetime.timedelta, optional
        The seasonal component's smoothing parameter for STL. It defines how much
        the seasonal component is smoothed. If given as an integer,
        it must be an odd number. If None, a default value will be used.

    stl_kwargs : dict[str, float], optional
        Additional keyword arguments for the STL decomposition.
        These allow fine-tuning of the decomposition process.
        (https://www.statsmodels.org/stable/index.html)

    ar_kwargs : dict, optional
        Keyword arguments to be passed to the autoregressive model
        (e.g., order for ARIMA).
    backcast : bool, optional
        If True, the model will be trained to backcast (predict the past), otherwise,
        it will perform standard forward forecasting.

    Attributes
    ----------
    forecaster_ : dict
        Dictionary containing the fitted forecaster for each feature in the time series.
    train_dat_end_ : pandas.Timestamp
        Timestamp of the last data point used in training.
    training_freq_ : pandas.tseries.offsets.BaseOffset
        Frequency of the training data, either provided explicitly or inferred.

    """

    def __init__(
        self,
        period: int | str | dt.timedelta = "24h",
        trend: int | str | dt.timedelta = "15d",
        ar_model: str = "ARIMA",
        seasonal: int | str | dt.timedelta = None,
        stl_kwargs: dict[str, float] = None,
        ar_kwargs: str | dict = None,
        backcast: bool = False,
    ):
        super().__init__(period, trend, seasonal, stl_kwargs)
        self.backcast = backcast
        self.ar_model = ar_model
        self.ar_kwargs = ar_kwargs

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        ar_model = MODEL_MAP[self.ar_model]
        if self.ar_kwargs is None:
            ar_kwargs = MODEL_DEFAULT_CONF[self.ar_model]
        else:
            ar_kwargs = self.ar_kwargs

        self._pre_fit(X)
        self.training_freq_ = (
            X.index.freq if X.index.freq is not None else X.index.inferred_freq
        )
        if self.backcast:
            X = X[::-1]
        self.train_dat_end_ = X.index[-1]
        self.forecaster_ = {}

        for feat in X:
            self.forecaster_[feat] = STLForecast(
                endog=X[feat].to_numpy(),
                model=ar_model,
                model_kwargs=ar_kwargs,
                **self.stl_kwargs,
            ).fit()

        return self

    def predict(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(
            self,
            attributes=[
                "forecaster_",
                "train_dat_end_",
                "training_freq_",
            ],
        )

        X = check_and_return_dt_index_df(X)
        check_array(X)

        if X.index.shape[0] == 2:
            X.index.freq = pd.tseries.frequencies.to_offset(
                abs(X.index[-1] - X.index[0])
            )

        if X.index.shape[0] > 1 and X.index.freq != self.training_freq_:
            raise ValueError(
                f"Required prediction freq {X.index.freq} "
                f"differs from training_freq_ {self.training_freq_}"
            )

        if (self.backcast and X.index[-1] >= self.train_dat_end_) or (
            not self.backcast and X.index[0] <= self.train_dat_end_
        ):
            direction = "future" if self.backcast else "past"
            raise ValueError(
                f"Cannot forecast on {direction} values or training data. "
                f"{'Backcast' if self.backcast else 'Forecast'} can only happen "
                f"{'before' if self.backcast else 'after'} {self.train_dat_end_}"
            )

        output_index = X.index[::-1] if self.backcast else X.index

        if set(self.forecaster_.keys()) != set(X.columns):
            warnings.warn(
                "Columns in X differs from columns in the training DataSet. "
                "Forecast will be performed for the trained data",
                UserWarning,
            )

        casting_steps = int(
            len(output_index)
            + abs(output_index[0] - self.train_dat_end_) / self.training_freq_
            - 1
        )
        steps_to_jump = casting_steps - len(output_index)
        inferred_df = pd.DataFrame(index=output_index)
        for feat in self.forecaster_.keys():
            cast = self.forecaster_[feat].forecast(casting_steps)
            inferred_df[feat] = cast[steps_to_jump:]

        return inferred_df.sort_index()


class SkProphet(RegressorMixin, BaseEstimator, TideBaseMixin):
    """
    A scikit-learn compatible wrapper for Meta Prophet forecasting model.

    This class combines the functionality of Prophet with scikit-learn's API,
    allowing it to be used in scikit-learn pipelines and model selection tools.
    It supports multi-feature forecasting, with a separate Prophet model fitted
    for each feature.

    Parameters
    ----------
    prophet_kwargs : dict, optional (default={})
        Additional keyword arguments to be passed to the Prophet model.
    changepoint_prior_scale : float, optional (default=0.05)
        Determines the flexibility of the automatic changepoint selection.
        Large values allow many changepoints, small values allow few changepoints.
    seasonality_prior_scale : float, optional (default=10.0)
        Parameter modulating the strength of the seasonality model.
        Larger values allow the model to fit larger seasonal fluctuations.
    return_upper_lower_bounds : bool, optional (default=False)
        If True, return upper and lower prediction bounds along with the forecast.
    backcast : bool, optional (default=False)
        No effect, just here for tide FillGapAR compatibility.

    Attributes
    ----------
    forecaster_ : dict
        A dictionary of fitted Prophet models, one for each feature.
    feature_names_in_ : list
        The feature names seen during fit.

    Methods
    -------
    fit(X, y=None)
        Fit the Prophet model to the input data.
    predict(X)
        Make predictions using the fitted Prophet model.
    """

    def __init__(
        self,
        prophet_kwargs: dict = {},
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        return_upper_lower_bounds: bool = False,
        backcast: bool = False,
    ):
        super().__init__()
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.prophet_kwargs = prophet_kwargs
        self.return_upper_lower_bounds = return_upper_lower_bounds
        self.backcast = backcast

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.forecaster_ = {}
        self.fit_check_features(X)
        if self.return_upper_lower_bounds:
            self.added_columns = []
            for bound in ["upper", "lower"]:
                for feat in self.feature_names_in_:
                    parts = feat.split("__")
                    parts[0] = f"{parts[0]}_{bound}"
                    self.added_columns.append("__".join(parts))

        for feat in X:
            x = series_to_prophet_df(X[feat])
            self.forecaster_[feat] = Prophet(
                seasonality_prior_scale=self.seasonality_prior_scale,
                changepoint_prior_scale=self.changepoint_prior_scale,
                **self.prophet_kwargs,
            ).fit(x)
        return self

    def predict(self, X: pd.Series | pd.DataFrame):
        X = check_and_return_dt_index_df(X)
        check_is_fitted(
            self,
            attributes=["forecaster_", "feature_names_in_"],
        )
        if not np.all([f in self.feature_names_in_ for f in X.columns]):
            raise ValueError(
                "One of the requested feature was not present during fitting"
            )

        X = check_and_return_dt_index_df(X)
        inferred_df = pd.DataFrame(index=X.index)
        for feat in self.forecaster_.keys():
            x = series_to_prophet_df(X.index)
            prediction = self.forecaster_[feat].predict(x)
            inferred_df[feat] = prediction["yhat"].values
            if self.return_upper_lower_bounds:
                for bound in ["upper", "lower"]:
                    parts = feat.split("__")
                    parts[0] = f"{parts[0]}_{bound}"
                    bound_feat = "__".join(parts)
                    inferred_df[bound_feat] = prediction[f"yhat_{bound}"].values

        return inferred_df.sort_index()
