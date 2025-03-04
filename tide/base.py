import os

import datetime as dt
import typing
from abc import ABC, abstractmethod

import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from tide.utils import (
    check_and_return_dt_index_df,
    timedelta_to_int,
    validate_odd_param,
    process_stl_odd_args,
    get_blocks_lte_and_gte,
    get_idx_freq_delta_or_min_time_interval,
    get_tags_max_level,
    NAME_LEVEL_MAP,
    get_blocks_mask_lte_and_gte,
)

from tide.meteo import get_oikolab_df


class TideBaseMixin:
    """
    This class is designed to provide Tide base functionalities including :
    - checking features in and out
    - checking mandatory features
    - Modifying features names according to tide's tags

    Parameters
    ----------
    required_columns : str or list[str], optional
        Column names that must be present in the input data. Defaults to None.
    removed_columns : str or list[str], optional
        Column that will be removed during the transform process. Defaults to None.
    added_columns : str or list[str], optional
        Column that will be added to the output feature set during transform
        process. Defaults to None.

    Methods
    -------
    check_features(X):
        Ensures that the required columns are present in the input DataFrame.
    fit_check_features(X):
        Checks required columns and stores the initial feature names.
    get_feature_names_out():
        Computes the final set of feature names, accounting for added and removed
        columns.
    get_feature_names_in():
        Returns the names of the features as initially fitted.
    """

    def __init__(self, required_columns: str | list[str] = None):
        self.required_columns = required_columns

    def check_required_features(self, X):
        if self.required_columns is not None:
            if not set(self.required_columns).issubset(X.columns):
                raise ValueError("One or several required columns are missing")

    def fit_check_features(self, X):
        self.check_required_features(X)
        self.feature_names_in_ = self.feature_names_out_ = list(X.columns)

    def get_set_tags_values_columns(self, X, level: int | str, value: str):
        nb_tags = get_tags_max_level(X.columns)
        level = NAME_LEVEL_MAP(level) if isinstance(level, str) else level
        if level > nb_tags:
            raise ValueError(
                f"Asking for level {level} tag (indexing from 0). "
                f"Only {nb_tags} tags found in columns"
            )

        new_columns = []
        for col in X.columns:
            parts = col.split("__")
            parts[level] = value
            new_columns.append("__".join(parts))

        return new_columns

    def set_tags_values(self, X, tag_level: int, value: str):
        X.columns = self.get_set_tags_values_columns(X, tag_level, value)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["feature_names_in_", "feature_names_out_"])
        return self.feature_names_out_

    def get_feature_names_in(self):
        check_is_fitted(self, attributes=["feature_names_in_"])
        return self.feature_names_in_


class BaseProcessing(ABC, TransformerMixin, BaseEstimator, TideBaseMixin):
    """
    Abstract base class for processing pipelines with feature checks and
    transformation logic.

    This class is designed to facilitate transformations by checking input data
    (DataFrame or Series with DatetimeIndex), ensuring the presence
    of required features, tracking added and removed features, and enabling
    seamless integration with scikit-learn's API through fit and transform
    methods.

    Parameters
    ----------
    required_columns : str or list[str], optional
        Column names that must be present in the input data. Defaults to None.
    removed_columns : str or list[str], optional
        Column that will be removed during the transform process. Defaults to None.
    added_columns : str or list[str], optional
        Column that will be added to the output feature set during transform
        process. Defaults to None.

    Methods
    -------
    check_features(X):
        Ensures that the required columns are present in the input DataFrame.
    fit_check_features(X):
        Checks required columns and stores the initial feature names.
    get_feature_names_out():
        Computes the final set of feature names, accounting for added and removed
        columns.
    get_feature_names_in():
        Returns the names of the features as initially fitted.
    fit(X, y=None):
        Fits the transformer to the input data.
    transform(X):
        Applies the transformation to the input data.
    _fit_implementation(X, y=None):
        Abstract method for the fitting logic. Must be implemented by subclasses.
    _transform_implementation(X):
        Abstract method for the transformation logic. Must be implemented by
        subclasses.
    """

    def __init__(
        self,
        required_columns: str | list[str] = None,
    ):
        TideBaseMixin.__init__(self, required_columns=required_columns)
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.fit_check_features(X)
        self._fit_implementation(X, y)
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        self.check_required_features(X)
        X = check_and_return_dt_index_df(X)
        return self._transform_implementation(X)

    @abstractmethod
    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        """Operations happening during fitting process"""
        pass

    @abstractmethod
    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        """Operations happening during transforming process"""
        pass


class BaseSTL(BaseEstimator):
    def __init__(
        self,
        period: int | str | dt.timedelta = "24h",
        trend: int | str | dt.timedelta = "15d",
        seasonal: int | str | dt.timedelta = None,
        stl_kwargs: dict[str, typing.Any] = None,
    ):
        self.stl_kwargs = stl_kwargs
        self.period = period
        self.trend = trend
        self.seasonal = seasonal

    def _pre_fit(self, X: pd.Series | pd.DataFrame):
        self.stl_kwargs = {} if self.stl_kwargs is None else self.stl_kwargs
        check_array(X)

        self.stl_kwargs["period"] = timedelta_to_int(self.period, X)
        validate_odd_param("trend", self.trend)
        self.stl_kwargs["trend"] = self.trend
        process_stl_odd_args("trend", X, self.stl_kwargs)
        if self.seasonal is not None:
            self.stl_kwargs["seasonal"] = self.seasonal
            process_stl_odd_args("seasonal", X, self.stl_kwargs)


class BaseFiller:
    def __init__(
        self,
        gaps_lte: str | pd.Timedelta | dt.timedelta = None,
        gaps_gte: str | pd.Timedelta | dt.timedelta = None,
    ):
        self.gaps_lte = gaps_lte
        self.gaps_gte = gaps_gte

    def get_gaps_dict_to_fill(self, X: pd.Series | pd.DataFrame):
        return get_blocks_lte_and_gte(
            X,
            is_null=True,
            lte=self.gaps_lte,
            gte=self.gaps_gte,
        )

    def get_gaps_mask(self, X: pd.Series | pd.DataFrame):
        return get_blocks_mask_lte_and_gte(
            X,
            is_null=True,
            lte=self.gaps_lte,
            gte=self.gaps_gte,
        )


class BaseOikoMeteo:
    def __init__(
        self,
        lat: float = 43.47,
        lon: float = -1.51,
        model: str = "era5",
        env_oiko_api_key: str = "OIKO_API_KEY",
    ):
        self.lat = lat
        self.lon = lon
        self.model = model
        self.env_oiko_api_key = env_oiko_api_key

    def get_api_key_from_env(self):
        self.api_key_ = os.getenv(self.env_oiko_api_key)

    def get_meteo_from_idx(self, dt_idx: pd.DatetimeIndex, param: list[str]):
        check_is_fitted(self, attributes=["api_key_"])
        x_freq = get_idx_freq_delta_or_min_time_interval(dt_idx)
        end = (
            dt_idx[-1]
            if dt_idx[-1] <= dt_idx[-1].replace(hour=23, minute=0)
            else dt_idx[-1] + pd.Timedelta("1h")
        )
        df = get_oikolab_df(
            lat=self.lat,
            lon=self.lon,
            start=dt_idx[0],
            end=end,
            api_key=self.api_key_,
            param=param,
            model=self.model,
        )

        df = df[param]
        if x_freq < pd.Timedelta("1h"):
            df = df.asfreq(x_freq).interpolate("linear")
        elif x_freq >= pd.Timedelta("1h"):
            df = df.resample(x_freq).mean()
        return df.loc[dt_idx, :]
