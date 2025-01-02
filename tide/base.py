import os

import datetime as dt
import typing
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from tide.utils import (
    check_and_return_dt_index_df,
    timedelta_to_int,
    validate_odd_param,
    process_stl_odd_args,
    get_data_blocks,
    get_idx_freq_delta_or_min_time_interval,
    ensure_list,
)

from tide.meteo import get_oikolab_df


def _ensure_list(item):
    """
    Ensures the input is returned as a list.

    Parameters
    ----------
    item : any
        The input item to be converted to a list if it is not already one.
        If the input is `None`, an empty list is returned.

    Returns
    -------
    list
        - If `item` is `None`, returns an empty list.
        - If `item` is already a list, it is returned as is.
        - Otherwise, wraps the `item` in a list and returns it.
    """
    if item is None:
        return []
    return item if isinstance(item, list) else [item]


class BaseProcessing(ABC, TransformerMixin, BaseEstimator):
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
        removed_columns: str | list[str] = None,
        added_columns: str | list[str] = None,
    ):
        self.required_columns = required_columns
        self.removed_columns = removed_columns
        self.added_columns = added_columns

    def check_features(self, X):
        if self.required_columns is not None:
            if not set(self.required_columns).issubset(X.columns):
                raise ValueError("One or several required columns are missing")

    def fit_check_features(self, X):
        self.check_features(X)
        self.feature_names_in_ = list(X.columns)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["feature_names_in_"])
        added_columns = ensure_list(self.added_columns)
        removed_columns = ensure_list(self.removed_columns)

        features_out = self.feature_names_in_.copy() + added_columns
        return [feature for feature in features_out if feature not in removed_columns]

    def get_feature_names_in(self):
        check_is_fitted(self, attributes=["feature_names_in_"])
        return self.feature_names_in_

    def fit(self, X: pd.Series | pd.DataFrame, y=None):
        X = check_and_return_dt_index_df(X)
        self.fit_check_features(X)
        self._fit_implementation(X, y)
        return self

    def transform(self, X: pd.Series | pd.DataFrame):
        self.check_features(X)
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

        X = check_and_return_dt_index_df(X)
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
        X = check_and_return_dt_index_df(X)
        return get_data_blocks(
            X,
            is_null=True,
            select_inner=False,
            lower_td_threshold=self.gaps_lte,
            upper_td_threshold=self.gaps_gte,
            upper_threshold_inclusive=True,
            lower_threshold_inclusive=True,
            return_combination=False,
        )

    def get_gaps_mask(self, X: pd.Series | pd.DataFrame):
        gaps_dict = self.get_gaps_dict_to_fill(X)
        df_mask = pd.DataFrame(index=X.index)
        for col, idx_list in gaps_dict.items():
            if idx_list:
                combined_idx = pd.concat([idx.to_series() for idx in idx_list]).index
                df_mask[col] = X.index.isin(combined_idx)
            else:
                df_mask[col] = np.zeros_like(X.shape[0]).astype(bool)

        return df_mask


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
        elif x_freq > pd.Timedelta("1h"):
            df = df.resample(x_freq).mean()
        return df.loc[dt_idx, :]
