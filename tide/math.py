from enum import Enum

import pandas as pd
import numpy as np

from scipy import integrate
from collections.abc import Callable

from tide.utils import check_and_return_dt_index_df


def cosd(angle):
    """
    Cosine with angle input in degrees
    """
    res = np.cos(np.radians(angle))
    return res


def sind(angle):
    """
    Sine with angle input in degrees
    """

    res = np.sin(np.radians(angle))
    return res


def time_gradient(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Calculates the time gradient of a given time series `data`
    between two optional time bounds `begin` and `end`.

    Parameters:
    -----------
    data : pandas Series or DataFrame
        The time series to compute the gradient on.
        If a Series is provided, it will be converted to a DataFrame
        with a single column

    Returns:
    --------
    gradient : pandas DataFrame
        A DataFrame containing the gradient of the input time series
        for each column. The index will be a DatetimeIndex
        and the columns will be the same as the input.

    Raises:
    -------
    ValueError
        If `time_series` is not a pandas Series or DataFrame.
        If the index of `time_series` is not a pandas DateTimeIndex.

    Notes:
    ------
    This function applies the `time_data_control` function to ensure that
    the input `time_series` is formatted correctly
    for time series analysis. Then, it selects a subset of the data between
     `begin` and `end` if specified. Finally, the function computes
    the gradient of each column of the subset of the data, using the
    `np.gradient` function and the time difference between consecutive
    data points.
    """

    data = check_and_return_dt_index_df(data)

    ts_list = []
    for col in data:
        col_ts = data[col].dropna()

        chrono = col_ts.index - col_ts.index[0]
        chrono_sec = chrono.to_series().dt.total_seconds()

        ts_list.append(
            pd.Series(np.gradient(col_ts, chrono_sec), index=col_ts.index, name=col)
        )

    return pd.concat(ts_list, axis=1)


def time_integrate(data: pd.DataFrame | pd.Series) -> pd.Series:
    """
    Perform time Integration of given time series in X DartaFrame or in a Series.
    The function computes the integral of each column using `scipy.integrate.trapz`
    function and the time difference between consecutive data points.

    Parameters:
    -----------
    data : pandas Series or DataFrame
        The time series to integrate. If a Series is provided, it will be
        converted to a DataFrame with a single column.
    """

    data = check_and_return_dt_index_df(data)
    chrono = (data.index - data.index[0]).to_series()
    chrono = chrono.dt.total_seconds()

    res_series = pd.Series(dtype="float64")
    for col in data:
        res_series[col] = integrate.trapezoid(data[col], chrono)

    return res_series


def aggregate_time_series(
    data: pd.DataFrame,
    agg_method: Callable = np.mean,
    agg_method_kwarg: dict = None,
    reference_df: pd.DataFrame = None,
) -> pd.Series:
    """
    A function to perform data aggregation operations on a given DataFrame using a
    specified aggregation method. It also supports aggregation with respect to a
    reference DataFrame (eg. for error functions).

    Parameters:
    - X (pd.DataFrame): The DataFrame containing the data to be aggregated.
    - agg_method (Callable, optional): The aggregation method to be applied. Default is
        np.sum.
    - agg_method_kwarg (dict, optional): Additional keyword arguments to be passed to
        the aggregation method. Default is an empty dictionary.
    - reference_df (pd.DataFrame | None, optional): A reference DataFrame for error
        function aggregation. If provided, both result_df and reference_df should have
        the same shape. Default is None.

    Returns:
    - pd.Series: A pandas Series containing the aggregated values with column names as
        indices.

    Raises:
    - ValueError: If reference_df is provided and result_df and reference_df have
      inconsistent shapes.

    """

    agg_method_kwarg = {} if agg_method_kwarg is None else agg_method_kwarg
    data = check_and_return_dt_index_df(data)

    if reference_df is not None:
        reference_df = check_and_return_dt_index_df(reference_df)
        if not data.shape == reference_df.shape:
            raise ValueError(
                "Cannot perform aggregation results_df and "
                "reference_df have inconsistent shapes"
            )
        return pd.Series(
            [
                agg_method(data.iloc[:, i], reference_df.iloc[:, i], **agg_method_kwarg)
                for i in range(len(data.columns))
            ],
            index=data.columns,
        )

    else:
        return pd.Series(
            [
                agg_method(data.iloc[:, i], **agg_method_kwarg)
                for i in range(len(data.columns))
            ],
            index=data.columns,
        )


class AggMethod(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"
    CUMSUM = "CUMSUM"
    DIFF = "DIFF"
    TIME_INTEGRATE = "TIME_INTEGRATE"


AGG_METHOD_MAP = {
    "MEAN": "mean",
    "SUM": "sum",
    "CUMSUM": "cumsum",
    "DIFF": "diff",
    "TIME_INTEGRATE": time_integrate,
}
