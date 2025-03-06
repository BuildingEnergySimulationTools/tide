import pandas as pd
import numpy as np
import datetime as dt
from functools import partial
from collections.abc import Callable

from sklearn.utils.validation import check_is_fitted
from scipy.ndimage import gaussian_filter1d

from tide.base import BaseProcessing, BaseFiller, BaseOikoMeteo
from tide.math import time_gradient
from tide.utils import (
    get_data_blocks,
    get_outer_timestamps,
    check_and_return_dt_index_df,
    parse_request_to_col_names,
    ensure_list,
)
from tide.regressors import SkSTLForecast, SkProphet
from tide.classifiers import STLEDetector
from tide.meteo import sun_position, beam_component, sky_diffuse, ground_diffuse

FUNCTION_MAP = {"mean": np.mean, "average": np.average, "sum": np.sum, "dot": np.dot}

MODEL_MAP = {"STL": SkSTLForecast, "Prophet": SkProphet}

OIKOLAB_DEFAULT_MAP = {
    "temperature": "t_ext__°C__outdoor__meteo",
    "dewpoint_temperature": "t_dp__°C__outdoor__meteo",
    "mean_sea_level_pressure": "pressure__Pa__outdoor__meteo",
    "wind_speed": "wind_speed__m/s__outdoor__meteo",
    "100m_wind_speed": "100m_wind_speed__m/s__outdoor__meteo",
    "relative_humidity": "rh__0-1RH__outdoor__meteo",
    "surface_solar_radiation": "gho__w/m²__outdoor__meteo",
    "direct_normal_solar_radiation": "dni__w/m²__outdoor__meteo",
    "surface_diffuse_solar_radiation": "dhi__w/m²__outdoor__meteo",
    "surface_thermal_radiation": "thermal_radiation__w/m²__outdoor__meteo",
    "total_cloud_cover": "total_cloud_cover__0-1cover__outdoor__meteo",
    "total_precipitation": "total_precipitation__mm__outdoor__meteo",
}


class Identity(BaseProcessing):
    """A transformer that returns input data unchanged.

    Parameters
    ----------
    None

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of input columns (set during fit).
    feature_names_out_ : list[str]
        Names of output columns (same as input).

    Methods
    -------
    fit(X, y=None)
        No-op, returns self.
    transform(X)
        Returns input unchanged.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"temp__°C": [20, 21, 22], "humid__%": [45, 50, 55]})
    >>> identity = Identity()
    >>> result = identity.fit_transform(df)
    >>> assert (result == df).all().all()  # Data unchanged
    >>> assert list(result.columns) == list(df.columns)  # Column order preserved

    Returns
    -------
    pd.DataFrame
        The input data without any modifications.
    """

    def __init__(self):
        super().__init__()

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        return X


class ReplaceDuplicated(BaseProcessing):
    """A transformer that replaces duplicated values in each column with a specified value.

    This transformer identifies and replaces duplicated values in each column
    of a pandas DataFrame, keeping either the first, last, or no occurrence
    of duplicated values.

    Parameters
    ----------
    keep : str, default 'first'
        Specify which of the duplicated (if any) value to keep.
        Allowed arguments : 'first', 'last', False.
            - 'first': Keep first occurrence of duplicated values
            - 'last': Keep last occurrence of duplicated values
            - False: Keep no occurrence (replace all duplicates)

    value : float, default np.nan
        Value used to replace the non-kept duplicated values.

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of input columns (set during fit).
    feature_names_out_ : list[str]
        Names of output columns (same as input).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from datetime import datetime, timezone
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:04:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {"temp__°C": [20, 20, 22, 22, 23], "humid__%": [45, 45, 50, 50, 55]},
    ...     index=dates,
    ... )
    >>> # Keep first occurrence of duplicates
    >>> replacer = ReplaceDuplicated(keep="first", value=np.nan)
    >>> result = replacer.fit_transform(df)
    >>> print(result)
                           temp__°C  humid__%
    2024-01-01 00:00:00+00:00      20.0      45.0
    2024-01-01 00:01:00+00:00       NaN       NaN
    2024-01-01 00:02:00+00:00      22.0      50.0
    2024-01-01 00:03:00+00:00       NaN       NaN
    2024-01-01 00:04:00+00:00      23.0      55.0

    Returns
    -------
    pd.DataFrame
        The DataFrame with duplicated values replaced according to the specified strategy.
        The output maintains the same DateTimeIndex as the input.
    """

    def __init__(self, keep="first", value=np.nan):
        super().__init__()
        self.keep = keep
        self.value = value

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        for col in X.columns:
            X.loc[X[col].duplicated(keep=self.keep), col] = self.value
        return X


class Dropna(BaseProcessing):
    """A transformer that removes rows containing missing values from a DataFrame.

    This transformer removes rows from a DataFrame based on the presence of
    missing values (NaN) according to the specified strategy.

    Parameters
    ----------
    how : str, default 'all'
        How to drop missing values in the data:
            - 'all': Drop row if all values are missing
            - 'any': Drop row if any value is missing
            - int: Drop row if at least this many values are missing

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of input columns (set during fit).
    feature_names_out_ : list[str]
        Names of output columns (same as input).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:04:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "temp__°C": [20, np.nan, 22, np.nan, np.nan],
    ...         "humid__%": [45, 50, np.nan, np.nan, np.nan],
    ...     },
    ...     index=dates,
    ... )
    >>> # Drop rows where all values are missing
    >>> dropper = Dropna(how="all")
    >>> result = dropper.fit_transform(df)
    >>> print(result)
                           temp__°C  humid__%
    2024-01-01 00:00:00+00:00      20.0      45.0
    2024-01-01 00:01:00+00:00       NaN      50.0
    2024-01-01 00:02:00+00:00      22.0       NaN
    >>> # Drop rows with any missing value
    >>> dropper_strict = Dropna(how="any")
    >>> result_strict = dropper_strict.fit_transform(df)
    >>> print(result_strict)
                           temp__°C  humid__%
    2024-01-01 00:00:00+00:00      20.0      45.0

    Returns
    -------
    pd.DataFrame
        The DataFrame with rows containing missing values removed according to
        the specified strategy. The output maintains the same DateTimeIndex
        structure as the input, with rows removed.
    """

    def __init__(self, how="all"):
        super().__init__()
        self.how = how

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        return X.dropna(how=self.how)


class RenameColumns(BaseProcessing):
    """A transformer that renames columns in a DataFrame.

    This transformer allows renaming DataFrame columns either by providing a list
    of new names in the same order as the current columns, or by providing a
    dictionary mapping old names to new names.

    Parameters
    ----------
    new_names : list[str] | dict[str, str]
        New names for the columns. Can be specified in two ways:
            - list[str]: List of new names in the same order as current columns.
            Must have the same length as the number of columns.
            - dict[str, str]: Dictionary mapping old column names to new names.
            Keys must be existing column names, values are the new names.

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of input columns (set during fit).
    feature_names_out_ : list[str]
        Names of output columns after renaming.

    Examples
    --------
    >>> import pandas as pd
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:02:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {"temp__°C": [20, 21, 22], "humid__%": [45, 50, 55]}, index=dates
    ... )
    >>> # Rename using a list (maintains order)
    >>> renamer_list = RenameColumns(["temperature__°C", "humidity__%"])
    >>> result_list = renamer_list.fit_transform(df)
    >>> print(result_list)
                           temperature__°C  humidity__%
    2024-01-01 00:00:00+00:00           20.0        45.0
    2024-01-01 00:01:00+00:00           21.0        50.0
    2024-01-01 00:02:00+00:00           22.0        55.0
    >>> # Rename using a dictionary (selective renaming)
    >>> renamer_dict = RenameColumns({"temp__°C": "temperature__°C"})
    >>> result_dict = renamer_dict.fit_transform(df)
    >>> print(result_dict)
                           temperature__°C  humid__%
    2024-01-01 00:00:00+00:00           20.0      45.0
    2024-01-01 00:01:00+00:00           21.0      50.0
    2024-01-01 00:02:00+00:00           22.0      55.0

    Returns
    -------
    pd.DataFrame
        The DataFrame with renamed columns.
    """

    def __init__(self, new_names: list[str] | dict[str, str]):
        super().__init__()
        self.new_names = new_names

    def _fit_implementation(self, X, y=None):
        if isinstance(self.new_names, list):
            if len(self.new_names) != len(X.columns):
                raise ValueError(
                    "Length of new_names list must match the number "
                    "of columns in the DataFrame."
                )
            self.feature_names_out_ = self.new_names
        elif isinstance(self.new_names, dict):
            self.feature_names_out_ = list(X.rename(columns=self.new_names))

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["feature_names_in_", "feature_names_out_"])
        if isinstance(self.new_names, list):
            X.columns = self.new_names
        elif isinstance(self.new_names, dict):
            X.rename(columns=self.new_names, inplace=True)
        return X


class SkTransform(BaseProcessing):
    """A transformer that applies scikit-learn transformers to a pandas DataFrame.

    This transformer wraps any scikit-learn transformer and applies it to a pandas
    DataFrame while preserving the DataFrame's index and column structure. It is
    particularly useful when you want to use scikit-learn's preprocessing tools
    (like StandardScaler, MinMaxScaler, etc.) while maintaining the time series
    nature of your data.

    Parameters
    ----------
    transformer : object
        A scikit-learn transformer to apply on the data. Must implement fit(),
        transform(), and optionally inverse_transform() methods.

    Attributes
    ----------
    transformer_ : object
        The fitted scikit-learn transformer.
    feature_names_in_ : list[str]
        Names of input columns (set during fit).
    feature_names_out_ : list[str]
        Names of output columns (same as input).

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.preprocessing import StandardScaler
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:02:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {"temp__°C": [20, 21, 22], "humid__%": [45, 50, 55]}, index=dates
    ... )
    >>> # Apply StandardScaler while preserving DataFrame structure
    >>> sk_transform = SkTransform(StandardScaler())
    >>> result = sk_transform.fit_transform(df)
    >>> print(result)
                           temp__°C  humid__%
    2024-01-01 00:00:00+00:00     -1.0     -1.0
    2024-01-01 00:01:00+00:00      0.0      0.0
    2024-01-01 00:02:00+00:00      1.0      1.0
    >>> # Inverse transform to get back original values
    >>> original = sk_transform.inverse_transform(result)
    >>> print(original)
                           temp__°C  humid__%
    2024-01-01 00:00:00+00:00     20.0     45.0
    2024-01-01 00:01:00+00:00     21.0     50.0
    2024-01-01 00:02:00+00:00     22.0     55.0

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with the same index and column structure as the input.
        The values are transformed according to the specified scikit-learn transformer.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        self.transformer.fit(X)
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["feature_names_in_"])
        return pd.DataFrame(
            data=self.transformer.transform(X), index=X.index, columns=X.columns
        )

    def inverse_transform(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["feature_names_in_"])
        X = check_and_return_dt_index_df(X)
        return pd.DataFrame(
            data=self.transformer.inverse_transform(X), index=X.index, columns=X.columns
        )


class ReplaceThreshold(BaseProcessing):
    """A transformer that replaces values in a DataFrame based on threshold values.

    This transformer replaces values in a DataFrame that fall outside specified
    upper and lower thresholds with a given replacement value. It is useful for
    handling outliers or extreme values in time series data.

    Parameters
    ----------
    upper : float, optional (default=None)
        The upper threshold value. Values greater than this threshold will be
        replaced with the specified value.
    lower : float, optional (default=None)
        The lower threshold value. Values less than this threshold will be
        replaced with the specified value.
    value : float, optional (default=np.nan)
        The value to use for replacing values that fall outside the thresholds.

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of input columns (set during fit).
    feature_names_out_ : list[str]
        Names of output columns (same as input).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:04:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {"temp__°C": [20, 25, 30, 35, 40], "humid__%": [45, 50, 55, 60, 65]},
    ...     index=dates,
    ... )
    >>> # Replace values outside thresholds with NaN
    >>> replacer = ReplaceThreshold(upper=35, lower=20, value=np.nan)
    >>> result = replacer.fit_transform(df)
    >>> print(result)
                           temp__°C  humid__%
    2024-01-01 00:00:00+00:00      20.0       NaN
    2024-01-01 00:01:00+00:00      25.0       NaN
    2024-01-01 00:02:00+00:00      30.0       NaN
    2024-01-01 00:03:00+00:00       NaN       NaN
    2024-01-01 00:04:00+00:00       NaN       NaN

    Returns
    -------
    pd.DataFrame
        The DataFrame with values outside the specified thresholds replaced
        with the given value. The output maintains the same DateTimeIndex
        and column structure as the input.
    """

    def __init__(self, upper=None, lower=None, value=np.nan):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.value = value

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        pass

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        if self.lower is not None:
            lower_mask = X < self.lower
        else:
            lower_mask = pd.DataFrame(
                np.full(X.shape, False), index=X.index, columns=X.columns
            )

        if self.upper is not None:
            upper_mask = X > self.upper
        else:
            upper_mask = pd.DataFrame(
                np.full(X.shape, False), index=X.index, columns=X.columns
            )

        X[np.logical_or(lower_mask, upper_mask)] = self.value

        return X


class DropTimeGradient(BaseProcessing):
    """
    A transformer that removes values in a DataFrame based on the time gradient.

    The time gradient is calculated as the difference of consecutive values in
    the time series divided by the time delta between each value (in seconds).
    If the gradient is below the `lower_rate` or above the `upper_rate`,
    then the value is set to NaN.

    Parameters
    ----------
    dropna : bool, default=True
        Whether to remove NaN values from the DataFrame before processing.
    upper_rate : float, optional
        The upper rate threshold in units of value/second. If the gradient is greater than or equal to
        this value, the value will be set to NaN.
        Example: For a temperature change of 5°C per minute, set upper_rate=5/60 ≈ 0.083
    lower_rate : float, optional
        The lower rate threshold in units of value/second. If the gradient is less than or equal to
        this value, the value will be set to NaN.
        Example: For a pressure change of 100 Pa per minute, set lower_rate=100/60 ≈ 1.67

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of input columns (set during fit).
    feature_names_out_ : list[str]
        Names of output columns (same as input).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:04:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "temp__°C": [20, 25, 30, 35, 40],  # Steady increase of 5°C/min
    ...         "humid__%": [45, 45, 45, 45, 45],  # Constant
    ...         "press__Pa": [1000, 1000, 900, 1000, 1000],  # Sudden change
    ...     },
    ...     index=dates,
    ... )
    >>> # Remove values with gradients outside thresholds
    >>> # For temperature: 5°C/min = 5/60 ≈ 0.083°C/s
    >>> # For pressure: 100 Pa/min = 100/60 ≈ 1.67 Pa/s
    >>> dropper = DropTimeGradient(upper_rate=0.083, lower_rate=0.001)
    >>> result = dropper.fit_transform(df)
    >>> print(result)
                           temp__°C  humid__%  press__Pa
    2024-01-01 00:00:00+00:00      20.0      45.0     1000.0
    2024-01-01 00:01:00+00:00      25.0       NaN     1000.0
    2024-01-01 00:02:00+00:00      30.0       NaN       NaN
    2024-01-01 00:03:00+00:00      35.0       NaN     1000.0
    2024-01-01 00:04:00+00:00      40.0      45.0     1000.0

    Notes
    -----
    - The gradient is calculated as (value2 - value1) / (time2 - time1 in seconds)
    - For the upper_rate threshold, both the current and next gradient must exceed
      the threshold for a value to be removed
    - For the lower_rate threshold, only the current gradient needs to be below
      the threshold for a value to be removed
    - NaN values are handled according to the dropna parameter:
        - If True (default): NaN values are removed before processing
        - If False: NaN values are kept and may affect gradient calculations
    - The rate parameters (upper_rate and lower_rate) must be specified in units of
      value/second. To convert from per-minute rates, divide by 60.

    Returns
    -------
    pd.DataFrame
        The DataFrame with values removed based on their time gradients.
        The output maintains the same DateTimeIndex and column structure as the input.
    """

    def __init__(self, dropna=True, upper_rate=None, lower_rate=None):
        super().__init__()
        self.dropna = dropna
        self.upper_rate = upper_rate
        self.lower_rate = lower_rate

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        pass

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        X_transformed = []
        for column in X.columns:
            X_column = X[column]
            if self.dropna:
                original_index = X_column.index.copy()
                X_column = X_column.dropna()

            time_delta = X_column.index.to_series().diff().dt.total_seconds()
            abs_der = abs(X_column.diff().divide(time_delta, axis=0))
            abs_der_two = abs(X_column.diff(periods=2).divide(time_delta, axis=0))
            if self.upper_rate is not None:
                mask_der = abs_der >= self.upper_rate
                mask_der_two = abs_der_two >= self.upper_rate
            else:
                mask_der = pd.Series(
                    np.full(X_column.shape, False),
                    index=X_column.index,
                    name=X_column.name,
                )
                mask_der_two = mask_der

            if self.lower_rate is not None:
                mask_constant = abs_der <= self.lower_rate
            else:
                mask_constant = pd.Series(
                    np.full(X_column.shape, False),
                    index=X_column.index,
                    name=X_column.name,
                )

            mask_to_remove = np.logical_and(mask_der, mask_der_two)
            mask_to_remove = np.logical_or(mask_to_remove, mask_constant)

            X_column[mask_to_remove] = np.nan
            if self.dropna:
                X_column = X_column.reindex(original_index)
            X_transformed.append(X_column)
        return pd.concat(X_transformed, axis=1)


class ApplyExpression(BaseProcessing):
    """A transformer that applies a mathematical expression to a pandas DataFrame.

    This transformer allows you to apply any valid Python mathematical expression
    to a pandas DataFrame. The expression is evaluated using pandas' `eval` function,
    which provides efficient evaluation of mathematical expressions.

    Parameters
    ----------
    expression : str
        A string representing a valid Python mathematical expression.
        The expression can use the input DataFrame `X` as a variable.
        Common operations include:
            - Basic arithmetic: +, -, *, /, **, %
            - Comparison: >, <, >=, <=, ==, !=
            - Boolean operations: &, |, ~
            - Mathematical functions: abs(), sqrt(), pow(), etc.
            Example: "X * 2" or "X / 1000" or "X ** 2"

    new_unit : str, optional (default=None)
        The new unit to apply to the column names after transformation.
        If provided, the transformer will update the unit part of the column names
        (the part after the second "__" in the Tide naming convention).
        Example: If input columns are "power__W__building" and new_unit="kW",
        output columns will be "power__kW__building".

    Examples
    --------
    >>> import pandas as pd
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:02:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "power__W__building": [1000, 2000, 3000],
    ...     },
    ...     index=dates,
    ... )
    >>> # Convert power from W to kW
    >>> transformer = ApplyExpression("X / 1000", "kW")
    >>> result = transformer.fit_transform(df)
    >>> print(result)
                           power__kW__building
    2024-01-01 00:00:00+00:00             1.0
    2024-01-01 00:01:00+00:00             2.0
    2024-01-01 00:02:00+00:00             3.0


    Notes
    -----
    - The expression is evaluated using pandas' `eval` function, which is optimized
      for numerical operations on DataFrames.
    - The input DataFrame `X` is available in the expression context.
    - When using `new_unit`, the transformer follows the Tide naming convention
      of "name__unit__block" for column names.
    - The transformer preserves the DataFrame's index and column structure.
    - All mathematical operations are applied element-wise to the DataFrame.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with the mathematical expression applied to all values.
        If new_unit is specified, the column names are updated accordingly.
    """

    def __init__(self, expression: str, new_unit: str = None):
        super().__init__()
        self.expression = expression
        self.new_unit = new_unit

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        if self.new_unit is not None:
            self.feature_names_out_ = self.get_set_tags_values_columns(
                X.copy(), 1, self.new_unit
            )

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        X = eval(self.expression)
        if self.new_unit is not None:
            X.columns = self.feature_names_out_
        return X


class TimeGradient(BaseProcessing):
    """A transformer that calculates the time gradient (derivative) of a pandas DataFrame.

    This transformer computes the rate of change of values with respect to time.
    The gradient is calculated using the time difference between consecutive data points.

    Parameters
    ----------
    new_unit : str, optional (default=None)
        The new unit to apply to the column names after transformation.
        If provided, the transformer will update the unit part of the column names
        (the part after the second "__" in the Tide naming convention).
        Example: If input columns are "energy__J__building" and new_unit="W",
        output columns will be "energy__W__building".

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:04:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> # Create energy data (in Joules) with varying consumption
    >>> df = pd.DataFrame(
    ...     {
    ...         "energy__J__building": [
    ...             0,  # Start at 0 J
    ...             360000,  # 1 kWh = 3600000 J
    ...             720000,  # 2 kWh
    ...             1080000,  # 3 kWh
    ...             1440000,  # 4 kWh
    ...         ]
    ...     },
    ...     index=dates,
    ... )
    >>> # Calculate power (W) from energy (J) using time gradient
    >>> # Power = Energy / time (in seconds)
    >>> transformer = TimeGradient(new_unit="W")
    >>> result = transformer.fit_transform(df)
    >>> print(result)
                           energy__W__building
    2024-01-01 00:00:00+00:00            NaN
    2024-01-01 00:01:00+00:00         6000.0
    2024-01-01 00:02:00+00:00         6000.0
    2024-01-01 00:03:00+00:00         6000.0
    2024-01-01 00:04:00+00:00         6000.0

    Notes
    -----
    - The time gradient is calculated as (value2 - value1) / (time2 - time1 in seconds)
    - The first and last values in each column will be NaN since they don't have
      enough neighbors to calculate the gradient
    - When using new_unit, the transformer follows the Tide naming convention
      of "name__unit__block" for column names

    Returns
    -------
    pd.DataFrame
        The DataFrame with time gradients calculated for each column.
        The output maintains the same DateTimeIndex as the input.
        If new_unit is specified, the column names are updated accordingly.
    """

    def __init__(self, new_unit: str = None):
        super().__init__()
        self.new_unit = new_unit

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        if self.new_unit is not None:
            self.feature_names_out_ = self.get_set_tags_values_columns(
                X.copy(), 1, self.new_unit
            )

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        original_index = X.index.copy()
        derivative = time_gradient(X)
        derivative = derivative.reindex(original_index)
        if self.new_unit is not None:
            derivative.columns = self.feature_names_out_
        return derivative


class Ffill(BaseFiller, BaseProcessing):
    """A transformer that forward-fills missing values in a pandas DataFrame.

    This transformer fills missing values (NaN) in a DataFrame by propagating
    the last valid observation forward. It is particularly useful when past
    values are more relevant for filling gaps than future values.

    Parameters
    ----------
    limit : int, optional (default=None)
        The maximum number of consecutive NaN values to forward-fill.
        If specified, only gaps with this many or fewer consecutive NaN values
        will be filled. Must be greater than 0 if not None.
        Example: If limit=2, a gap of 3 or more NaN values will only be
        partially filled.

    gaps_lte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only fill gaps with duration less than or equal to this value.

    gaps_gte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only fill gaps with duration greater than or equal to this value.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:04:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "temp__°C__room": [20, np.nan, np.nan, 23, 24],
    ...         "press__Pa__room": [1000, np.nan, 900, np.nan, 1000],
    ...     },
    ...     index=dates,
    ... )
    >>> # Forward-fill all missing values
    >>> filler = Ffill()
    >>> result = filler.fit_transform(df)
    >>> print(result)
                           temp__°C__room  press__Pa__room
    2024-01-01 00:00:00+00:00          20.0          1000.0
    2024-01-01 00:01:00+00:00          20.0          1000.0
    2024-01-01 00:02:00+00:00          20.0           900.0
    2024-01-01 00:03:00+00:00          23.0           900.0
    2024-01-01 00:04:00+00:00          24.0          1000.0
    >>> # Forward-fill with limit of 1
    >>> filler_limited = Ffill(limit=1)
    >>> result_limited = filler_limited.fit_transform(df)
    >>> print(result_limited)
                           temp__°C__room  press__Pa__room
    2024-01-01 00:00:00+00:00          20.0          1000.0
    2024-01-01 00:01:00+00:00          20.0          1000.0
    2024-01-01 00:02:00+00:00           NaN           900.0
    2024-01-01 00:03:00+00:00          23.0           900.0
    2024-01-01 00:04:00+00:00          24.0          1000.0
    >>> # Forward-fill only gaps of 1 hour or less
    >>> filler_timed = Ffill(gaps_lte="1h")
    >>> result_timed = filler_timed.fit_transform(df)
    >>> print(result_timed)
                           temp__°C__room  press__Pa__room
    2024-01-01 00:00:00+00:00          20.0          1000.0
    2024-01-01 00:01:00+00:00           NaN          1000.0
    2024-01-01 00:02:00+00:00           NaN           900.0
    2024-01-01 00:03:00+00:00          23.0           900.0
    2024-01-01 00:04:00+00:00          24.0          1000.0

    Notes
    -----
    - NaN values at the beginning of the time series will remain unfilled since
      there are no past values to propagate

    Returns
    -------
    pd.DataFrame
        The DataFrame with missing values forward-filled according to the specified
        parameters. The output maintains the same DateTimeIndex and column
        structure as the input.
    """

    def __init__(
        self,
        limit: int = None,
        gaps_lte: str | pd.Timedelta | dt.timedelta = None,
        gaps_gte: str | pd.Timedelta | dt.timedelta = None,
    ):
        BaseFiller.__init__(self)
        BaseProcessing.__init__(self)
        self.limit = limit
        self.gaps_gte = gaps_gte
        self.gaps_lte = gaps_lte

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        filled_x = X.ffill(limit=self.limit)

        if not (self.gaps_gte or self.gaps_lte):
            return filled_x

        gaps_mask = self.get_gaps_mask(X)
        X[gaps_mask] = filled_x[gaps_mask]
        return X


class Bfill(BaseFiller, BaseProcessing):
    """A transformer that back-fills missing values in a pandas DataFrame.

    This transformer fills missing values (NaN) in a DataFrame by propagating
    the next valid observation backward. It is particularly useful when future
    values are more relevant for filling gaps than past values.

    Parameters
    ----------
    limit : int, optional (default=None)
        The maximum number of consecutive NaN values to back-fill.
        If specified, only gaps with this many or fewer consecutive NaN values
        will be filled. Must be greater than 0 if not None.
        Example: If limit=2, a gap of 3 or more NaN values will only be
        partially filled.

    gaps_lte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only fill gaps with duration less than or equal to this value.

    gaps_gte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only fill gaps with duration greater than or equal to this value.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:04:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "temp__°C__room": [20, np.nan, np.nan, 23, 24],
    ...         "press__Pa__room": [1000, np.nan, 900, np.nan, 1000],
    ...     },
    ...     index=dates,
    ... )
    >>> # Back-fill all missing values
    >>> filler = Bfill()
    >>> result = filler.fit_transform(df)
    >>> print(result)
                           temp__°C__room  press__Pa__room
    2024-01-01 00:00:00+00:00          20.0          1000.0
    2024-01-01 00:01:00+00:00          23.0           900.0
    2024-01-01 00:02:00+00:00          23.0           900.0
    2024-01-01 00:03:00+00:00          23.0          1000.0
    2024-01-01 00:04:00+00:00          24.0          1000.0
    >>> # Back-fill with limit of 1
    >>> filler_limited = Bfill(limit=1)
    >>> result_limited = filler_limited.fit_transform(df)
    >>> print(result_limited)
                           temp__°C__room  press__Pa__room
    2024-01-01 00:00:00+00:00          20.0          1000.0
    2024-01-01 00:01:00+00:00          23.0           900.0
    2024-01-01 00:02:00+00:00           NaN           900.0
    2024-01-01 00:03:00+00:00          23.0          1000.0
    2024-01-01 00:04:00+00:00          24.0          1000.0

    Notes
    -----
    - The transformer fills NaN values by propagating the next valid observation
      backward in time
    - When limit is specified, only gaps with that many or fewer consecutive NaN
      values will be filled
    - The gaps_lte and gaps_gte parameters allow filtering gaps based on their
      duration before filling
    - The transformer preserves the DataFrame's index and column structure
    - NaN values at the end of the time series will remain unfilled since there
      are no future values to propagate

    Returns
    -------
    pd.DataFrame
        The DataFrame with missing values back-filled according to the specified
        parameters. The output maintains the same DateTimeIndex and column
        structure as the input.
    """

    def __init__(
        self,
        limit: int = None,
        gaps_lte: str | pd.Timedelta | dt.timedelta = None,
        gaps_gte: str | pd.Timedelta | dt.timedelta = None,
    ):
        BaseFiller.__init__(self)
        BaseProcessing.__init__(self)
        self.limit = limit
        self.gaps_gte = gaps_gte
        self.gaps_lte = gaps_lte

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        filled_x = X.bfill(limit=self.limit)

        if not (self.gaps_gte or self.gaps_lte):
            return filled_x

        gaps_mask = self.get_gaps_mask(X)
        X[gaps_mask] = filled_x[gaps_mask]
        return X


class FillNa(BaseFiller, BaseProcessing):
    """
    A transformer that fills missing values in a pandas DataFrame with a specified value.

    Parameters
    ----------
    value : float
        The value to use for filling missing values.

    gaps_lte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only fill gaps with duration less than or equal to this value.

    gaps_gte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only fill gaps with duration greater than or equal to this value.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> from tide.processing import FillNa

    >>> # Create a DataFrame with missing values and timezone-aware index
    >>> dates = pd.date_range(start="2024-01-01", periods=5, freq="1h", tz="UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "temperature__°C": [20.0, np.nan, np.nan, 22.0, 23.0],
    ...         "pressure__Pa": [1013.0, np.nan, 1015.0, np.nan, 1014.0],
    ...     },
    ...     index=dates,
    ... )

    >>> # Fill all missing values with 0
    >>> filler = FillNa(value=0)
    >>> df_filled = filler.fit_transform(df)
    >>> print(df_filled)
                             temperature__°C  pressure__Pa
    2024-01-01 00:00:00+00:00        20.0       1013.0
    2024-01-01 01:00:00+00:00         0.0          0.0
    2024-01-01 02:00:00+00:00         0.0       1015.0
    2024-01-01 03:00:00+00:00        22.0          0.0
    2024-01-01 04:00:00+00:00        23.0       1014.0

    >>> # Fill only gaps of 1 hour or less with -999
    >>> filler = FillNa(value=-999, gaps_lte="1h")
    >>> df_filled = filler.fit_transform(df)
    >>> print(df_filled)
                             temperature__°C  pressure__Pa
    2024-01-01 00:00:00+00:00        20.0       1013.0
    2024-01-01 01:00:00+00:00      np.nan       -999.0
    2024-01-01 02:00:00+00:00      np.nan       1015.0
    2024-01-01 03:00:00+00:00        22.0       -999.0
    2024-01-01 04:00:00+00:00        23.0       1014.0

    Notes
    -----
    - When using gap duration parameters (gaps_lte or gaps_gte), only gaps within
      the specified time ranges will be filled
    - This transformer is particularly useful for:
        - Replacing missing values with a known default value
        - Handling sensor errors or invalid measurements

    Returns
    -------
    pd.DataFrame
        A DataFrame with missing values filled according to the specified parameters.
        The output maintains the same structure and index as the input DataFrame.
    """

    def __init__(
        self,
        value: float,
        gaps_lte: str | pd.Timedelta | dt.timedelta = None,
        gaps_gte: str | pd.Timedelta | dt.timedelta = None,
    ):
        BaseFiller.__init__(self)
        BaseProcessing.__init__(self)
        self.value = value
        self.gaps_gte = gaps_gte
        self.gaps_lte = gaps_lte

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        if self.gaps_gte or self.gaps_lte:
            gaps = self.get_gaps_dict_to_fill(X)
            for col, gaps in gaps.items():
                for gap in gaps:
                    X.loc[gap, col] = X.loc[gap, col].fillna(self.value)
            return X
        else:
            return X.fillna(self.value)


class Interpolate(BaseFiller, BaseProcessing):
    """
    A transformer that interpolates missing values in a pandas DataFrame using various methods.

    Parameters
    ----------
    method : str, default="linear"
        The interpolation method to use. Sample of useful available methods:
            - "linear": Linear interpolation (default)
            - "slinear": Spline interpolation of order 1
            - "quadratic": Spline interpolation of order 2
            - "cubic": Spline interpolation of order 3
            - "barycentric": Barycentric interpolation
            - "polynomial": Polynomial interpolation
            - "krogh": Krogh interpolation
            - "piecewise_polynomial": Piecewise polynomial interpolation
            - "spline": Spline interpolation
            - "pchip": Piecewise cubic Hermite interpolation
            - "akima": Akima interpolation
            - "cubicspline": Cubic spline interpolation
            - "from_derivatives": Interpolation from derivatives

    gaps_lte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only interpolate gaps with duration less than or equal to this value.

    gaps_gte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only interpolate gaps with duration greater than or equal to this value.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> from tide.processing import Interpolate

    >>> # Create a DataFrame with missing values and timezone-aware index
    >>> dates = pd.date_range(start="2024-01-01", periods=5, freq="1h", tz="UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "temperature__°C": [20.0, np.nan, np.nan, 22.0, 23.0],
    ...         "pressure__Pa": [1013.0, np.nan, 1015.0, np.nan, 1014.0],
    ...     },
    ...     index=dates,
    ... )

    >>> # Linear interpolation of all missing values
    >>> interpolator = Interpolate(method="linear")
    >>> df_interpolated = interpolator.fit_transform(df)
    >>> print(df_interpolated)
                             temperature__°C  pressure__Pa
    2024-01-01 00:00:00+00:00        20.0       1013.0
    2024-01-01 01:00:00+00:00        20.7       1014.0
    2024-01-01 02:00:00+00:00        21.3       1015.0
    2024-01-01 03:00:00+00:00        22.0       1014.5
    2024-01-01 04:00:00+00:00        23.0       1014.0

    Notes
    -----
    - When using gap duration parameters (gaps_lte or gaps_gte), only gaps within
      the specified time ranges will be interpolated
    - Different interpolation methods may produce different results:
        - Linear interpolation is simple but may not capture complex patterns
        - Cubic interpolation provides smoother curves but may overshoot

    Returns
    -------
    pd.DataFrame
        A DataFrame with missing values interpolated according to the specified parameters.
        The output maintains the same structure and index as the input DataFrame.
    """

    def __init__(
        self,
        method: str = "linear",
        gaps_lte: str | pd.Timedelta | dt.timedelta = None,
        gaps_gte: str | pd.Timedelta | dt.timedelta = None,
    ):
        BaseFiller.__init__(self)
        BaseProcessing.__init__(self)
        self.method = method
        self.gaps_gte = gaps_gte
        self.gaps_lte = gaps_lte

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        gaps_mask = self.get_gaps_mask(X)
        X_full = X.interpolate(method=self.method)
        X[gaps_mask] = X_full[gaps_mask]
        return X


class Resample(BaseProcessing):
    """A transformer that resamples time series data to a different frequency.

    This transformer allows you to resample time series data to a different frequency
    while applying specified aggregation methods. It supports both simple resampling
    with a single method for all columns and custom methods for specific columns
    using Tide's naming convention.

    Parameters
    ----------
    rule : str | pd.Timedelta | dt.timedelta
        The frequency to resample to. Can be specified as:
            - String: '1min', '5min', '1h', '1D', etc.
            - Timedelta object: pd.Timedelta('1 hour')
            - datetime.timedelta object: dt.timedelta(hours=1)

    method : str | Callable, default='mean'
        The default aggregation method to use for resampling.
        Can be:
            - String: 'mean', 'sum', 'min', 'max', 'std', etc.
            - Callable: Any function that can be used with pandas' resample

    tide_format_methods : dict[str, str | Callable], optional (default=None)
        A dictionary mapping Tide tag components to specific aggregation methods.
        Keys are the components to match (name, unit, block, sub_block).
        Values are the aggregation methods to use for matching columns.
        Example: {'name': 'power', 'method': 'sum'} will use sum aggregation
        for all columns with 'power' in their name.

    columns_methods : list[tuple[list[str], str | Callable]], optional (default=None)
        A list of tuples specifying custom methods for specific columns.
        Each tuple contains:
            - list[str]: List of column names to apply the method to
            - str | Callable: The aggregation method to use
        Example: [(['power__W__building'], 'sum')]

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:04:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "power__W__building": [1000, 1200, 1100, 1300, 1400],
    ...         "temp__°C__room": [20, 21, 22, 23, 24],
    ...         "humid__%__room": [45, 46, 47, 48, 49],
    ...     },
    ...     index=dates,
    ... )
    >>> # Resample to 5-minute intervals using mean
    >>> resampler = Resample(rule="5min")
    >>> result = resampler.fit_transform(df)
    >>> print(result)
                           power__W__building  temp__°C__room  humid__%__room
    2024-01-01 00:00:00+00:00           1100.0           21.0           46.0
    2024-01-01 00:05:00+00:00           1350.0           23.5           48.5
    >>> # Resample with custom methods
    >>> resampler_custom = Resample(
    ...     rule="5min",
    ...     tide_format_methods={"name": "power", "method": "min"},
    ...     columns_methods=[(["temp__°C__room"], "max")],
    ... )


    Notes
    -----
    - When using tide_format_methods, the matching is done on the Tide tag components
      (name__unit__block__sub_block)
    - If tide_format_methods is provided, it takes precedence over columns_methods
      and completely replaces it during fitting
    - If no custom method is specified for a column, the default method is used
    - The output frequency is determined by the rule parameter
    - Missing values in the input are handled according to the specified methods

    Returns
    -------
    pd.DataFrame
        The resampled DataFrame with the specified frequency and aggregation methods.
        The output maintains the same column structure as the input, with values
        aggregated according to the specified methods.
    """

    def __init__(
        self,
        rule: str | pd.Timedelta | dt.timedelta,
        method: str | Callable = "mean",
        tide_format_methods: dict[str, str | Callable] = None,
        columns_methods: list[tuple[list[str], str | Callable]] = None,
    ):
        super().__init__()
        self.rule = rule
        self.method = method
        self.tide_format_methods = tide_format_methods
        self.columns_methods = columns_methods

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        if self.tide_format_methods:
            self.columns_methods = []
            for req, method in self.tide_format_methods.items():
                self.columns_methods.append(
                    (parse_request_to_col_names(X.columns, req), method)
                )

        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["feature_names_in_"])
        if not self.columns_methods:
            agg_dict = {col: self.method for col in X.columns}
        else:
            agg_dict = {col: agg for cols, agg in self.columns_methods for col in cols}
            for col in X.columns:
                if col not in agg_dict.keys():
                    agg_dict[col] = self.method

        return X.resample(rule=self.rule).agg(agg_dict)[X.columns]


class AddTimeLag(BaseProcessing):
    """A transformer that adds time-lagged features to a pandas DataFrame.

    This transformer creates new features by shifting existing features in time,
    allowing the creation of past or future values as new features. This is
    particularly useful for time series analysis where historical or future
    values might be relevant predictors.

    Parameters
    ----------
    time_lag : str | pd.Timedelta | dt.timedelta, default="1h"
        The time lag to apply when creating new features. Can be specified as:
            - A string (e.g., "1h", "30min", "1d")
            - A pandas Timedelta object
            - A datetime timedelta object
        A positive time lag creates features with past values, while a negative
        time lag creates features with future values.

    features_to_lag : str | list[str] | None, default=None
        The features to create lagged versions of. If None, all features in the
        input DataFrame will be lagged. Can be specified as:
            - A single feature name (string)
            - A list of feature names
            - None (to lag all features)

    feature_marker : str | None, default=None
        The prefix to use for the new lagged feature names. If None, the
        string representation of time_lag followed by an underscore is used.
        For example, with time_lag="1h", features will be prefixed with "1h_".

    drop_resulting_nan : bool, default=False
        Whether to drop rows containing NaN values that result from the lag
        operation. This is useful when you want to ensure complete data for
        the lagged features.

    Examples
    --------
    >>> import pandas as pd
    >>> from tide.processing import AddTimeLag
    >>> # Create sample data
    >>> dates = pd.date_range(start="2024-01-01", periods=5, freq="1h", tz="UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "power__W__building": [100, 200, 300, 400, 500],
    ...         "temp__°C__room": [20, 21, 22, 23, 24],
    ...     },
    ...     index=dates,
    ... )
    >>> # Add 1-hour lagged features
    >>> lagger = AddTimeLag(time_lag="1h")
    >>> result = lagger.fit_transform(df)
    >>> print(result)
                           power__W__building  temp__°C__room  1h_power__W__building  1h_temp__°C__room
    2024-01-01 00:00:00               100.0           20.0                   NaN               NaN
    2024-01-01 01:00:00               200.0           21.0                 100.0              20.0
    2024-01-01 02:00:00               300.0           22.0                 200.0              21.0
    2024-01-01 03:00:00               400.0           23.0                 300.0              22.0
    2024-01-01 04:00:00               500.0           24.0                 400.0              23.0
    >>> # Add custom lagged features with specific marker
    >>> lagger_custom = AddTimeLag(
    ...     time_lag="1h",
    ...     features_to_lag=["power__W__building"],
    ...     feature_marker="prev_",
    ...     drop_resulting_nan=True,
    ... )
    >>> result_custom = lagger_custom.fit_transform(df)
    >>> print(result_custom)
                           power__W__building  temp__°C__room  prev_power__W__building
    2024-01-01 00:00:00               200.0           21.0                    100.0
    2024-01-01 01:00:00               300.0           22.0                    200.0
    2024-01-01 02:00:00               400.0           23.0                    300.0
    2024-01-01 03:00:00               500.0           24.0                    400.0

    Notes
    -----
    - The transformer preserves the original features and adds new lagged versions
    - Lagged features are created by shifting the index and concatenating with
      the original data
    - When drop_resulting_nan=True, rows with NaN values in lagged features
      are removed from the output
    - The feature_marker parameter allows for custom naming of lagged features
    - The transformer supports both positive (past) and negative (future) lags

    Returns
    -------
    pd.DataFrame
        The input DataFrame with additional lagged features. The original
        features are preserved, and new lagged features are added with the
        specified prefix.
    """

    def __init__(
        self,
        time_lag: str | pd.Timedelta | dt.timedelta = "1h",
        features_to_lag: str | list[str] = None,
        feature_marker: str = None,
        drop_resulting_nan=False,
    ):
        BaseProcessing.__init__(self)
        self.time_lag = time_lag
        self.features_to_lag = features_to_lag
        self.feature_marker = feature_marker
        self.drop_resulting_nan = drop_resulting_nan

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        if self.features_to_lag is None:
            self.features_to_lag = X.columns
        else:
            self.features_to_lag = (
                [self.features_to_lag]
                if isinstance(self.features_to_lag, str)
                else self.features_to_lag
            )
        self.feature_marker = (
            str(self.time_lag) + "_"
            if self.feature_marker is None
            else self.feature_marker
        )

        self.required_columns = self.features_to_lag
        self.feature_names_out_.extend(
            [self.feature_marker + name for name in self.required_columns]
        )

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["feature_names_in_", "feature_names_out_"])
        to_lag = X[self.features_to_lag].copy()
        to_lag.index = to_lag.index + self.time_lag
        to_lag.columns = self.feature_marker + to_lag.columns
        X_transformed = pd.concat([X, to_lag], axis=1)
        if self.drop_resulting_nan:
            X_transformed = X_transformed.dropna()
        return X_transformed


class GaussianFilter1D(BaseProcessing):
    """A transformer that applies a 1D Gaussian filter to smooth time series data.

    This transformer applies a one-dimensional Gaussian filter to each column of
    the input DataFrame, effectively reducing high-frequency noise while preserving
    the overall trend and important features of the time series.

    Parameters
    ----------
    sigma : float, default=5
        Standard deviation of the Gaussian kernel. Controls the level of smoothing:
            - Larger values result in smoother output but may lose fine details
            - Smaller values preserve more details but may not reduce noise effectively
            - Must be positive

    mode : str, default='nearest'
        How to handle values outside the input boundaries. Options are:
            - 'nearest': Use the nearest edge value (default)
            - 'reflect': Reflect values around the edge
            - 'mirror': Mirror values around the edge
            - 'constant': Use a constant value (0)
            - 'wrap': Wrap values around the edge

    truncate : float, default=4.0
        The filter window size in terms of standard deviations. Values outside
        the range (mean ± truncate * sigma) are ignored. This parameter:
            - Controls the effective size of the filter window
            - Affects the computational efficiency
            - Must be positive

    Examples
    --------
    >>> import pandas as pd
    >>> from tide.processing import GaussianFilter1D
    >>> # Create sample data with timezone-aware index
    >>> dates = pd.date_range(start="2024-01-01", periods=5, freq="1h", tz="UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "power__W__building": [100, 150, 200, 180, 220],
    ...         "temp__°C__room": [20, 21, 22, 21, 23],
    ...     },
    ...     index=dates,
    ... )
    >>> # Apply Gaussian filter with default settings
    >>> smoother = GaussianFilter1D(sigma=2)
    >>> result = smoother.fit_transform(df)
    >>> print(result)
                           power__W__building  temp__°C__room
    2024-01-01 00:00:00+00:00           130.0           20.0
    2024-01-01 01:00:00+00:00           149.0           20.0
    2024-01-01 02:00:00+00:00           169.0           21.0
    2024-01-01 03:00:00+00:00           187.0           21.0
    2024-01-01 04:00:00+00:00           201.0           22.0

    Notes
    -----
    - The input DataFrame must have a timezone-aware DatetimeIndex
    - The filter is applied independently to each column
    - The output maintains the same index and column structure as the input
    - The smoothing effect is more pronounced at the edges of the time series


    Returns
    -------
    pd.DataFrame
        The smoothed DataFrame with the same structure as the input. Each column
        has been smoothed using the 1D Gaussian filter with the specified parameters.
    """

    def __init__(self, sigma=5, mode="nearest", truncate=4.0):
        super().__init__()
        self.sigma = sigma
        self.mode = mode
        self.truncate = truncate

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["feature_names_in_"])
        gauss_filter = partial(
            gaussian_filter1d, sigma=self.sigma, mode=self.mode, truncate=self.truncate
        )

        return X.apply(gauss_filter)


class CombineColumns(BaseProcessing):
    """A transformer that combines multiple columns in a DataFrame using various aggregation methods.

    This transformer creates a new column by combining values from multiple input columns
    using specified aggregation methods. It supports weighted and unweighted combinations,
    and can optionally drop the original columns.

    Parameters
    ----------
    function : str
        The aggregation function to use for combining columns. Valid options are:
            - "mean": Arithmetic mean of the columns
            - "sum": Sum of the columns
            - "average": Weighted average of the columns (requires weights)
            - "dot": Dot product of the columns with weights (weighted sum)

    weights : list[float | int] | np.ndarray, default=None
        Weights to apply when using 'average' or 'dot' functions. Must be provided
        for these functions and must match the number of columns. Ignored for
        'mean' and 'sum' functions.

    drop_columns : bool, default=False
        Whether to drop the original columns after combining them. If True, only
        the combined result column is returned.

    result_column_name : str, default="combined"
        The name for the new column containing the combined values.

    Examples
    --------
    >>> import pandas as pd
    >>> from tide.processing import CombineColumns
    >>> # Create sample data with timezone-aware index
    >>> dates = pd.date_range(start="2024-01-01", periods=3, freq="1h", tz="UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "power__W__building1": [100, 200, 300],
    ...         "power__W__building2": [150, 250, 350],
    ...         "power__W__building3": [200, 300, 400],
    ...     },
    ...     index=dates,
    ... )
    >>> # Combine columns using mean
    >>> combiner = CombineColumns(function="mean", result_column_name="power__W__avg")
    >>> result = combiner.fit_transform(df)
    >>> print(result)
                           power__W__building1  power__W__building2  power__W__building3  power__W__avg
    2024-01-01 00:00:00+00:00           100.0              150.0              200.0         150.0
    2024-01-01 01:00:00+00:00           200.0              250.0              300.0         250.0
    2024-01-01 02:00:00+00:00           300.0              350.0              400.0         350.0
    >>> # Combine columns using weighted average
    >>> combiner_weighted = CombineColumns(
    ...     function="average",
    ...     weights=[0.5, 0.3, 0.2],
    ...     result_column_name="power__W__weighted",
    ...     drop_columns=True,
    ... )
    >>> result_weighted = combiner_weighted.fit_transform(df)
    >>> print(result_weighted)
                           power__W__weighted
    2024-01-01 00:00:00+00:00          135.0
    2024-01-01 01:00:00+00:00          235.0
    2024-01-01 02:00:00+00:00          335.0

    Notes
    -----
    - The input DataFrame must have a timezone-aware DatetimeIndex
    - Weights must be provided when using 'average' or 'dot' functions
    - Weights are ignored for 'mean' and 'sum' functions
    - The number of weights must match the number of columns being combined
    - When drop_columns=True, only the combined result column is returned
    - The transformer preserves the index of the input DataFrame
    Returns
    -------
    pd.DataFrame
        The DataFrame with the combined column added. If drop_columns=True,
        only the combined column is returned. The output maintains the same
        index as the input.
    """

    def __init__(
        self,
        function: str,
        weights: list[float | int] | np.ndarray = None,
        drop_columns: bool = False,
        result_column_name: str = "combined",
    ):
        BaseProcessing.__init__(self)
        self.function = function
        self.weights = weights
        self.drop_columns = drop_columns
        self.result_column_name = result_column_name

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        self.fit_check_features(X)
        self.method_ = FUNCTION_MAP[self.function]
        if self.function in ["mean", "sum"] and self.weights is not None:
            raise ValueError(
                f"Weights have been provided, but {self.function} "
                f"cannot use it. Use one of 'average' or 'dot'"
            )

        if self.drop_columns:
            self.feature_names_out_ = []

        self.feature_names_out_.append(self.result_column_name)

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(
            self, attributes=["feature_names_in_", "feature_names_out_", "method_"]
        )

        if self.function == "average":
            X[self.result_column_name] = self.method_(X, axis=1, weights=self.weights)
        elif self.function == "dot":
            X[self.result_column_name] = self.method_(X, self.weights)

        else:
            X[self.result_column_name] = self.method_(X, axis=1)

        return X[self.feature_names_out_]


class STLFilter(BaseProcessing):
    """A transformer that applies Seasonal-Trend decomposition using LOESS (STL)
    to detect and filter outliers in time series data.

    This transformer decomposes each column of the input DataFrame into seasonal,
    trend, and residual components using STL decomposition. It then identifies
    outliers in the residual component based on an absolute threshold and replaces
    them with NaN values.

    Parameters
    ----------
    period : int | str | dt.timedelta
        The periodicity of the seasonal component. Can be specified as:
            - An integer for the number of observations in one seasonal cycle
            - A string representing the time frequency (e.g., '15T' for 15 minutes)
            - A timedelta object representing the duration of the seasonal cycle

    trend : int | str | dt.timedelta
        The length of the trend smoother. Must be odd and larger than season.
        Typically set to around 150% of the seasonal period. The choice depends
        on the characteristics of your time series.

    absolute_threshold : int | float
        The threshold for detecting anomalies in the residual component.
        Any value in the residual that exceeds this threshold (in absolute value)
        is considered an anomaly and replaced by NaN.

    seasonal : int | str | dt.timedelta, default=None
        The length of the smoothing window for the seasonal component.
        If not provided, it is inferred based on the period.
        Must be an odd integer if specified as an int.
        Can also be specified as a string representing a time frequency or a
        timedelta object.

    stl_additional_kwargs : dict[str, float], default=None
        Additional keyword arguments to pass to the STL decomposition.


    Notes
    -----
    - The STL decomposition is applied independently to each column
    - Outliers are detected based on the residual component of the decomposition
    - Detected outliers are replaced with NaN values
    - The trend parameter should be larger than the period parameter
    - The seasonal parameter is optional and defaults to an inferred value
    - The transformer preserves the index and column structure of the input

    Returns
    -------
    pd.DataFrame
        The input DataFrame with outliers replaced by NaN values. The output
        maintains the same index and column structure as the input.
    """

    def __init__(
        self,
        period: int | str | dt.timedelta,
        trend: int | str | dt.timedelta,
        absolute_threshold: int | float,
        seasonal: int | str | dt.timedelta = None,
        stl_additional_kwargs: dict[str, float] = None,
    ):
        super().__init__()
        self.period = period
        self.trend = trend
        self.absolute_threshold = absolute_threshold
        self.seasonal = seasonal
        self.stl_additional_kwargs = stl_additional_kwargs

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        self.stl_ = STLEDetector(
            self.period,
            self.trend,
            self.absolute_threshold,
            self.seasonal,
            self.stl_additional_kwargs,
        )
        self.stl_.fit(X)
        return self

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["feature_names_in_", "stl_"])
        errors = self.stl_.predict(X)
        errors = errors.astype(bool)
        for col in errors:
            X.loc[errors[col], col] = np.nan

        return X


class FillGapsAR(BaseFiller, BaseProcessing):
    """
    A transformer that fills gaps in time series data using autoregressive models.

    This transformer identifies and fills gaps in time series data using a specified
    model (e.g., Prophet). The filling process depends on the `recursive_fill` parameter:

    When recursive_fill=True:
        1. Identifies gaps in the data and filters them based on size thresholds
        2. Uses the largest continuous block of valid data to fit the model
        3. Fills neighboring gaps using backcasting or forecasting
        4. Optionally handles high-frequency data by:
            - Resampling to a larger timestep for better pattern recognition
            - Performing predictions at the resampled timestep
            - Using linear interpolation to restore original resolution
        5. Repeats steps 2-4 until no more gaps remain

    When recursive_fill=False:
        1. Identifies gaps in the data and filters them based on size thresholds
        2. Uses the entire dataset to fit the model
        3. Fills all gaps in a single pass using the fitted model
        4. Optionally handles high-frequency data as described above

    Parameters
    ----------
    model_name : str, default="Prophet"
        The name of the model to use for gap filling. Currently supports "Prophet" and "STL".
        Note: STL model requires recursive_fill=True as it cannot handle NaN values.

    model_kwargs : dict, default={}
        Additional keyword arguments to pass to the model during initialization.

    gaps_lte : str | datetime | pd.Timestamp, default=None
        Upper threshold for gap size. Gaps larger than this will not be filled.
        Can be a string (e.g., "1D"), datetime object, or pd.Timestamp.

    gaps_gte : str | datetime | pd.Timestamp, default=None
        Lower threshold for gap size. Gaps smaller than this will not be filled.
        Can be a string (e.g., "1h"), datetime object, or pd.Timestamp.

    resample_at_td : str | timedelta | pd.Timedelta, default=None
        Optional resampling period for high-frequency data. If provided, data will be
        resampled to this frequency before model fitting and prediction.

    recursive_fill : bool, default=False
        Whether to recursively fill gaps until no more gaps remain. If False, only
        performs one pass of gap filling. Must be True when using STL model.

    Examples
    --------
    >>> import pandas as pd
    >>> from tide.processing import FillGapsAR
    >>> # Create sample data with gaps
    >>> dates = pd.date_range(start="2024-01-01", periods=24, freq="1h", tz="UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "power__W__building": [
    ...             100,
    ...             np.nan,
    ...             np.nan,
    ...             180,
    ...             220,
    ...             190,
    ...             np.nan,
    ...             230,
    ...             180,
    ...             160,
    ...             140,
    ...             120,
    ...             110,
    ...             130,
    ...             150,
    ...             170,
    ...             190,
    ...             210,
    ...             230,
    ...             220,
    ...             200,
    ...             180,
    ...             160,
    ...             140,
    ...         ]
    ...     },
    ...     index=dates,
    ... )
    >>> # Fill gaps using Prophet model (non-recursive)
    >>> filler = FillGapsAR(
    ...     model_name="Prophet", gaps_lte="1D", gaps_gte="1h", resample_at_td="1h"
    ... )
    >>> result = filler.fit_transform(df)
    >>> # Fill gaps using STL model (recursive required)
    >>> filler = FillGapsAR(
    ...     model_name="STL",
    ...     gaps_lte="1D",
    ...     gaps_gte="1h",
    ...     recursive_fill=True,  # Required for STL
    ... )
    >>> result = filler.fit_transform(df)

    Notes
    -----
    - Gaps are filled independently for each column
    - For high-frequency data, resampling can improve pattern recognition
    - When recursive_fill=True, the model is fitted on the largest continuous block
      of valid data for each gap
    - When recursive_fill=False, the model is fitted on the entire dataset
    - STL model requires recursive_fill=True as it cannot handle NaN values
    - Prophet model requires additional dependencies (prophet package)

    Returns
    -------
    pd.DataFrame
        DataFrame with gaps filled using the specified model. The output maintains
        the same structure and timezone information as the input.
    """

    def __init__(
        self,
        model_name: str = "Prophet",
        model_kwargs: dict = {},
        gaps_lte: str | dt.datetime | pd.Timestamp = None,
        gaps_gte: str | dt.datetime | pd.Timestamp = None,
        resample_at_td: str | dt.timedelta | pd.Timedelta = None,
        recursive_fill: bool = False,
    ):
        BaseFiller.__init__(self, gaps_lte, gaps_gte)
        BaseProcessing.__init__(self)
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.resample_at_td = resample_at_td
        gaps_lte = pd.Timedelta(gaps_lte) if isinstance(gaps_lte, str) else gaps_lte
        resample_at_td = (
            pd.Timedelta(resample_at_td)
            if isinstance(resample_at_td, str)
            else resample_at_td
        )
        if (
            resample_at_td is not None
            and gaps_lte is not None
            and gaps_lte < resample_at_td
        ):
            raise ValueError(
                f"Cannot predict data for gaps LTE to {gaps_lte} with data"
                f"at a {resample_at_td} timestep"
            )
        self.recursive_fill = recursive_fill

    def _check_forecast_horizon(self, idx):
        idx_dt = idx[-1] - idx[0]
        if idx_dt == dt.timedelta(0):
            idx_dt = idx.freq
        if idx_dt < pd.to_timedelta(self.resample_at_td):
            raise ValueError(
                f"Forecaster is asked to predict at {idx_dt} in the future "
                f"or in the past."
                f" But data used for fitting have a {self.resample_at_td} frequency"
            )

    def _get_x_and_idx_at_freq(self, x, idx, backcast):
        if self.resample_at_td is not None:
            self._check_forecast_horizon(idx)
            x_out = x.resample(self.resample_at_td).mean()
            idx_out = pd.date_range(idx[0], idx[-1], freq=self.resample_at_td).floor(
                self.resample_at_td
            )
            idx_out.freq = idx_out.inferred_freq
        else:
            x_out = x
            idx_out = idx

        return x_out, idx_out

    def _fill_up_sampling(self, X, idx, col):
        beg = idx[0] - idx.freq
        end = idx[-1] + idx.freq
        # Interpolate linearly between inferred values and using neighbor data
        X.loc[idx, col] = X.loc[beg:end, col].interpolate()
        # If gap is at boundaries
        if beg < X.index[0]:
            X.loc[idx, col] = X.loc[idx, col].bfill()
        if end > X.index[-1]:
            X.loc[idx, col] = X.loc[idx, col].ffill()

    def fill_x(self, X, group, col, idx, backcast):
        check_is_fitted(self, attributes=["model_"])
        bc_model = self.model_(backcast=backcast, **self.model_kwargs)
        if self.resample_at_td:
            self._check_forecast_horizon(idx)
        x_fit, idx_pred = self._get_x_and_idx_at_freq(X.loc[group, col], idx, backcast)
        bc_model.fit(x_fit)
        to_predict = idx_pred.to_series()
        to_predict.name = col
        to_predict = to_predict[to_predict.index.isin(X.index)]
        # Here a bit dirty. STL doesn't allow forecast on its fitting set
        if self.model_name == "STL":
            to_predict = to_predict[~to_predict.isin(x_fit.index)]

        X.loc[to_predict, col] = bc_model.predict(to_predict).to_numpy().flatten()

        if self.resample_at_td is not None:
            self._fill_up_sampling(X, idx, col)

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        self.model_ = MODEL_MAP[self.model_name]

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["model_"])
        gaps = self.get_gaps_dict_to_fill(X)
        for col in X:
            if not self.recursive_fill:
                for idx in gaps[col]:
                    self.fill_x(X, X.index, col, idx, backcast=None)
            else:
                while gaps[col]:
                    data_blocks = get_data_blocks(X[col], return_combination=False)[col]
                    data_timedelta = [block[-1] - block[0] for block in data_blocks]
                    biggest_group = data_blocks[
                        data_timedelta.index(max(data_timedelta))
                    ]
                    start, end = get_outer_timestamps(biggest_group, X.index)
                    indices_to_delete = []
                    for i, idx in enumerate(gaps[col]):
                        if start in idx:
                            self.fill_x(X, biggest_group, col, idx, backcast=True)
                            indices_to_delete.append(i)
                        elif end in idx:
                            self.fill_x(X, biggest_group, col, idx, backcast=False)
                            indices_to_delete.append(i)

                    for i in sorted(indices_to_delete, reverse=True):
                        del gaps[col][i]
        return X


class ExpressionCombine(BaseProcessing):
    """A transformer that combines DataFrame columns using a mathematical expression.

    This transformer evaluates a mathematical expression using specified columns from a DataFrame,
    creating a new column with the result. It supports both simple aggregations and complex
    physical expressions, with the option to drop the source columns after computation.

    Parameters
    ----------
    columns_dict : dict[str, str]
        Dictionary mapping expression variables to DataFrame column names.
        Keys are the variable names used in the expression, and values are the
        corresponding column names in the DataFrame.

    expression : str
        Mathematical expression to evaluate, using variables defined in columns_dict.
        The expression should be a valid Python mathematical expression that can be
        evaluated using pandas.eval().

    result_column_name : str
        Name of the new column that will contain the evaluated expression result.
        Must not already exist in the DataFrame.

    drop_columns : bool, default=False
        Whether to drop the source columns used in the expression after computation.
        If True, only the result column and other non-source columns are kept.

    Attributes
    ----------
    feature_names_out_ : list[str]
        List of column names in the transformed DataFrame. If drop_columns is True,
        excludes the source columns used in the expression.

    Raises
    ------
    ValueError
        If result_column_name already exists in the DataFrame.

    Examples
    --------
    >>> from tide import ExpressionCombine
    >>> import pandas as pd
    >>> # Create sample data
    >>> df = pd.DataFrame(
    ...     {
    ...         "Tin__°C__building": [20, 21, 22],
    ...         "Text__°C__outdoor": [10, 11, 12],
    ...         "mass_flwr__m3/h__hvac": [1, 2, 3],
    ...     }
    ... )
    >>> # Calculate ventilation losses
    >>> combiner = ExpressionCombine(
    ...     columns_dict={
    ...         "T1": "Tin__°C__building",
    ...         "T2": "Text__°C__outdoor",
    ...         "m": "mass_flwr__m3/h__hvac",
    ...     },
    ...     expression="(T1 - T2) * m * 1004 * 1.204",
    ...     result_column_name="loss_ventilation__J__hvac",
    ...     drop_columns=True,
    ... )
    >>> # Transform the data
    >>> result = combiner.fit_transform(df)
    """

    def __init__(
        self,
        columns_dict: dict[str, str],
        expression: str,
        result_column_name: str,
        drop_columns: bool = False,
    ):
        BaseProcessing.__init__(self, required_columns=list(columns_dict.values()))
        self.columns_dict = columns_dict
        self.required_columns = list(columns_dict.values())
        self.expression = expression
        self.result_column_name = result_column_name
        self.drop_columns = drop_columns

    def _fit_implementation(self, X, y=None):
        if self.drop_columns:
            self.feature_names_out_ = list(X.columns.drop(self.required_columns))
        if self.result_column_name in self.feature_names_out_:
            raise ValueError(
                f"label_name {self.result_column_name} already in X columns. "
                f"It cannot be overwritten"
            )
        self.feature_names_out_.append(self.result_column_name)

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        exp = self.expression
        for key, val in self.columns_dict.items():
            exp = exp.replace(key, f'X["{val}"]')

        X.loc[:, self.result_column_name] = pd.eval(exp, target=X)
        return X[self.feature_names_out_]


class FillOikoMeteo(BaseFiller, BaseOikoMeteo, BaseProcessing):
    """A transformer that fills data gaps using meteorological data from the Oikolab API.

    This transformer identifies gaps in time series data and fills them with corresponding
    meteorological data retrieved from the Oikolab API. It supports filtering gaps based on
    their size and can handle different data frequencies through automatic interpolation
    or resampling.

    Parameters
    ----------
    gaps_lte : str | pd.Timedelta | dt.timedelta, default=None
        Maximum gap size to fill. Gaps larger than this will be ignored.
        Can be specified as a string (e.g., "24h") or timedelta object.

    gaps_gte : str | pd.Timedelta | dt.timedelta, default=None
        Minimum gap size to fill. Gaps smaller than this will be ignored.
        Can be specified as a string (e.g., "1h") or timedelta object.

    lat : float, default=43.47
        Latitude of the location for which to retrieve meteorological data.

    lon : float, default=-1.51
        Longitude of the location for which to retrieve meteorological data.

    columns_param_map : dict[str, str], default=None
        Mapping of input columns to Oikolab API parameters. If None, all columns
        will be filled with temperature data. Available Oikolab parameters are:
            - temperature
            - dewpoint_temperature
            - mean_sea_level_pressure
            - wind_speed
            - 100m_wind_speed
            - relative_humidity
            - surface_solar_radiation
            - direct_normal_solar_radiation
            - surface_diffuse_solar_radiation
            - surface_thermal_radiation
            - total_cloud_cover
            - total_precipitation

    model : str, default="era5"
        The meteorological model to use for data retrieval.

    env_oiko_api_key : str, default="OIKO_API_KEY"
        Name of the environment variable containing the Oikolab API key.

    Examples
    --------
    >>> from tide import FillOikoMeteo
    >>> import pandas as pd
    >>> # Create sample data with gaps
    >>> df = pd.DataFrame(
    ...     {
    ...         "temperature": [20, None, 22, None, 24],
    ...         "humidity": [50, None, 55, None, 60],
    ...     },
    ...     index=pd.date_range("2024-01-01", periods=5, freq="H"),
    ... )
    >>> # Initialize and fit the transformer
    >>> filler = FillOikoMeteo(
    ...     gaps_gte="1h",
    ...     gaps_lte="24h",
    ...     lat=43.47,
    ...     lon=-1.51,
    ...     columns_param_map={
    ...         "temperature": "temperature",
    ...         "humidity": "relative_humidity",
    ...     },
    ... )
    >>> # Transform the data
    >>> result = filler.fit_transform(df)

    Notes
    -----
    - Requires an Oikolab API key to be set as an environment variable.
    - If columns_param_map is not provided, all columns will be filled with temperature data
      to comply with scikit-learn API recommendations.
    - Automatically handles different data frequencies through interpolation or resampling.
    """

    def __init__(
        self,
        gaps_lte: str | pd.Timedelta | dt.timedelta = None,
        gaps_gte: str | pd.Timedelta | dt.timedelta = None,
        lat: float = 43.47,
        lon: float = -1.51,
        columns_param_map: dict[str, str] = None,
        model: str = "era5",
        env_oiko_api_key: str = "OIKO_API_KEY",
    ):
        BaseFiller.__init__(self, gaps_lte, gaps_gte)
        BaseOikoMeteo.__init__(self, lat, lon, model, env_oiko_api_key)
        BaseProcessing.__init__(self)
        self.columns_param_map = columns_param_map

    def _fit_implementation(self, X, y=None):
        if self.columns_param_map is None:
            # Dumb action fill everything with temperature
            self.columns_param_map = {col: "temperature" for col in X.columns}
        self.get_api_key_from_env()

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["api_key_"])
        gaps_dict = self.get_gaps_dict_to_fill(X)
        for col, idx_list in gaps_dict.items():
            if col in self.columns_param_map.keys():
                for idx in idx_list:
                    df = self.get_meteo_from_idx(idx, [self.columns_param_map[col]])
                    X.loc[idx, col] = df.loc[idx, self.columns_param_map[col]]
        return X


class AddOikoData(BaseOikoMeteo, BaseProcessing):
    """
    A transformer class to fetch and integrate Oikolab meteorological data
    into a given time-indexed DataFrame or Series.

    It retrieves weather data such as temperature, wind speed, or humidity
    at specified latitude and longitude, and adds it to the input DataFrame
    under user-specified column names.

    Parameters
    ----------
    lat : float, optional
        Latitude of the location for which meteorological data is to be fetched.
        Default is 43.47.
    lon : float, optional
        Longitude of the location for which meteorological data is to be fetched.
        Default is -1.51.
    param_columns_map : dict[str, str], optional
        A mapping of meteorological parameter names (keys) to column names (values)
        in the resulting DataFrame. Default is `OIKOLAB_DEFAULT_MAP`.
        Example:
         `{"temperature": "text__°C__meteo", "wind_speed": "wind__m/s__meteo"}`
    model : str, optional
        The meteorological model to use for fetching data. Default is "era5".
    env_oiko_api_key : str, optional
        The name of the environment variable containing the Oikolab API key.
        Default is "OIKO_API_KEY".

    Methods
    -------
    fit(X: pd.Series | pd.DataFrame, y=None)
        Checks the input DataFrame for conflicts with target column names
        and validates the API key availability.

    transform(X: pd.Series | pd.DataFrame)
        Fetches meteorological data and appends it to the input DataFrame
        under the specified column names at given frequency.

    Notes
    -----
    - This class requires access to the Oikolab API, and a valid API key must
      be set as an environment variable.
    - The input DataFrame must have a DateTimeIndex for fetching data at specific
      time frequencies.
    """

    def __init__(
        self,
        lat: float = 43.47,
        lon: float = -1.51,
        param_columns_map: dict[str, str] = OIKOLAB_DEFAULT_MAP,
        model: str = "era5",
        env_oiko_api_key: str = "OIKO_API_KEY",
    ):
        BaseOikoMeteo.__init__(self, lat, lon, model, env_oiko_api_key)
        BaseProcessing.__init__(self)
        self.param_columns_map = param_columns_map

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        mask = X.columns.isin(self.param_columns_map.values())
        if mask.any():
            raise ValueError(
                f"Cannot add Oikolab meteo data. {X.columns[mask]} already in columns"
            )
        self.get_api_key_from_env()
        self.feature_names_out_.extend(list(self.param_columns_map.values()))

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(
            self, attributes=["api_key_", "feature_names_in_", "feature_names_out_"]
        )
        df = self.get_meteo_from_idx(X.index, list(self.param_columns_map.keys()))
        X.loc[:, list(self.param_columns_map.values())] = df.to_numpy()
        return X


class AddSolarAngles(BaseProcessing):
    """A transformer that adds solar angles (azimuth and elevation) to a DataFrame.

    This transformer calculates and adds solar azimuth and elevation angles for a given
    location and time series. The angles are calculated using the Astronomical Almanac's
    algorithm (1950-2050) as described in Michalsky (1988) and subsequent papers.

    Parameters
    ----------
    lat : float, default=43.47
        Latitude of the location in decimal degrees.

    lon : float, default=-1.51
        Longitude of the location in decimal degrees.

    data_bloc : str, default="OTHER"
        Name of the data block to store the solar angles.

    data_sub_bloc : str, default="OTHER_SUB_BLOC"
        Name of the sub-block to store the solar angles.

    Examples
    --------
    >>> from tide import AddSolarAngles
    >>> import pandas as pd
    >>> # Create sample data with datetime index
    >>> df = pd.DataFrame(
    ...     {"temperature": [20, 21, 22]},
    ...     index=pd.date_range("2024-01-01", periods=3, freq="H"),
    ... )
    >>> # Add solar angles
    >>> transformer = AddSolarAngles(
    ...     lat=43.47, lon=-1.51, data_bloc="SOLAR", data_sub_bloc="ANGLES"
    ... )
    >>> # Transform the data
    >>> result = transformer.fit_transform(df)

    Notes
    -----
    - Requires a DataFrame with a DateTimeIndex.
    - Adds two new columns: solar_azimuth and solar_elevation.
    - Uses the Astronomical Almanac's algorithm for solar position calculations.
    - Valid for years 1950-2050. Given the course of the world right now, I don't think
      anyone will need to use this transformer for dates after 2050.
    """

    def __init__(
        self,
        lat: float = 43.47,
        lon: float = -1.51,
        data_bloc: str = "OTHER",
        data_sub_bloc: str = "OTHER_SUB_BLOC",
    ):
        self.lat = lat
        self.lon = lon
        self.data_bloc = data_bloc
        self.data_sub_bloc = data_sub_bloc
        BaseProcessing.__init__(self)

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        self.fit_check_features(X)
        self.feature_names_out_.extend(
            [
                f"sun_el__angle_deg__{self.data_bloc}__{self.data_sub_bloc}",
                f"sun_az__angle_deg__{self.data_bloc}__{self.data_sub_bloc}",
            ]
        )

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        df = pd.DataFrame(
            data=np.array([sun_position(date, self.lat, self.lon) for date in X.index]),
            columns=self.feature_names_out_[-2:],
            index=X.index,
        )
        return pd.concat([X, df], axis=1)


class ProjectSolarRadOnSurfaces(BaseProcessing):
    """
    A transformer that projects solar radiation onto surfaces with specific orientations and tilts.

    This transformer calculates the total solar radiation incident on surfaces by combining:
        - Direct beam radiation (projected onto the tilted surface)
        - Diffuse sky radiation (from the sky dome)
        - Ground-reflected radiation (albedo effect)

    Parameters
    ----------
    bni_column_name : str
        Name of the column containing beam normal irradiance (BNI) data in W/m².
        This is the direct solar radiation perpendicular to the sun's rays.

    dhi_column_name : str
        Name of the column containing diffuse horizontal irradiance (DHI) data in W/m².
        This is the scattered solar radiation from the sky dome.

    ghi_column_name : str
        Name of the column containing global horizontal irradiance (GHI) data in W/m².
        This is the total solar radiation on a horizontal surface.

    lat : float, default=43.47
        Latitude of the location in degrees. Positive for northern hemisphere.

    lon : float, default=-1.51
        Longitude of the location in degrees. Positive for eastern hemisphere.

    surface_azimuth_angles : int | float | list[int | float], default=180.0
        Azimuth angles of the surfaces in degrees east of north.
            - 0°: North-facing
            - 90°: East-facing
            - 180°: South-facing

    surface_tilt_angle : float | list[float], default=35.0
        Tilt angles of the surfaces in degrees from horizontal.
            - 0°: Horizontal surface
            - 90°: Vertical surface
            - 180°: Horizontal surface facing down

    albedo : float, default=0.25
        Ground reflectivity or albedo coefficient.
        Typical values:
            - 0.1-0.2: Dark surfaces (asphalt, forest)
            - 0.2-0.3: Grass, soil
            - 0.3-0.4: Light surfaces (concrete, sand)
            - 0.4-0.5: Snow
            - 0.8-0.9: Fresh snow

    surface_name : str | list[str], default="az_180_tilt_35"
        Names for the output columns following Tide naming convention.
        Example: "south_facing_35deg" will create
        "south_facing_35deg__W/m²__OTHER__OTHER_SUB_BLOC"

    data_bloc : str, default="OTHER"
        Tide bloc name for the output columns.

    data_sub_bloc : str, default="OTHER_SUB_BLOC"
        Tide sub_bloc name for the output columns.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> from tide.processing import ProjectSolarRadOnSurfaces
    >>> import pytz

    >>> # Create a DataFrame with solar radiation data and timezone-aware index
    >>> dates = pd.date_range(start="2024-01-01", periods=3, freq="1h", tz="UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "bni__W/m²__outdoor__meteo": [
    ...             800,
    ...             900,
    ...             1000,
    ...         ],  # Direct normal irradiance
    ...         "dhi__W/m²__outdoor__meteo": [
    ...             200,
    ...             250,
    ...             300,
    ...         ],  # Diffuse horizontal irradiance
    ...         "ghi__W/m²__outdoor__meteo": [
    ...             600,
    ...             700,
    ...             800,
    ...         ],  # Global horizontal irradiance
    ...     },
    ...     index=dates,
    ... )

    >>> # Project radiation on a south-facing surface tilted at 35 degrees
    >>> projector = ProjectSolarRadOnSurfaces(
    ...     bni_column_name="bni__W/m²__outdoor__meteo",
    ...     dhi_column_name="dhi__W/m²__outdoor__meteo",
    ...     ghi_column_name="ghi__W/m²__outdoor__meteo",
    ...     surface_azimuth_angles=180.0,  # South-facing
    ...     surface_tilt_angle=35.0,  # 35-degree tilt
    ...     surface_name="south_facing_35deg",
    ...     data_bloc="SOLAR",
    ...     data_sub_bloc="ROOF",
    ... )
    >>> result = projector.fit_transform(df)
    >>> print(result)
                             bni__W/m²__outdoor__meteo  dhi__W/m²__outdoor__meteo  ghi__W/m²__outdoor__meteo  south_facing_35deg__W/m²__SOLAR__ROOF
    2024-01-01 00:00:00+00:00                    800.0                     200.0                     600.0                                   850.5
    2024-01-01 01:00:00+00:00                    900.0                     250.0                     700.0                                   950.2
    2024-01-01 02:00:00+00:00                   1000.0                     300.0                     800.0                                  1050.8

    Notes
    -----
    - All input radiation values must be in W/m²
    - The output radiation values are also in W/m²

    Returns
    -------
    pd.DataFrame
        The input DataFrame with additional columns containing the total solar
        radiation projected onto each specified surface. The output maintains
        the same DateTimeIndex as the input.
    """

    def __init__(
        self,
        bni_column_name: str,
        dhi_column_name: str,
        ghi_column_name: str,
        lat: float = 43.47,
        lon: float = -1.51,
        surface_azimuth_angles: int | float | list[int | float] = 180.0,
        surface_tilt_angle: float | list[float] = 35.0,
        albedo: float = 0.25,
        surface_name: str | list[str] = "az_180_tilt_35",
        data_bloc: str = "OTHER",
        data_sub_bloc: str = "OTHER_SUB_BLOC",
    ):
        BaseProcessing.__init__(self)
        self.bni_column_name = bni_column_name
        self.dhi_column_name = dhi_column_name
        self.ghi_column_name = ghi_column_name
        self.lat = lat
        self.lon = lon
        self.surface_azimuth_angles = surface_azimuth_angles
        self.surface_tilt_angle = surface_tilt_angle
        self.albedo = albedo
        self.surface_name = surface_name
        self.data_bloc = data_bloc
        self.data_sub_bloc = data_sub_bloc

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        if (
            not len(ensure_list(self.surface_azimuth_angles))
            == len(ensure_list(self.surface_tilt_angle))
            == len(ensure_list(self.surface_name))
        ):
            raise ValueError("Number of surface azimuth, tilt and name does not match")

        self.required_columns = [
            self.bni_column_name,
            self.dhi_column_name,
            self.ghi_column_name,
        ]
        self.added_columns = [
            f"{name}__W/m²__{self.data_bloc}__{self.data_sub_bloc}"
            for name in ensure_list(self.surface_name)
        ]
        self.feature_names_out_.extend(self.added_columns)

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        sun_pos = np.array([sun_position(date, self.lat, self.lon) for date in X.index])
        for az, til, name in zip(
            ensure_list(self.surface_azimuth_angles),
            ensure_list(self.surface_tilt_angle),
            self.added_columns,
        ):
            X[name] = (
                beam_component(
                    til, az, 90 - sun_pos[:, 0], sun_pos[:, 1], X[self.bni_column_name]
                )
                + sky_diffuse(til, X[self.dhi_column_name])
                + ground_diffuse(til, X[self.ghi_column_name], self.albedo)
            )

        return X


class FillOtherColumns(BaseFiller, BaseProcessing):
    """A transformer that fills missing values in specified columns using values
    from corresponding filler columns.

    This transformer is useful when you have multiple columns measuring the
    same quantity (e.g., temperature from different sensors) and want to use one
    column to fill gaps in another. Or fill gaps with computed values, for example
    solar radiations on a pyranometer from projected radiations based on
    meteo services.

    Parameters
    ----------
    gaps_lte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only fill gaps with duration less than or equal to this value.

    gaps_gte : str | pd.Timedelta | dt.timedelta, optional (default=None)
        Only fill gaps with duration greater than or equal to this value.

    columns_map : dict[str, str], optional (default={})
        A mapping of target columns to their corresponding filler columns.
        Keys are the columns with gaps to be filled.
        Values are the columns to use for filling the gaps.
        Example: {'temp__°C__room1': 'temp__°C__room2'}

    drop_filling_columns : bool, default=False
        Whether to remove the filler columns after filling the gaps.
        If True, only the target columns remain in the output.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:04:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "temp__°C__room1": [20, np.nan, np.nan, 23, 24],
    ...         "temp__°C__room2": [21, 22, 22, 22, 23],
    ...         "humid__%__room1": [45, np.nan, 47, np.nan, 49],
    ...         "humid__%__room2": [46, 46, 48, 48, 50],
    ...     },
    ...     index=dates,
    ... )
    >>> # Fill gaps in room1 using room2 data
    >>> filler = FillOtherColumns(
    ...     columns_map={
    ...         "temp__°C__room1": "temp__°C__room2",
    ...         "humid__%__room1": "humid__%__room2",
    ...     }
    ... )
    >>> result = filler.fit_transform(df)
    >>> print(result)
                           temp__°C__room1  temp__°C__room2  humid__%__room1  humid__%__room2
    2024-01-01 00:00:00+00:00          20.0           21.0            45.0           46.0
    2024-01-01 00:01:00+00:00          22.0           22.0            46.0           46.0
    2024-01-01 00:02:00+00:00          22.0           22.0            47.0           48.0
    2024-01-01 00:03:00+00:00          23.0           22.0            48.0           48.0
    2024-01-01 00:04:00+00:00          24.0           23.0            49.0           50.0
    >>> # Fill gaps and drop filler columns
    >>> filler_drop = FillOtherColumns(
    ...     columns_map={
    ...         "temp__°C__room1": "temp__°C__room2",
    ...         "humid__%__room1": "humid__%__room2",
    ...     },
    ...     drop_filling_columns=True,
    ... )
    >>> result_drop = filler_drop.fit_transform(df)
    >>> print(result_drop)
                           temp__°C__room1  humid__%__room1
    2024-01-01 00:00:00+00:00          20.0            45.0
    2024-01-01 00:01:00+00:00          22.0            46.0
    2024-01-01 00:02:00+00:00          22.0            47.0
    2024-01-01 00:03:00+00:00          23.0            48.0
    2024-01-01 00:04:00+00:00          24.0            49.0

    Notes
    -----
    - When using gap duration parameters (gaps_lte or gaps_gte), only gaps within
      the specified time ranges will be filled
    - The filler columns must contain valid values at the timestamps where
      the target columns have gaps
    - If drop_filling_columns is True, the output DataFrame will only contain
      the target columns with filled gaps

    Returns
    -------
    pd.DataFrame
        The DataFrame with gaps filled using values from the specified filler columns.
        If drop_filling_columns is True, the filler columns are removed from the output.
    """

    def __init__(
        self,
        gaps_lte: str | pd.Timedelta | dt.timedelta = None,
        gaps_gte: str | pd.Timedelta | dt.timedelta = None,
        columns_map: dict[str, str] = {},
        drop_filling_columns: bool = False,
    ):
        BaseFiller.__init__(self, gaps_lte, gaps_gte)
        BaseProcessing.__init__(self)
        self.columns_map = columns_map
        self.drop_filling_columns = drop_filling_columns

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        self.required_columns = list(self.columns_map.keys()) + list(
            self.columns_map.values()
        )
        if self.drop_filling_columns:
            self.removed_columns = list(self.columns_map.values())
            self.feature_names_out_ = list(X.columns.drop(self.removed_columns))

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        gap_dict = self.get_gaps_dict_to_fill(X[list(self.columns_map.keys())])
        for col, idxs in gap_dict.items():
            for idx in idxs:
                X.loc[idx, col] = X.loc[idx, self.columns_map[col]]
        return (
            X.drop(self.removed_columns, axis="columns")
            if self.drop_filling_columns
            else X
        )


class DropColumns(BaseProcessing):
    """A transformer that removes specified columns from a pandas DataFrame.

    It is particularly useful for data preprocessing when certain columns are
    no longer needed or for removing intermediate calculation columns.

    Parameters
    ----------
    columns : str | list[str], optional (default=None)
        The column name or a list of column names to be dropped.
        If None, no columns are dropped and the DataFrame is returned unchanged.
        Example: 'temp__°C' or ['temp__°C', 'humid__%']

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of input columns (set during fit).
    feature_names_out_ : list[str]
        Names of output columns (input columns minus dropped columns).

    Examples
    --------
    >>> import pandas as pd
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:02:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "temp__°C": [20, 21, 22],
    ...         "humid__%": [45, 50, 55],
    ...         "press__Pa": [1000, 1010, 1020],
    ...     },
    ...     index=dates,
    ... )
    >>> # Drop a single column
    >>> dropper = DropColumns(columns="temp__°C")
    >>> result = dropper.fit_transform(df)
    >>> print(result)
                           humid__%  press__Pa
    2024-01-01 00:00:00+00:00     45.0     1000.0
    2024-01-01 00:01:00+00:00     50.0     1010.0
    2024-01-01 00:02:00+00:00     55.0     1020.0
    >>> # Drop multiple columns
    >>> dropper_multi = DropColumns(columns=["temp__°C", "humid__%"])
    >>> result_multi = dropper_multi.fit_transform(df)
    >>> print(result_multi)
                           press__Pa
    2024-01-01 00:00:00+00:00     1000.0
    2024-01-01 00:01:00+00:00     1010.0
    2024-01-01 00:02:00+00:00     1020.0

    Notes
    -----
    - If a specified column doesn't exist in the DataFrame, it will be silently
      ignored
    - The order of remaining columns is preserved
    - If no columns are specified (columns=None), the DataFrame is returned
      unchanged

    Returns
    -------
    pd.DataFrame
        The DataFrame with specified columns removed. The output maintains
        the same DateTimeIndex as the input, with only the specified columns
        removed.
    """

    def __init__(self, columns: str | list[str] = None):
        self.columns = columns
        BaseProcessing.__init__(self)

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        self.required_columns = self.columns
        if self.columns is not None:
            self.feature_names_out_ = list(X.columns.drop(self.columns))

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        return (
            X.drop(self.required_columns, axis="columns")
            if self.columns is not None
            else X
        )


class ReplaceTag(BaseProcessing):
    """A transformer that replaces components of Tide tag names with new values.

    This transformer allows you to selectively replace parts of Tide tag names
    (components separated by "__") with new values. It is particularly useful
    for standardizing tag names, updating units, or changing block/sub-block
    names across multiple columns.

    Parameters
    ----------
    tag_map : dict[str, str], optional (default=None)
        A dictionary mapping old tag components to new values.
        Keys are the components to replace, values are their replacements.
        Example: {'°C': 'K', 'room1': 'room2'}
        If None, no replacements are made and the DataFrame is returned unchanged.

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of input columns (set during fit).
    feature_names_out_ : list[str]
        Names of output columns with replaced tag components.

    Examples
    --------
    >>> import pandas as pd
    >>> # Create DataFrame with DateTimeIndex
    >>> dates = pd.date_range(
    ...     start="2024-01-01 00:00:00", end="2024-01-01 00:02:00", freq="1min"
    ... ).tz_localize("UTC")
    >>> df = pd.DataFrame(
    ...     {
    ...         "temp__°C__room1__north": [20, 21, 22],
    ...         "humid__%__room1__north": [45, 50, 55],
    ...         "press__Pa__room1__north": [1000, 1010, 1020],
    ...     },
    ...     index=dates,
    ... )
    >>> # Replace room1 with room2 and °C with K
    >>> replacer = ReplaceTag(
    ...     tag_map={
    ...         "room1": "room2",
    ...         "°C": "K",  # It is dumb, just for the exemple
    ...     }
    ... )
    >>> result = replacer.fit_transform(df)
    >>> print(result)
                           temp__K__room2__north  humid__%__room2__north  press__Pa__room2__north
    2024-01-01 00:00:00+00:00              20.0                     0.45                   1000.0
    2024-01-01 00:01:00+00:00              21.0                     0.50                   1010.0
    2024-01-01 00:02:00+00:00              22.0                     0.55                   1020.0

    Notes
    -----
    - Tide tags follow the format "name__unit__block__sub_block"
    - The transformer preserves the order of tag components
    - Components not specified in tag_map remain unchanged
    - If tag_map is None, the DataFrame is returned unchanged

    Returns
    -------
    pd.DataFrame
        The DataFrame with updated column names based on the tag replacements.
        The output maintains the same DateTimeIndex and data values as the input.
    """

    def __init__(self, tag_map: dict[str, str] = None):
        self.tag_map = tag_map
        BaseProcessing.__init__(self)

    def _fit_implementation(self, X: pd.Series | pd.DataFrame, y=None):
        self.fit_check_features(X)
        self.feature_names_out_ = []
        for col in self.feature_names_in_:
            parts = col.split("__")
            updated_parts = [self.tag_map.get(part, part) for part in parts]
            self.feature_names_out_.append("__".join(updated_parts))

    def _transform_implementation(self, X: pd.Series | pd.DataFrame):
        check_is_fitted(self, attributes=["feature_names_in_", "feature_names_out_"])
        X.columns = self.feature_names_out_
        return X
