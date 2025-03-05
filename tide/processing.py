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
    >>> df = pd.DataFrame({
    ...     'temp__°C': [20, 21, 22],
    ...     'humid__%': [45, 50, 55]
    ... })
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:04:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C': [20, 20, 22, 22, 23],
    ...     'humid__%': [45, 45, 50, 50, 55]
    ... }, index=dates)
    >>> # Keep first occurrence of duplicates
    >>> replacer = ReplaceDuplicated(keep='first', value=np.nan)
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:04:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C': [20, np.nan, 22, np.nan, np.nan],
    ...     'humid__%': [45, 50, np.nan, np.nan, np.nan]
    ... }, index=dates)
    >>> # Drop rows where all values are missing
    >>> dropper = Dropna(how='all')
    >>> result = dropper.fit_transform(df)
    >>> print(result)
                           temp__°C  humid__%
    2024-01-01 00:00:00+00:00      20.0      45.0
    2024-01-01 00:01:00+00:00       NaN      50.0
    2024-01-01 00:02:00+00:00      22.0       NaN
    >>> # Drop rows with any missing value
    >>> dropper_strict = Dropna(how='any')
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:02:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C': [20, 21, 22],
    ...     'humid__%': [45, 50, 55]
    ... }, index=dates)
    >>> # Rename using a list (maintains order)
    >>> renamer_list = RenameColumns(['temperature__°C', 'humidity__%'])
    >>> result_list = renamer_list.fit_transform(df)
    >>> print(result_list)
                           temperature__°C  humidity__%
    2024-01-01 00:00:00+00:00           20.0        45.0
    2024-01-01 00:01:00+00:00           21.0        50.0
    2024-01-01 00:02:00+00:00           22.0        55.0
    >>> # Rename using a dictionary (selective renaming)
    >>> renamer_dict = RenameColumns({
    ...     'temp__°C': 'temperature__°C'
    ... })
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:02:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C': [20, 21, 22],
    ...     'humid__%': [45, 50, 55]
    ... }, index=dates)
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:04:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C': [20, 25, 30, 35, 40],
    ...     'humid__%': [45, 50, 55, 60, 65]
    ... }, index=dates)
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:04:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C': [20, 25, 30, 35, 40],  # Steady increase of 5°C/min
    ...     'humid__%': [45, 45, 45, 45, 45],  # Constant
    ...     'press__Pa': [1000, 1000, 900, 1000, 1000]  # Sudden change
    ... }, index=dates)
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:02:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'power__W__building': [1000, 2000, 3000],
    ... }, index=dates)
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:04:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> # Create energy data (in Joules) with varying consumption
    >>> df = pd.DataFrame({
    ...     'energy__J__building': [
    ...         0,          # Start at 0 J
    ...         360000,     # 1 kWh = 3600000 J
    ...         720000,     # 2 kWh
    ...         1080000,    # 3 kWh
    ...         1440000     # 4 kWh
    ...     ]
    ... }, index=dates)
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
    """
    A class to front-fill missing values in a Pandas DataFrame.
    the limit argument allows the function to stop frontfilling at a certain
    number of missing value

    Parameters:
        limit: int, default None If limit is specified, this is the maximum number
        of consecutive NaN values to forward/backward fill.
        In other words, if there is a gap with more than this number of consecutive
        NaNs, it will only be partially filled.
        If limit is not specified, this is the maximum number of entries along
        the entire axis where NaNs will be filled. Must be greater than 0 if not None.

    Methods:
        fit(self, X, y=None):
            Does nothing. Returns the object itself.
        transform(self, X):
            Fill missing values in the input DataFrame.
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:04:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C__room': [20, np.nan, np.nan, 23, 24],
    ...     'press__Pa__room': [1000, np.nan, 900, np.nan, 1000]
    ... }, index=dates)
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
    >>> import pytz

    # Create a DataFrame with missing values and timezone-aware index
    >>> dates = pd.date_range(start='2024-01-01', periods=5, freq='1h', tz='UTC')
    >>> df = pd.DataFrame({
    ...     'temperature__°C': [20.0, np.nan, np.nan, 22.0, 23.0],
    ...     'pressure__Pa': [1013.0, np.nan, 1015.0, np.nan, 1014.0]
    ... }, index=dates)

    # Fill all missing values with 0
    >>> filler = FillNa(value=0)
    >>> df_filled = filler.fit_transform(df)
    >>> print(df_filled)
                             temperature__°C  pressure__Pa
    2024-01-01 00:00:00+00:00        20.0       1013.0
    2024-01-01 01:00:00+00:00         0.0          0.0
    2024-01-01 02:00:00+00:00         0.0       1015.0
    2024-01-01 03:00:00+00:00        22.0          0.0
    2024-01-01 04:00:00+00:00        23.0       1014.0

    # Fill only gaps of 1 hour or less with -999
    >>> filler = FillNa(value=-999, gaps_lte='1h')
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
      * Replacing missing values with a known default value
      * Handling sensor errors or invalid measurements

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
    >>> import pytz

    # Create a DataFrame with missing values and timezone-aware index
    >>> dates = pd.date_range(start='2024-01-01', periods=5, freq='1h', tz='UTC')
    >>> df = pd.DataFrame({
    ...     'temperature__°C': [20.0, np.nan, np.nan, 22.0, 23.0],
    ...     'pressure__Pa': [1013.0, np.nan, 1015.0, np.nan, 1014.0]
    ... }, index=dates)

    # Linear interpolation of all missing values
    >>> interpolator = Interpolate(method='linear')
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
      * Linear interpolation is simple but may not capture complex patterns
      * Cubic interpolation provides smoother curves but may overshoot

    Returns
    -------
    pd.DataFrame
        A DataFrame with missing values interpolated according to the specified parameters.
        The output maintains the same structure and index as the input DataFrame.
    """

    def __init__(
        self,
        method: str = "linear",
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

    # Create a DataFrame with solar radiation data and timezone-aware index
    >>> dates = pd.date_range(start='2024-01-01', periods=3, freq='1h', tz='UTC')
    >>> df = pd.DataFrame({
    ...     'bni__W/m²__outdoor__meteo': [800, 900, 1000],  # Direct normal irradiance
    ...     'dhi__W/m²__outdoor__meteo': [200, 250, 300],  # Diffuse horizontal irradiance
    ...     'ghi__W/m²__outdoor__meteo': [600, 700, 800]   # Global horizontal irradiance
    ... }, index=dates)

    # Project radiation on a south-facing surface tilted at 35 degrees
    >>> projector = ProjectSolarRadOnSurfaces(
    ...     bni_column_name='bni__W/m²__outdoor__meteo',
    ...     dhi_column_name='dhi__W/m²__outdoor__meteo',
    ...     ghi_column_name='ghi__W/m²__outdoor__meteo',
    ...     surface_azimuth_angles=180.0,  # South-facing
    ...     surface_tilt_angle=35.0,      # 35-degree tilt
    ...     surface_name='south_facing_35deg',
    ...     data_bloc='SOLAR',
    ...     data_sub_bloc='ROOF'
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:04:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C__room1': [20, np.nan, np.nan, 23, 24],
    ...     'temp__°C__room2': [21, 22, 22, 22, 23],
    ...     'humid__%__room1': [45, np.nan, 47, np.nan, 49],
    ...     'humid__%__room2': [46, 46, 48, 48, 50]
    ... }, index=dates)
    >>> # Fill gaps in room1 using room2 data
    >>> filler = FillOtherColumns(
    ...     columns_map={
    ...         'temp__°C__room1': 'temp__°C__room2',
    ...         'humid__%__room1': 'humid__%__room2'
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
    ...         'temp__°C__room1': 'temp__°C__room2',
    ...         'humid__%__room1': 'humid__%__room2'
    ...     },
    ...     drop_filling_columns=True
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:02:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C': [20, 21, 22],
    ...     'humid__%': [45, 50, 55],
    ...     'press__Pa': [1000, 1010, 1020]
    ... }, index=dates)
    >>> # Drop a single column
    >>> dropper = DropColumns(columns='temp__°C')
    >>> result = dropper.fit_transform(df)
    >>> print(result)
                           humid__%  press__Pa
    2024-01-01 00:00:00+00:00     45.0     1000.0
    2024-01-01 00:01:00+00:00     50.0     1010.0
    2024-01-01 00:02:00+00:00     55.0     1020.0
    >>> # Drop multiple columns
    >>> dropper_multi = DropColumns(columns=['temp__°C', 'humid__%'])
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
    ...     start='2024-01-01 00:00:00',
    ...     end='2024-01-01 00:02:00',
    ...     freq='1min'
    ... ).tz_localize('UTC')
    >>> df = pd.DataFrame({
    ...     'temp__°C__room1__north': [20, 21, 22],
    ...     'humid__%__room1__north': [45, 50, 55],
    ...     'press__Pa__room1__north': [1000, 1010, 1020]
    ... }, index=dates)
    >>> # Replace room1 with room2 and °C with K
    >>> replacer = ReplaceTag(tag_map={
    ...     'room1': 'room2',
    ...     '°C': 'K' # It is dumb, just for the exemple
    ... })
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
