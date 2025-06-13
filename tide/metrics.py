import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length


def nmbe(y_pred: pd.Series, y_true: pd.Series) -> float:
    """Normalized Mean Biased Error

    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    :return:
    Normalized Mean biased error as float
    """
    check_consistent_length(y_pred, y_true)
    return np.sum(y_pred - y_true) / np.sum(y_true) * 100


def rmse(y_pred: pd.Series, y_true: pd.Series, ddof: int = 1) -> float:
    """Compute the Root Mean Squared Error (RMSE) between predicted and true values.

    The square root of the average squared differences between predicted
    and actual values. This implementation includes an optional adjustment for the
    number of model parameters, which is useful when calculating an unbiased estimator
    of the variance and to comply with french regulation FD X30-148.

    Parameters
    ----------
    y_pred : pd.Series
        Estimated target values. Must be a 1D pandas Series with the same length as
        `y_true`.

    y_true : pd.Series
        Ground truth (correct) target values. Must be a 1D pandas Series with the same
        length as `y_pred`.

    ddof : int, default=1
        Number of model parameters used in the prediction. This is subtracted from the
        number of samples in the denominator to adjust for degrees of freedom, similar
        to the unbiased variance estimator in statistics.

    Returns
    -------
    float
        Root mean squared error between `y_pred` and `y_true`.

    Examples
    --------
    >>> import pandas as pd
    >>> from tide.metrics import rmse
    >>> y_true = pd.Series([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = pd.Series([2.5, 0.0, 2.0, 8.0])
    >>> rmse(y_pred, y_true)
    0.6123724356957945

    Notes
    -----
    - The RMSE is calculated as:
        sqrt( sum((y_pred - y_true)^2) / (n_samples - ddof) )
    - `ddof` should reflect the number of model parameters fitted to the data.
      For standard RMSE, set `n_parameters=0`.
    - This function assumes that `y_pred` and `y_true` are aligned (i.e., same index).
      If using pandas Series with different indices, align them explicitly or convert
      to NumPy arrays before passing.
    - If `ddof >= len(y_true)`, a ValueError is raised.
    """
    check_consistent_length(y_pred, y_true)

    n = y_true.shape[0]
    if n <= ddof:
        raise ValueError(
            "Number of samples must be greater than number of model parameters."
        )

    err = y_pred.to_numpy() - y_true.to_numpy()
    return np.sqrt(np.sum(err**2) / (n - ddof))


def cv_rmse(y_pred: pd.Series, y_true: pd.Series, ddof: int = 1) -> float:
    """
    Compute the Coefficient of Variation of the Root Mean Squared Error (CV(RMSE)).

    CV(RMSE) is a normalized version of RMSE, typically expressed as a percentage.
    It represents the RMSE as a proportion of the mean of the true values.

    Parameters
    ----------
    y_pred : pd.Series
        Estimated target values. Must be a 1D pandas Series with the same length as
        `y_true`.

    y_true : pd.Series
        Ground truth (correct) target values. Must be a 1D pandas Series with the
        same length as `y_pred`.

    ddof : int, default=1
        Degrees of freedom adjustment for the RMSE calculation. This value is subtracted
        from the number of samples when calculating the mean squared error denominator.
        For standard RMSE (not adjusted), set `ddof=0`.
        It is meant to comply with french regulation FD X30-148.

    Returns
    -------
    float
        Coefficient of variation of the RMSE, expressed as a percentage.

    Examples
    --------
    >>> import pandas as pd
    >>> from mymetrics import cv_rmse
    >>> y_true = pd.Series([100, 102, 98, 101])
    >>> y_pred = pd.Series([98, 100, 95, 99])
    >>> cv_rmse(y_pred, y_true)
    2.777...

    Notes
    -----
    - Expressing RMSE as a percentage of the mean allows for scale-independent
      performance comparison across different datasets or target magnitudes.
    - If the mean of `y_true` is 0, a ZeroDivisionError will be raised.
    - `y_pred` and `y_true` must be aligned and of equal length.
    - This implementation is commonly used in energy modeling
    (e.g., ASHRAE Guideline 14).

    Raises
    ------
    ValueError
        If number of samples is less than or equal to `ddof`.

    ZeroDivisionError
        If the mean of `y_true` is zero.
    """
    check_consistent_length(y_pred, y_true)

    mean_true = np.mean(y_true)
    if mean_true == 0:
        raise ZeroDivisionError("Mean of y_true is zero, CV(RMSE) is undefined.")

    return (1 / mean_true) * rmse(y_pred, y_true, ddof) * 100
