import pandas as pd
import numpy as np
import datetime as dt
from bigtree import dict_to_tree, levelordergroup_iter
from bigtree.node import node
from typing import TypeVar
import re

T = TypeVar("T", bound=node.Node)

# Default tag names for unit, bloc, sub_bloc
DEFAULT_TAGS = ["DIMENSIONLESS", "OTHER", "OTHER_SUB_BLOC"]

# Tree architecture depending on the number of level.
# From all the time series in the same group of DATA
# To 3 levels of tags unit__bloc_sub_bloc

LEVEL_FORMAT = {
    0: lambda pt: f"DATA__{pt[0]}",
    1: lambda pt: f"DATA__{pt[1]}__{pt[0]}",
    2: lambda pt: f"DATA__{pt[2]}__{pt[1]}__{pt[0]}",
    3: lambda pt: f"DATA__{pt[2]}__{pt[3]}__{pt[1]}__{pt[0]}",
}

LEVEL_NAME_MAP = {0: "name", 1: "unit", 2: "bloc", 3: "sub_bloc"}
NAME_LEVEL_MAP = {name: level for level, name in LEVEL_NAME_MAP.items()}

TREE_LEVEL_NAME_MAP = {
    5: {"name": 4, "unit": 3, "bloc": 1, "sub_bloc": 2},
    4: {"name": 3, "unit": 2, "bloc": 1},
    3: {"name": 2, "unit": 1},
    2: {"name": 1},
}


def get_tree_depth_from_level(tree_max_depth: int, level: int | str):
    level = LEVEL_NAME_MAP[level] if isinstance(level, int) else level
    if tree_max_depth not in TREE_LEVEL_NAME_MAP:
        raise ValueError(
            f"Unsupported root depth of {tree_max_depth}. Allowed depths are 2 to 5."
        )

    level_indices = TREE_LEVEL_NAME_MAP[tree_max_depth]

    if level not in level_indices:
        raise ValueError(
            f"Unknown level {level}. Allowed levels are{level_indices.keys()}"
        )

    return level_indices[level]


def get_data_level_values(data_root, level: int | str):
    """
    Return a list of string containing values of tag at specified level.
    Warning bloc, unit and sub_bloc level ar unique
    :param data_root: big tree root
    :param level: int or string corresponding to tag level
    :return: list of values
    """
    tree_level = get_tree_depth_from_level(data_root.max_depth, level)

    nodes = [
        [node.name for node in node_group]
        for node_group in levelordergroup_iter(data_root)
    ]

    selected_nodes = nodes[tree_level]

    if level in ["bloc", "unit", "sub_bloc"]:
        # Return list with no duplicates
        return list(dict.fromkeys(selected_nodes))
    else:
        return selected_nodes


def get_tags_max_level(data_columns: pd.Index | list[str]) -> int:
    """
    Returns max used tag level from data columns names
    :param data_columns: DataFrame columns holding time series names with tags
    """
    return max(len(col.split("__")) - 1 for col in data_columns)


def edit_tag_value_by_level(col_name: str, level: int | str, new_tag_name: str) -> str:
    parts = col_name.split("__")
    if level > len(parts) - 1:
        raise ValueError(
            f"Cannot edit tag name at level index {level}. Columns have only {len(parts)} tag levels."
        )
    parts[level] = new_tag_name
    return "__".join(parts)


class NamedList:
    def __init__(self, elements: list):
        self.elements = elements

    def __repr__(self):
        return self.elements.__repr__()

    def __getitem__(self, key: str | list[str] | slice):
        if isinstance(key, slice):
            start = self.elements.index(key.start) if key.start is not None else None
            stop = self.elements.index(key.stop) + 1 if key.stop is not None else None
            return self.elements[start:stop]
        elif isinstance(key, str):
            return [self.elements[self.elements.index(key)]]
        elif isinstance(key, list):
            return [elmt for elmt in key if elmt in self.elements]
        else:
            raise TypeError("Invalid key type")


def col_name_tag_enrichment(col_name: str, tag_levels: int) -> str:
    """
    Enriches a column name by adding default tags until it reaches the specified
    number of tag levels.

    This function takes an input column name that may already contain tags
    (separated by double underscores "__") and appends default tags as needed to
    reach the specified `tag_levels`. Default tags are sourced from `DEFAULT_TAGS`.
    The enriched column name is then formatted according to the level-specific
    format in `LEVEL_FORMAT`.

    :param col_name: str. The original column name, which may contain some or all
        required tags.
    :param tag_levels: int. The target number of tags to achieve in the enriched
        column name. If the existing tags are fewer than this number, default tags
        are added.
    :return: str. The enriched column name with the specified number of tags.
    """
    split_col = col_name.split("__")
    num_tags = len(split_col)
    pt = split_col + DEFAULT_TAGS[num_tags - 1 : 4]
    return LEVEL_FORMAT[tag_levels](pt)


def get_data_col_names_from_root(data_root):
    return [
        [node.get_attr("col_name") for node in node_group]
        for node_group in levelordergroup_iter(data_root)
    ][-1]


def parse_request_to_col_names(
    data_columns: pd.Index | list[str], request: str | pd.Index | list[str] = None
) -> list[str]:
    if request is None:
        return list(data_columns)

    elif isinstance(request, pd.Index) or isinstance(request, list):
        return [col for col in request if col in data_columns]

    else:
        request_parts = request.split("__")

        if not (1 <= len(request_parts) <= 4):
            raise ValueError(
                f"Request '{request}' is malformed. "
                f"Use 'name__unit__bloc__sub_bloc' format or a "
                f"combination of these tags."
            )

        full_tag_col_map = {
            col_name_tag_enrichment(col, get_tags_max_level(data_columns)): col
            for col in data_columns
        }

        def find_exact_match(search_str, target):
            pattern = rf"(?:^|__)(?:{re.escape(search_str)})(?:$|__)"
            match = re.search(pattern, target)
            return match is not None

        return [
            full_tag_col_map[augmented_col]
            for augmented_col in full_tag_col_map.keys()
            if all(find_exact_match(part, augmented_col) for part in request_parts)
        ]


def data_columns_to_tree(columns: pd.Index | list[str]) -> T:
    """
    Parses column names and organizes them in a hierarchical structure.
    Column names must follow the format: "name__unit__bloc__sub_bloc" with tags
    separated by "__". Supported tags are: name, unit, bloc, and sub_bloc.
    Tree depth is automatically determined from the greater number of tags in a
    column name.
    Tags are supposed to be written in the above order.
    If only one tag is given, and tree depth is 4, it will be considered as name
    and the remaining tags will be set to DIMENSIONLESS, OTHER, OTHER

    :param columns: DataFrame columns or list of strings containing names of measured
    data time series. Names should follow the "name__unit__bloc_sub_bloc"
    naming convention
    """
    tag_levels = get_tags_max_level(columns)

    if not 0 <= tag_levels <= 3:
        raise ValueError(
            f"Only up to 4 tags are allowed; found tag level {tag_levels}."
        )

    parsed_dict = {}
    for col in columns:
        parsed_dict[col_name_tag_enrichment(col, tag_levels)] = {"col_name": col}

    return dict_to_tree(parsed_dict, sep="__")


def check_and_return_dt_index_df(X: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if not (isinstance(X, pd.Series) or isinstance(X, pd.DataFrame)):
        raise ValueError(
            f"Invalid X data, was expected an instance of pandas Dataframe "
            f"or Pandas Series. Got {type(X)}"
        )
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X index is not a pandas DateTime index")

    if X.index.tz is None:
        raise ValueError("X index must be tz_localized")

    return X.to_frame() if isinstance(X, pd.Series) else X


def _lower_bound(series, bound, bound_inclusive: bool, inner: bool):
    ops = {
        (False, False): np.less,
        (False, True): np.greater,
        (True, False): np.less_equal,
        (True, True): np.greater_equal,
    }
    op = ops[(bound_inclusive, inner)]
    return op(series, bound)


def _upper_bound(series, bound, bound_inclusive: bool, inner: bool):
    ops = {
        (False, False): np.greater,
        (False, True): np.less,
        (True, False): np.greater_equal,
        (True, True): np.less_equal,
    }
    op = ops[(bound_inclusive, inner)]
    return op(series, bound)


def _get_series_bloc(
    date_series: pd.Series,
    is_null: bool = False,
    select_inner: bool = True,
    lower_td_threshold: str | dt.timedelta = None,
    upper_td_threshold: str | dt.timedelta = None,
    lower_threshold_inclusive: bool = True,
    upper_threshold_inclusive: bool = True,
):
    data = check_and_return_dt_index_df(date_series).squeeze()
    freq = get_idx_freq_delta_or_min_time_interval(data.index)
    # If data index has no frequency, a frequency based on minimum
    # timedelta is set.
    df = data.asfreq(freq)

    lower_td_threshold = (
        pd.Timedelta(lower_td_threshold)
        if isinstance(lower_td_threshold, str)
        else lower_td_threshold
    )
    upper_td_threshold = (
        pd.Timedelta(upper_td_threshold)
        if isinstance(upper_td_threshold, str)
        else upper_td_threshold
    )

    if not df.dtype == bool:
        filt = df.isnull() if is_null else ~df.isnull()
    else:
        filt = ~df if is_null else df

    if ~np.any(filt):
        return []

    idx = df.index[filt]
    time_diff = idx.to_series().diff()
    split_points = np.where(time_diff != pd.Timedelta(df.index.freq))[0][1:]
    consecutive_indices = np.split(idx, split_points)
    durations = np.array([idx[-1] - idx[0] + freq for idx in consecutive_indices])

    lower_mask = upper_mask = np.ones_like(durations, dtype=bool)

    # Left bound
    if lower_td_threshold is not None:
        lower_mask = _lower_bound(
            durations, lower_td_threshold, lower_threshold_inclusive, select_inner
        )

    # Right bound
    if upper_td_threshold is not None:
        upper_mask = _upper_bound(
            durations, upper_td_threshold, upper_threshold_inclusive, select_inner
        )

    if upper_td_threshold is None and lower_td_threshold is not None:
        upper_mask = lower_mask

    if lower_td_threshold is None and upper_td_threshold is not None:
        lower_mask = upper_mask

    mask = lower_mask & upper_mask if select_inner else lower_mask | upper_mask

    return [
        pd.DatetimeIndex(indices, freq=freq)
        for indices, keep in zip(consecutive_indices, mask)
        if keep
    ]


def get_blocks_lte_and_gte(
    data: pd.Series | pd.DataFrame,
    lte: str | dt.timedelta = None,
    gte: str | dt.timedelta = None,
    is_null: bool = False,
    return_combination: bool = False,
):
    """
    Get blocks of data ore gaps (nan) based on duration thresholds.

    Returns them in a dictionary as list of DateTimeIndex. The keys values are
    data columns (or name if data is a Series).


    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        The input data to be processed.
    lte : str or datetime.timedelta, optional
        The upper time threshold. Can be a string (e.g., '1h') or a timedelta object.
    gte : str or datetime.timedelta, optional
        The lower time threshold. Can be a string (e.g., '30min') or a timedelta object.
    is_null : bool, default False
        Whether to select blocks where the data is null.

    Notes:
    ------
    - If both `lte` and `gte` are provided, and `lte` is smaller than `gte`, they
    will be swapped. The function determines whether to select data within or outside
    the boundaries based on the order of thresholds.
    return_combination : bool, optional
        If True (default), a combination column is created that checks for NaNs
        across all columns in the DataFrame. Gaps in this combination column represent
        rows where NaNs are present in any of the columns.
    """

    lower_th, upper_th = lte, gte
    select_inner = False
    if lower_th is not None and upper_th is not None:
        if pd.to_timedelta(lower_th) > pd.to_timedelta(upper_th):
            lower_th, upper_th = upper_th, lower_th
            select_inner = True

    return get_data_blocks(
        data=data,
        is_null=is_null,
        lower_td_threshold=lower_th,
        upper_td_threshold=upper_th,
        select_inner=select_inner,
        return_combination=return_combination,
    )


def get_blocks_mask_lte_and_gte(
    data: pd.Series | pd.DataFrame,
    lte: str | dt.timedelta = None,
    gte: str | dt.timedelta = None,
    is_null: bool = False,
    return_combination: bool = False,
) -> pd.DataFrame:
    """
    Creates a boolean mask DataFrame indicating the location of data blocks or gaps.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The input time series data with a DateTime index
    lte : str or timedelta, optional
        The minimum duration threshold
    gte : str or timedelta, optional
        The maximum duration threshold
    is_null : bool, default False
        Whether to find NaN blocks (True) or valid data blocks (False)
    return_combination : bool, optional
        If True (default), a combination column is created that checks for NaNs
        across all columns in the DataFrame. Gaps in this combination column represent
        rows where NaNs are present in any of the columns.

    Returns
    -------
    pd.DataFrame
        Boolean mask DataFrame with same index as input data and columns
        corresponding to the input data columns. True values indicate
        the presence of a block matching the criteria.
    """
    gaps_dict = get_blocks_lte_and_gte(data, lte, gte, is_null, return_combination)

    mask_data = {}
    for col, idx_list in gaps_dict.items():
        if idx_list:
            combined_idx = pd.concat([idx.to_series() for idx in idx_list]).index
            mask_data[col] = data.index.isin(combined_idx)
        else:
            mask_data[col] = np.zeros(data.shape[0], dtype=bool)

    return pd.DataFrame(mask_data, index=data.index)


def get_data_blocks(
    data: pd.Series | pd.DataFrame,
    is_null: bool = False,
    cols: str | list[str] = None,
    lower_td_threshold: str | dt.timedelta = None,
    upper_td_threshold: str | dt.timedelta = None,
    select_inner: bool = True,
    lower_threshold_inclusive: bool = True,
    upper_threshold_inclusive: bool = True,
    return_combination: bool = True,
):
    """
    Identifies groups of valid data if is_null = False, or groups of nan if
    is_null = True (gaps in measurements).

    The groups can be filtered using lower_dt_threshold or higher_dt_threshold.
    Their values can be included or not in the selection (lower_threshold_inclusive,
    upper_threshold_inclusive).

    Selection can be made inside the boundaries or outside using selec_inner

    Returns them in a dictionary as list of DateTimeIndex. The keys values are
    data columns (or name if data is a Series).

    The argument return_combination indicates if an additional key must be set to the
    dictionary to account for all data presence.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The input time series data with a DateTime index. NaN values are
        considered gaps.
    is_null : Bool, default False
        Whether to return groups with valid data, or groups of Nan values
        (is_null = True)
    cols : str or list[str], optional
        Columns to analyze. If None, uses all columns.
    select_inner : bool, default True
        If True, select groups within thresholds. If False, select groups outside thresholds.
    lower_td_threshold : str or timedelta, optional
        The minimum duration of a period for it to be considered valid.
        Can be passed as a string (e.g., '1d' for one day) or a `timedelta`.
        If None, no threshold is applied, NaN values are considered gaps.
    upper_td_threshold : str or timedelta, optional
        The maximum duration of a period for it to be considered valid.
        Can be passed as a string (e.g., '1d' for one day) or a `timedelta`.
        If None, no threshold is applied, NaN values are considered gaps.
    lower_threshold_inclusive : bool, optional
        Include the gaps of exactly lower_td_threshold duration
    upper_threshold_inclusive : bool, optional
        Include the gaps of exactly upper_td_threshold duration
    return_combination : bool, optional
        If True (default), a combination column is created that checks for NaNs
        across all columns in the DataFrame. Gaps in this combination column represent
        rows where NaNs are present in any of the columns.

    Returns
    -------
    dict[str, list[pd.DatetimeIndex]]
        A dictionary where the keys are the column names (or "combination" if
        `return_combination` is True) and the values are lists of `DatetimeIndex`
        objects.
        Each `DatetimeIndex` represents a group of one or several consecutive
        timestamps where the values in the corresponding column were NaN and
        exceeded the gap threshold.
    """
    data = check_and_return_dt_index_df(data)
    cols = ensure_list(cols) or list(data.columns)

    # Process each column
    idx_dict = {
        col: _get_series_bloc(
            data[col],
            is_null,
            select_inner,
            lower_td_threshold,
            upper_td_threshold,
            lower_threshold_inclusive,
            upper_threshold_inclusive,
        )
        for col in cols
    }

    if return_combination:
        idx_dict["combination"] = _get_series_bloc(
            ~data.isnull().any(axis=1),
            is_null,
            select_inner,
            lower_td_threshold,
            upper_td_threshold,
            lower_threshold_inclusive,
            upper_threshold_inclusive,
        )

    return idx_dict


def get_idx_freq_delta_or_min_time_interval(dt_idx: pd.DatetimeIndex):
    freq = dt_idx.inferred_freq
    if freq:
        freq = pd.to_timedelta("1" + freq) if freq.isalpha() else pd.to_timedelta(freq)
    else:
        freq = dt_idx.to_frame().diff().min().iloc[0]

    return freq


def get_outer_timestamps(idx: pd.DatetimeIndex, ref_index: pd.DatetimeIndex):
    try:
        out_start = ref_index[ref_index < idx[0]][-1]
    except IndexError:
        out_start = ref_index[0]

    try:
        out_end = ref_index[ref_index > idx[-1]][0]
    except IndexError:
        out_end = ref_index[-1]

    return out_start, out_end


def timedelta_to_int(td: int | str | dt.timedelta, df):
    if isinstance(td, int):
        return td
    else:
        if isinstance(td, str):
            td = pd.to_timedelta(td)
        return abs(int(td / df.index.freq))


def validate_odd_param(param_name, param_value):
    if isinstance(param_value, int) and param_value % 2 == 0:
        raise ValueError(
            f"{param_name}={param_value} is not valid, it must be an odd number"
        )


def process_stl_odd_args(param_name, X, stl_kwargs):
    param_value = stl_kwargs[param_name]
    if isinstance(param_value, int):
        # Is odd already check at init in case of int
        stl_kwargs[param_name] = param_value
    elif param_value is not None:
        processed_value = timedelta_to_int(param_value, X)
        if processed_value % 2 == 0:
            processed_value += 1  # Ensure the value is odd
        stl_kwargs[param_name] = processed_value


def ensure_list(item):
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
