import datetime as dt

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tide.utils import (
    check_and_return_dt_index_df,
    parse_request_to_col_names,
    data_columns_to_tree,
    get_data_level_values,
    get_data_blocks,
    get_outer_timestamps,
)


def get_cols_axis_maps_and_labels(
    columns: pd.Index | list[str],
    y_axis_level: str = None,
    y_tag_list: list[str] = None,
):
    """
    Map data columns to y-axis identifiers for plotly.
    It assigns each column in the dataset to a specific y-axis
    (e.g., "y", "y2", "y3", etc.). The mapping is based on Tide tags.

    Args:
        columns (pd.Index | list[str]):
            The list or index of column names to be mapped to axes.
        y_axis_level (str, optional):
            The level of the data hierarchy to use for determining the y-axis tags.
            Defaults to "unit" if not provided. name__unit__bloc__sub_bloc
        y_tag_list (list[str], optional):
            A pre-specified list of y-axis tags to use for mapping. If provided,
            it overrides the automatic determination of tags based on `y_axis_level`.

    Returns:
        dict[str, str]:
            A dictionary mapping each column name to its respective y-axis identifier.
            The keys are column names, and the values are strings like "y", "y2", etc.

    Raises:
        ValueError: If `columns` is not a list or Pandas Index.

    Notes:
        - If `y_tag_list` is provided, it is used directly; otherwise, the function
          generates the tags based on the column hierarchy and `y_axis_level`.
        - The primary y-axis is labeled as "y", with subsequent axes labeled as "y2",
          "y3", etc.
    """
    columns = list(columns) if isinstance(columns, pd.Index) else columns

    if y_tag_list:
        y_tags = y_tag_list
    else:
        root = data_columns_to_tree(columns)
        if root.max_depth >= 3:
            level = y_axis_level if y_axis_level else "unit"
            y_tags = get_data_level_values(root, level)
        else:
            return {col: {"yaxis": "y"} for col in columns}, {"y": columns}, columns

    col_axes_map = {}
    axes_col_map = {}
    for i, tag in enumerate(y_tags):
        selected_cols = parse_request_to_col_names(columns, tag)
        axes_col_map["y" if i == 0 else f"y{i + 1}"] = selected_cols
        for col in selected_cols:
            col_axes_map[col] = {"yaxis": "y"} if i == 0 else {"yaxis": f"y{i + 1}"}

    return col_axes_map, axes_col_map, y_tags


def get_yaxis_min_max(
    data: pd.DataFrame | pd.Series,
    y_axis_level: str = None,
    y_tag_list: list[str] = None,
):
    data = check_and_return_dt_index_df(data)
    _, axes_col_map, _ = get_cols_axis_maps_and_labels(
        data.columns, y_axis_level, y_tag_list
    )

    return {
        ax: (
            float(data[axes_col_map[ax]].min().min()),
            float(data[axes_col_map[ax]].max().max()),
        )
        for ax in axes_col_map.keys()
    }


def get_gap_scatter_dict(
    ts: pd.Series,
    y_axis_min_max,
    col_axes_map,
    lower_td: str | pd.Timedelta | dt.timedelta = None,
    rgb: tuple[int, int, int] = (102, 102, 102),
    alpha: float = 0.5,
):
    """
    Generate a list of dictionaries defining surfaces plots for
    visualizing gaps.
    Formats data for plotly


    Parameters:
    -----------
    ts : pd.Series
        The time series containing data and potential gaps (nan values).
    y_axis_min_max : dict
        A mapping of y-axis names to their corresponding (min, max) values.
        Used to define the vertical extent of the gap regions.
    col_axes_map : dict
        A mapping of column names to axis configurations. Each column
        maps to a dictionary containing plot-related details, including the y-axis name.
    lower_td : str | pd.Timedelta | dt.timedelta, optional
        The lower time delta threshold for identifying gaps. Only gaps larger than this
        threshold are considered. Can be a string (e.g., `"1H"`), a `pd.Timedelta`, or
        a `datetime.timedelta`. Defaults to `None`, meaning no minimum threshold is
        applied.
    rgb : tuple[float, float, float], optional
        RGB values defining the color of the shaded gap regions. Each component should
        be between 0 and 255. Defaults to `(102, 102, 102)` (a gray color).
    alpha : float, optional
        The transparency level of the shaded gap regions, ranging from 0 (completely
        transparent) to 1 (completely opaque). Defaults to `0.5`.

    """
    y_axis = col_axes_map[ts.name]["yaxis"]
    y_min, y_max = y_axis_min_max[y_axis]

    gap_list = get_data_blocks(
        ts,
        is_null=True,
        lower_td_threshold=lower_td,
        return_combination=False,
    )[ts.name]

    gap_dict_list = []
    for gp in gap_list:
        gap_start, gap_end = get_outer_timestamps(gp, ts.index)
        gap_dict_list.append(
            dict(
                x=[gap_start, gap_start, gap_end, gap_end],
                y=[y_min, y_max, y_max, y_min],
                mode="none",
                fill="toself",
                showlegend=False,
                fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})",
                yaxis=y_axis,
            )
        )

    return gap_dict_list


def plot_gaps_heatmap(
    data: pd.Series | pd.DataFrame,
    time_step: str | pd.Timedelta | dt.timedelta = None,
    title: str = None,
):
    """
    Generates a heatmap that visualizes data availability over time as a percentage.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time-indexed data to analyze for gaps. Non-NaN values are considered
        as available data, and NaN values represent gaps. Each column
        (if DataFrame) will be treated independently.
    time_step : str, pd.Timedelta, or datetime.timedelta, optional
        The resampling frequency for aggregating data availability.
         If specified, the availability is calculated as the mean percentage within
         each time interval.
        Examples: '1d' (daily), 'h' (hourly), or pd.Timedelta('15min')
        for 15-minute intervals.
    title : str, optional
        The title for the heatmap.
    """

    data = check_and_return_dt_index_df(data)
    zero_one_df = (~data.isna()).astype(int) * 100
    if time_step is not None:
        zero_one_df = zero_one_df.resample(time_step).mean()

    fig = px.imshow(
        zero_one_df.T,
        title=title,
        color_continuous_scale="RdBu",
        origin="lower",
        labels={"x": ""},
    )

    fig.update_layout(coloraxis_colorbar=dict(title="% of available data"))

    return fig


def add_multi_axis_scatter(
    fig: go.Figure,
    data: pd.Series | pd.DataFrame,
    title: str = None,
    y_axis_labels: [str] = None,
    y_axis_dict: dict[str, str] = None,
    mode_dict: dict[str, str] = None,
    axis_space: float = 0.03,
    y_title_standoff: int | float = 5,
    markers_opacity: float = 0.5,
    lines_width: float = 2.0,
):
    """
    Generates a Plotly line plot with support for multiple y-axes.

    This function plots time series data with one or more y-axes.
    Each column in the input `data` can be assigned to a specific y-axis.
    Primary y-axis labels are displayed on the left, while additional y-axes
    are positioned on the right with configurable spacing.

    Parameters:
    ----------
    data : pd.Series | pd.DataFrame
        The input time series data to be plotted. Each column will be represented
        by a separate line.
    title : str, optional
        Title of the plot. Defaults to "Gaps plot".
    y_axis_labels : list[str], optional
        List of labels for the y-axes. The first label corresponds to the left y-axis;
        additional labels correspond to the right y-axes in the order they appear.
    axis_space : float, optional
        Space between consecutive right y-axes. Defaults to 0.03.
    y_axis_dict : dict[str, str], optional
        Dictionary mapping each column in `data` to a y-axis. Keys are column names,
         values are y-axis identifiers (e.g., "y", "y2", "y3").
         Defaults to assigning all columns to "y".
    mode_dict : dict[str, str], optional
        Dictionary mapping each column in `data` to a Plotly mode
        (e.g., "lines", "markers"). If not specified, defaults to "lines" for
        all columns.
    y_title_standoff : int | float, optional
        Distance of the y-axis titles from the axis line. A smaller value moves
        titles closer to the axis. Defaults to 5.
    markers_opacity : float, optional
        Opacity of the markers in the plot. Ranges from 0 to 1. Defaults to 0.5.
    lines_width : float, optional
        Width of the lines in the plot. Defaults to 2.0.
    """

    data = check_and_return_dt_index_df(data)
    if y_axis_dict is None:
        y_axis_dict = {col: "y" for col in data.columns}

    for i, col in enumerate(data.columns):
        fig.add_scattergl(
            x=data.index,
            y=data[col],
            name=f"{col}",
            mode=mode_dict[col] if mode_dict is not None else None,
            line=dict(width=lines_width),
            marker=dict(opacity=markers_opacity),
            yaxis=y_axis_dict[col],
        )

    layout_dict = {
        "legend": dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
        "title": title,
        "yaxis": dict(
            title=y_axis_labels[0] if y_axis_labels is not None else None,
            side="left",
            title_standoff=y_title_standoff,
        ),
    }

    nb_right_y_axis = len(set(y_axis_dict.values())) - 1
    x_right_space = 1 - axis_space * nb_right_y_axis
    fig.update_xaxes(domain=(0, x_right_space))

    for i in range(nb_right_y_axis):
        layout_dict[f"yaxis{i + 2}"] = dict(
            title=y_axis_labels[1 + i] if y_axis_labels is not None else None,
            overlaying="y",
            side="right",
            position=x_right_space + i * axis_space,
            title_standoff=y_title_standoff,
        )

    fig.update_layout(layout_dict)

    return fig
