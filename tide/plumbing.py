import datetime as dt

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from tide.utils import (
    tide_request,
    check_and_return_dt_index_df,
    data_columns_to_tree,
    get_data_level_values,
    get_tree_depth_from_level,
    NamedList,
    get_blocks_lte_and_gte,
    get_blocks_mask_lte_and_gte,
)
from tide.plot import (
    plot_gaps_heatmap,
    get_cols_axis_maps_and_labels,
    get_gap_scatter_dict,
    get_yaxis_min_max,
)
import tide.processing as pc


def _dummy_df(columns, tz):
    return pd.DataFrame(
        data=np.ones((2, len(columns))),
        columns=columns,
        index=pd.date_range("2009", freq="h", periods=2, tz=tz),
    )


def _get_pipe_from_proc_list(
    data_columns: pd.Index | list[str],
    proc_list: list,
    tz: str | dt.timezone,
    verbose: bool = False,
) -> Pipeline:
    proc_units = [
        getattr(pc, proc[0])(
            *proc[1] if len(proc) > 1 and isinstance(proc[1], list) else (),
            **proc[1] if len(proc) > 1 and isinstance(proc[1], dict) else {},
        )
        for proc in proc_list
    ]
    pipe = make_pipeline(*proc_units, verbose=verbose)
    pipe.set_output(transform="pandas")
    pipe.fit(_dummy_df(data_columns, tz))
    return pipe


def _get_column_wise_transformer(
    proc_dict,
    data_columns: pd.Index | list[str],
    tz: str | dt.timezone,
    process_name: str = None,
    verbose: bool = False,
) -> ColumnTransformer | None:
    col_trans_list = []
    for req, proc_list in proc_dict.items():
        requested_col = tide_request(data_columns, req)
        if not requested_col:
            pass
        else:
            name = req.replace("__", "_")
            col_trans_list.append(
                (
                    f"{process_name}->{name}" if process_name is not None else name,
                    _get_pipe_from_proc_list(requested_col, proc_list, tz, verbose),
                    requested_col,
                )
            )

    if not col_trans_list:
        return None
    else:
        transformer = ColumnTransformer(
            col_trans_list,
            remainder="passthrough",
            verbose_feature_names_out=False,
            verbose=verbose,
        ).set_output(transform="pandas")
        transformer.fit(_dummy_df(data_columns, tz))
        return transformer


def get_pipeline_from_dict(
    data_columns: pd.Index | list[str],
    pipe_dict: dict = None,
    tz: str | dt.timezone = "UTC",
    verbose: bool = False,
):
    if pipe_dict is None:
        pipe = Pipeline([("Identity", pc.Identity())], verbose=verbose)
        return pipe.fit(_dummy_df(data_columns, "UTC"))
    else:
        steps_list = []
        step_columns = data_columns.copy()
        for step, op_conf in pipe_dict.items():
            if isinstance(op_conf, list):
                operation = _get_pipe_from_proc_list(step_columns, op_conf, tz, verbose)

            elif isinstance(op_conf, dict):
                operation = _get_column_wise_transformer(
                    op_conf, step_columns, tz, step, verbose
                )

            else:
                raise ValueError(f"{op_conf} is an invalid operation config")

            if operation is not None:
                steps_list.append((step, operation))
                step_columns = [str(feat) for feat in operation.get_feature_names_out()]

        return Pipeline(steps_list, verbose=verbose)


# ---------------------------------------------------------------------------
# Dash app helpers (used by Plumber.plot_dash)
# ---------------------------------------------------------------------------

_DASH_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def _node_to_dash_html(node, selected_cols, col_colors, is_data2=False):
    """Recursively convert a bigtree node to Dash HTML components."""
    from dash import dcc, html

    if node.is_leaf:
        col_name = getattr(node, "col_name", None)
        # fig_col is the name used in the figure (prefixed for data_2)
        fig_col = f"data_2->{col_name}" if is_data2 and col_name else (col_name or "")
        is_checked = (col_name in selected_cols) if col_name else False
        color = col_colors.get(col_name, _DASH_COLORS[0])
        label_style = {"fontStyle": "italic"} if is_data2 else {}
        return html.Div(
            [
                dcc.Checklist(
                    id={"type": "col-check", "index": fig_col},
                    options=[{"label": "", "value": fig_col}],
                    value=[fig_col] if is_checked else [],
                    style={"display": "inline-flex", "marginRight": "2px"},
                ),
                dcc.Input(
                    id={"type": "col-color", "index": fig_col},
                    type="color",
                    value=color,
                    style={
                        "width": "24px",
                        "height": "22px",
                        "padding": "0",
                        "border": "none",
                        "cursor": "pointer",
                        "marginRight": "6px",
                        "flexShrink": "0",
                    },
                ),
                html.Span(node.name, style=label_style),
            ],
            style={"display": "flex", "alignItems": "center", "padding": "3px 4px"},
        )
    else:
        children_html = [
            _node_to_dash_html(child, selected_cols, col_colors, is_data2)
            for child in node.children
        ]
        return html.Details(
            [
                html.Summary(
                    node.name,
                    style={
                        "cursor": "pointer",
                        "userSelect": "none",
                        "padding": "3px 0",
                        "fontWeight": "500",
                    },
                ),
                html.Div(children_html, style={"paddingLeft": "14px"}),
            ],
            open=True,
        )


def _build_dash_sidebar(
    data_1_cols, data_2_cols, col_colors_1, col_colors_2, selected_1, selected_2
):
    """Build the right-side sidebar with an expandable tree of time series."""
    from dash import html

    sidebar_items = []

    if list(data_1_cols):
        root_1 = data_columns_to_tree(list(data_1_cols))
        for child in root_1.children:
            sidebar_items.append(
                _node_to_dash_html(
                    child,
                    selected_cols=selected_1,
                    col_colors=col_colors_1,
                    is_data2=False,
                )
            )

    if list(data_2_cols):
        orig_data_2_cols = [c.replace("data_2->", "") for c in data_2_cols]
        root_2 = data_columns_to_tree(orig_data_2_cols)
        sidebar_items.append(html.Hr(style={"margin": "8px 0", "borderColor": "#ddd"}))
        sidebar_items.append(
            html.Div(
                "Steps 2",
                style={
                    "fontWeight": "600",
                    "padding": "4px 4px",
                    "fontSize": "0.85em",
                    "color": "#555",
                    "letterSpacing": "0.05em",
                    "textTransform": "uppercase",
                },
            )
        )
        for child in root_2.children:
            sidebar_items.append(
                _node_to_dash_html(
                    child,
                    selected_cols=selected_2,
                    col_colors=col_colors_2,
                    is_data2=True,
                )
            )

    return html.Div(sidebar_items, style={"padding": "8px", "fontSize": "0.85em"})


class Plumber:
    """A powerful class for managing and transforming time series data through configurable processing pipelines.

    The Plumber class is the core component of the Tide library, providing a comprehensive interface for:
    - Managing time series data with hierarchical column naming (name__unit__bloc__sub_bloc)
    - Creating and executing data processing pipelines with column-wise transformations
    - Analyzing and visualizing data gaps and quality
    - Plotting time series with customizable multi-axis layouts

    The class uses a tree structure to organize data columns based on their tags, allowing for:
    - Flexible data selection using tag-based queries
    - Hierarchical organization of data by unit, bloc, and sub-bloc
    - Automatic handling of data transformations at different steps

    Parameters
    ----------
    data : pd.Series or pd.DataFrame, optional
        Input time series data. Must have a datetime index with timezone information.
    pipe_dict : dict, optional
        Pipeline configuration dictionary. Each key represents a processing step
        and contains either:
        - A list of transformations to apply to all columns
        - A dictionary mapping column tags to specific transformations

    Attributes
    ----------
    data : pd.DataFrame
        The input time series data with datetime index
    root : Node
        Root node of the tree structure organizing column names
    pipe_dict : dict
        Configuration dictionary defining the processing pipeline steps

    Examples
    --------
    >>> from tide.plumbing import Plumber
    >>> import pandas as pd
    >>> # Create sample data with hierarchical column names
    >>> data = pd.DataFrame(
    ...     {
    ...         "temp__°C__zone1": [20, 21, np.nan, 23],
    ...         "humid__%HR__zone1": [50, 55, 60, np.nan],
    ...         "power__kW__hvac": [1.5, 1.8, 1.6, 1.7],
    ...     },
    ...     index=pd.date_range("2023", freq="h", periods=4, tz="UTC"),
    ... )
    >>> # Define pipeline configuration
    >>> pipe_dict = {
    ...     "pre_processing": {
    ...         "°C": [["ReplaceThreshold", {"upper": 25}]],
    ...         "%HR": [["ReplaceThreshold", {"upper": 100}]],
    ...     },
    ...     "common": [["Interpolate", ["linear"]]],
    ... }
    >>> # Initialize and process data
    >>> plumber = Plumber(data, pipe_dict)
    >>> corrected = plumber.get_corrected_data()
    >>> # Analyze gaps
    >>> gaps = plumber.get_gaps_description()
    >>> # Visualize data
    >>> plumber.plot(y_axis_level="unit")

    Notes
    -----
    - Column names can use any combination of tags (name, unit, bloc, sub_bloc)
      separated by double underscores. Examples:
      - Simple: "temperature"
      - With unit: "temperature__°C"
      - Full: "temperature__°C__zone1__room1"
    - Input data must have a datetime index with timezone information
    - Pipeline steps can be applied globally or to specific column groups
    - Supports all transformations from the processing module
    - Provides comprehensive gap analysis and visualization tools
    - Uses plotly for interactive data visualization
    """

    def __init__(self, data: pd.Series | pd.DataFrame = None, pipe_dict: dict = None):
        self.data = check_and_return_dt_index_df(data) if data is not None else None
        self.root = data_columns_to_tree(data.columns) if data is not None else None
        self.pipe_dict = pipe_dict

    def __repr__(self):
        if self.data is not None:
            tree_depth = self.root.max_depth
            tag_levels = ["name", "unit", "bloc", "sub_bloc"]
            rep_str = "tide.plumbing.Plumber object \n"
            rep_str += f"Number of tags : {tree_depth - 2} \n"
            for tag in range(1, tree_depth - 1):
                rep_str += f"=== {tag_levels[tag]} === \n"
                for lvl_name in get_data_level_values(self.root, tag_levels[tag]):
                    rep_str += f"{lvl_name}\n"
                rep_str += "\n"
            return rep_str
        else:
            return super().__repr__()

    def show(
        self,
        select: str | pd.Index | list[str] = None,
        steps: None | str | list[str] | slice = slice(None),
        depth_level: int | str = None,
    ):
        """Display the tree structure of selected data columns at selected steps for
        a given depth level.

        Parameters
        ----------
        select : str or pd.Index or list[str], optional
            Data selection using tide's tag system
        steps : None or str or list[str] or slice, default slice(None)
            Pipeline steps to apply before showing the tree
        depth_level : int or str, optional
            Maximum depth level to display in the tree
        """
        pipe = self.get_pipeline(select=select, steps=steps)
        loc_tree = data_columns_to_tree(pipe.get_feature_names_out())
        if depth_level is not None:
            depth_level = get_tree_depth_from_level(loc_tree.max_depth, depth_level)
        loc_tree.show(max_depth=depth_level)

    def get_gaps_description(
        self,
        select: str | pd.Index | list[str] = None,
        steps: None | str | list[str] | slice = slice(None),
        verbose: bool = False,
        gaps_lte: str | pd.Timedelta | dt.timedelta = None,
        gaps_gte: str | pd.Timedelta | dt.timedelta = None,
        return_combination: bool = True,
    ) -> pd.DataFrame:
        """
        Get a statistical description of gaps durations in the data.

        Parameters
        ----------
        select : str or pd.Index or list[str], optional
            Data selection using tide's tag system
        steps : None or str or list[str] or slice, default slice(None)
            Pipeline steps to apply before analyzing gaps
        verbose : bool, default False
            Whether to print information about pipeline steps
        gaps_lte : str or pd.Timedelta or dt.timedelta, optional
            Upper threshold for gap duration
        gaps_gte : str or pd.Timedelta or dt.timedelta, optional
            Lower threshold for gap duration
        return_combination : bool, default True
            Whether to include statistics for gaps aggregation.
            Useful to get statistics when all data are available.

        Returns
        -------
        pd.DataFrame
            DataFrame containing statistics about gap durations for each column.
            Statistics include:
            - data_presence_%: percentage of non-gap data points
            - count: number of gaps
            - mean: average gap duration
            - std: standard deviation of gap durations
            - min: shortest gap
            - 25%: first quartile
            - 50%: median
            - 75%: third quartile
            - max: longest gap
            Empty DataFrame if no gaps are found.
        """
        data = self.get_corrected_data(select, steps=steps, verbose=verbose)

        # Get gaps and calculate durations
        gaps_dict = get_blocks_lte_and_gte(
            data=data,
            lte=gaps_lte,
            gte=gaps_gte,
            is_null=True,
            return_combination=return_combination,
        )

        gap_durations = {}
        for col, gaps_list in gaps_dict.items():
            if not gaps_list:
                continue

            durations = []
            for gap in gaps_list:
                if len(gap) > 1:
                    durations.append(gap[-1] - gap[0] + pd.to_timedelta(gap.freq))
                else:
                    durations.append(pd.to_timedelta(gap.freq))

            if durations:
                gap_durations[col] = pd.Series(durations, name=col)

        if not gap_durations:
            return pd.DataFrame()

        stats_df = pd.concat([ser.describe() for ser in gap_durations.values()], axis=1)

        gaps_mask = get_blocks_mask_lte_and_gte(
            data=data,
            lte=gaps_lte,
            gte=gaps_gte,
            is_null=True,
            return_combination=return_combination,
        )

        presence_percentages = (1 - gaps_mask.mean()) * 100

        stats_df.loc["data_presence_%"] = presence_percentages[stats_df.columns]
        row_order = ["data_presence_%"] + [
            idx for idx in stats_df.index if idx != "data_presence_%"
        ]
        return stats_df.reindex(row_order)

    def set_data(self, data: pd.Series | pd.DataFrame):
        """Set new data for the Plumber instance.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            New time series data to process. Must have a datetime index with timezone information.
        """
        self.data = check_and_return_dt_index_df(data)
        self.root = data_columns_to_tree(data.columns)

    def select(
        self,
        select: str | pd.Index | list[str] = None,
    ):
        """Select columns based on tags.

        Parameters
        ----------
        select : str or pd.Index or list[str], optional
            Selection criteria using tide's tag system.
            Can be a unit (e.g., "°C"), location (e.g., "zone_1"),
            or any other tag in the column names.

        Returns
        -------
        pd.Index
            Selected column names
        """
        return tide_request(self.data, select)

    def get_pipeline(
        self,
        select: str | pd.Index | list[str] = None,
        steps: None | str | list[str] | slice = slice(None),
        verbose: bool = False,
    ) -> Pipeline:
        """Create a scikit-learn pipeline from the configuration.

        This method builds a scikit-learn Pipeline object based on the current configuration
        and selected data columns. The pipeline can be used to transform data according to
        the defined processing steps.

        Parameters
        ----------
        select : str or pd.Index or list[str], optional
            Data selection using tide's tag system. Can be:
            - A single tag (e.g., "°C" to select all temperature columns)
            - A full column name pattern (e.g., "temp__°C__zone1")
            If None, selects all columns.

        steps : None or str or list[str] or slice, default slice(None)
            Pipeline steps to include. Can be:
            - A single step name (e.g., "pre_processing")
            - A list of step names (e.g., ["pre_processing", "common"])
            - A slice object (e.g., slice("pre_processing", "common"))
            - None to return an Identity transformer
            - slice(None) to include all steps

        verbose : bool, default False
            Whether to print information about pipeline steps during creation

        Returns
        -------
        Pipeline
            A scikit-learn Pipeline object configured with the selected steps and columns.
            The pipeline will transform the data according to the processing steps defined
            in pipe_dict.

        Raises
        ------
        ValueError
            If data is not set (self.data is None)

        Examples
        --------
        >>> from tide.plumbing import Plumber
        >>> import pandas as pd
        >>> # Create sample data
        >>> data = pd.DataFrame(
        ...     {
        ...         "temp__°C__zone1": [20, 21, np.nan, 23],
        ...         "humid__%HR__zone1": [50, 55, 60, np.nan],
        ...         "power__kW__hvac": [1.5, 1.8, 1.6, 1.7],
        ...     },
        ...     index=pd.date_range("2023", freq="h", periods=4, tz="UTC"),
        ... )
        >>> # Define pipeline configuration
        >>> pipe_dict = {
        ...     "pre_processing": {
        ...         "°C": [["ReplaceThreshold", {"upper": 25}]],
        ...         "%HR": [["ReplaceThreshold", {"upper": 100}]],
        ...     },
        ...     "common": [["Interpolate", ["linear"]]],
        ... }
        >>> # Initialize Plumber
        >>> plumber = Plumber(data, pipe_dict)
        >>> # Get pipeline for temperature columns only
        >>> temp_pipe = plumber.get_pipeline(select="°C")
        >>> # Get pipeline for all columns with only pre-processing step
        >>> pre_pipe = plumber.get_pipeline(steps="pre_processing")
        >>> # Get pipeline for specific columns and steps
        >>> custom_pipe = plumber.get_pipeline(
        ...     select=["temp__°C__zone1", "power__kW__hvac"],
        ...     steps=["pre_processing", "common"],
        ... )
        """
        if self.data is None:
            raise ValueError("data is required to build a pipeline")
        selection = tide_request(self.data, select)
        if steps is None or self.pipe_dict is None:
            dict_to_pipe = None
        else:
            pipe_named_keys = NamedList(list(self.pipe_dict.keys()))
            selected_steps = pipe_named_keys[steps]
            dict_to_pipe = {key: self.pipe_dict[key] for key in selected_steps}

        return get_pipeline_from_dict(
            selection, dict_to_pipe, self.data.index.tz, verbose
        )

    def get_corrected_data(
        self,
        select: str | pd.Index | list[str] = None,
        start: str | dt.datetime | pd.Timestamp = None,
        stop: str | dt.datetime | pd.Timestamp = None,
        steps: None | str | list[str] | slice = slice(None),
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Apply pipeline transformations to selected data.

        This method applies the configured processing pipeline to the selected data columns
        within the specified time range. It returns a new DataFrame with the transformed data.

        Parameters
        ----------
        select : str or pd.Index or list[str], optional
            Data selection using tide's tag system. Can be:
            - A single tag (e.g., "°C" to select all temperature columns)
            - A full column name pattern (e.g., "temp__°C__zone1")
            If None, selects all columns.

        start : str or datetime or Timestamp, optional
            Start time for data slice. Can be:
            - A string in ISO format (e.g., "2023-01-01")
            - A datetime object
            - A pandas Timestamp
            If None, uses the first timestamp in the data.

        stop : str or datetime or Timestamp, optional
            End time for data slice. Can be:
            - A string in ISO format (e.g., "2023-12-31")
            - A datetime object
            - A pandas Timestamp
            If None, uses the last timestamp in the data.

        steps : None or str or list[str] or slice, default slice(None)
            Pipeline steps to apply. Can be:
            - A single step name (e.g., "pre_processing")
            - A list of step names (e.g., ["pre_processing", "common"])
            - A slice object (e.g., slice("pre_processing", "common"))
            - None to return an Identity transformer
            - slice(None) to include all steps

        verbose : bool, default False
            Whether to print information about pipeline steps during processing

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            If data is not set (self.data is None)

        Examples
        --------
        >>> from tide.plumbing import Plumber
        >>> import pandas as pd
        >>> # Create sample data
        >>> data = pd.DataFrame(
        ...     {
        ...         "temp__°C__zone1": [20, 21, np.nan, 23],
        ...         "humid__%HR__zone1": [50, 55, 60, np.nan],
        ...         "power__kW__hvac": [1.5, 1.8, 1.6, 1.7],
        ...     },
        ...     index=pd.date_range("2023", freq="h", periods=4, tz="UTC"),
        ... )
        >>> # Define pipeline configuration
        >>> pipe_dict = {
        ...     "pre_processing": {
        ...         "°C": [["ReplaceThreshold", {"upper": 25}]],
        ...         "%HR": [["ReplaceThreshold", {"upper": 100}]],
        ...     },
        ...     "common": [["Interpolate", ["linear"]]],
        ... }
        >>> # Initialize Plumber
        >>> plumber = Plumber(data, pipe_dict)
        >>> # Get corrected data for temperature columns only
        >>> temp_data = plumber.get_corrected_data(select="°C")
        >>> # Get corrected data for a specific time range
        >>> time_slice = plumber.get_corrected_data(
        ...     start="2023-01-01T00:00:00", stop="2023-01-01T12:00:00"
        ... )
        >>> # Get corrected data with specific steps
        >>> pre_processed = plumber.get_corrected_data(
        ...     select=["temp__°C__zone1", "power__kW__hvac"], steps="pre_processing"
        ... )
        """
        if self.data is None:
            raise ValueError("Cannot get corrected data. data are missing")
        select = tide_request(self.data, select)
        data = self.data.loc[
            start or self.data.index[0] : stop or self.data.index[-1], select
        ].copy()

        return self.get_pipeline(select, steps, verbose).fit_transform(data)

    def plot_gaps_heatmap(
        self,
        select: str | pd.Index | list[str] = None,
        start: str | dt.datetime | pd.Timestamp = None,
        stop: str | dt.datetime | pd.Timestamp = None,
        steps: None | str | list[str] | slice = slice(None),
        time_step: str | pd.Timedelta | dt.timedelta = None,
        title: str = None,
        verbose: bool = False,
    ):
        """Create a heatmap visualization of data gaps.

        This method generates an interactive heatmap using plotly that shows the presence
        and distribution of data gaps across different columns and time periods. The heatmap
        helps identify patterns in missing data and potential data quality issues.

        Parameters
        ----------
        select : str or pd.Index or list[str], optional
            Data selection using tide's tag system. Can be:
            - A single tag (e.g., "°C" to select all temperature columns)
            - A full column name pattern (e.g., "temp__°C__zone1")
            If None, selects all columns.

        start : str or datetime or Timestamp, optional
            Start time for visualization. Can be:
            - A string in ISO format (e.g., "2023-01-01")
            - A datetime object
            - A pandas Timestamp
            If None, uses the first timestamp in the data.

        stop : str or datetime or Timestamp, optional
            End time for visualization. Can be:
            - A string in ISO format (e.g., "2023-12-31")
            - A datetime object
            - A pandas Timestamp
            If None, uses the last timestamp in the data.

        steps : None or str or list[str] or slice, default slice(None)
            Pipeline steps to apply before visualization. Can be:
            - A single step name (e.g., "pre_processing")
            - A list of step names (e.g., ["pre_processing", "common"])
            - A slice object (e.g., slice("pre_processing", "common"))
            - None to return an Identity transformer
            - slice(None) to include all steps

        time_step : str or Timedelta or timedelta, optional
            Time step for aggregating gaps. Can be:
            - A string (e.g., "1h", "1d", "1w")
            - A timedelta object
            - A pandas Timedelta
            If None, uses the original data frequency.

        title : str, optional
            Plot title. If None, uses a default title based on the data selection.

        verbose : bool, default False
            Whether to print information about pipeline steps during processing

        Returns
        -------
        go.Figure
            A plotly Figure object containing the heatmap with:
            - Rows representing different columns
            - Columns representing time periods
            - Colors indicating presence (white) or absence (colored) of data
            - Interactive features (zoom, pan, hover information)

        Examples
        --------
        >>> from tide.plumbing import Plumber
        >>> import pandas as pd
        >>> # Create sample data with gaps
        >>> data = pd.DataFrame(
        ...     {
        ...         "temp__°C__zone1": [20, np.nan, 23, np.nan, 25],
        ...         "humid__%HR__zone1": [50, 55, np.nan, 60, np.nan],
        ...         "power__kW__hvac": [1.5, 1.8, 1.6, np.nan, 1.7],
        ...     },
        ...     index=pd.date_range("2023", freq="h", periods=5, tz="UTC"),
        ... )
        >>> # Initialize Plumber
        >>> plumber = Plumber(data)
        >>> # Create heatmap for all columns
        >>> fig = plumber.plot_gaps_heatmap()
        >>> fig.show()
        >>> # Create heatmap for temperature data with daily aggregation
        >>> fig = plumber.plot_gaps_heatmap(
        ...     select="°C", time_step="1d", title="Temperature Data Gaps"
        ... )
        >>> fig.show()
        >>> # Create heatmap for specific time range
        >>> fig = plumber.plot_gaps_heatmap(
        ...     start="2023-01-01T00:00:00", stop="2023-01-01T12:00:00"
        ... )
        >>> fig.show()
        """
        data = self.get_corrected_data(select, start, stop, steps, verbose)
        return plot_gaps_heatmap(data, time_step=time_step, title=title)

    def plot(
        self,
        select: str | pd.Index | list[str] = None,
        start: str | dt.datetime | pd.Timestamp = None,
        stop: str | dt.datetime | pd.Timestamp = None,
        y_axis_level: str = None,
        y_tag_list: list[str] = None,
        steps: None | str | list[str] | slice = slice(None),
        data_mode: str = "lines",
        steps_2: None | str | list[str] | slice = None,
        data_2_mode: str = "markers",
        markers_opacity: float = 0.8,
        lines_width: float = 2.0,
        title: str = None,
        plot_gaps: bool = False,
        gaps_lower_td: str | pd.Timedelta | dt.timedelta = None,
        gaps_rgb: tuple[int, int, int] = (31, 73, 125),
        gaps_alpha: float = 0.5,
        plot_gaps_2: bool = False,
        gaps_2_lower_td: str | pd.Timedelta | dt.timedelta = None,
        gaps_2_rgb: tuple[int, int, int] = (254, 160, 34),
        gaps_2_alpha: float = 0.5,
        axis_space: float = 0.03,
        y_title_standoff: int | float = 5,
        verbose: bool = False,
        use_resampler: bool = False,
    ):
        """Create an interactive time series plot.

        This method generates a highly customizable interactive plot using plotly that can show:
        - Multiple time series with automatic different y-axes based on unit
        - Two different versions of the data (e.g., raw and processed)
        - Data gaps visualization with customizable colors and opacity
        - Custom styling and layout options

        Parameters
        ----------
        select : str or pd.Index or list[str], optional
            Data selection using tide's tag system. Can be:
            - A single tag (e.g., "°C" to select all temperature columns)
            - A full column name pattern (e.g., "temp__°C__zone1")
            If None, selects all columns.

        start : str or datetime or Timestamp, optional
            Start time for plot. Can be:
            - A string in ISO format (e.g., "2023-01-01")
            - A datetime object
            - A pandas Timestamp
            If None, uses the first timestamp in the data.

        stop : str or datetime or Timestamp, optional
            End time for plot. Can be:
            - A string in ISO format (e.g., "2023-12-31")
            - A datetime object
            - A pandas Timestamp
            If None, uses the last timestamp in the data.

        y_axis_level : str, optional
            Tag level to use for y-axis grouping. Can be:
            - "unit" to group by measurement unit
            - "bloc" to group by data bloc
            - "sub_bloc" to group by sub-bloc
            If None, uses a single y-axis for all data.

        y_tag_list : list[str], optional
            List of tags for custom y-axis ordering. The order of tags in this list
            determines the order of y-axes from left to right.

        steps : None or str or list[str] or slice, default slice(None)
            Pipeline steps to apply for main data. Can be:
            - A single step name (e.g., "pre_processing")
            - A list of step names (e.g., ["pre_processing", "common"])
            - A slice object (e.g., slice("pre_processing", "common"))
            - None to return an Identity transformer
            - slice(None) to include all steps

        data_mode : str, default "lines"
            Plot mode for main data. Can be:
            - "lines" for line plots
            - "markers" for scatter plots
            - "lines+markers" for combined line and marker plots

        steps_2 : None or str or list[str] or slice, optional
            Pipeline steps to apply for secondary data. Used to compare different
            processing steps or versions of the data.

        data_2_mode : str, default "markers"
            Plot mode for secondary data. Same options as data_mode.

        markers_opacity : float, default 0.8
            Opacity for markers (0.0 to 1.0)

        lines_width : float, default 2.0
            Width of plot lines in pixels

        title : str, optional
            Plot title. If None, uses a default title based on the data selection.

        plot_gaps : bool, default False
            Whether to highlight gaps in main data

        gaps_lower_td : str or Timedelta or timedelta, optional
            Minimum duration for gap highlighting. Can be:
            - A string (e.g., "1h", "1d")
            - A timedelta object
            - A pandas Timedelta

        gaps_rgb : tuple[int, int, int], default (31, 73, 125)
            RGB color for main data gaps (0-255 range)

        gaps_alpha : float, default 0.5
            Opacity for main data gaps (0.0 to 1.0)

        plot_gaps_2 : bool, default False
            Whether to highlight gaps in secondary data

        gaps_2_lower_td : str or Timedelta or timedelta, optional
            Minimum duration for secondary data gap highlighting

        gaps_2_rgb : tuple[int, int, int], default (254, 160, 34)
            RGB color for secondary data gaps (0-255 range)

        gaps_2_alpha : float, default 0.5
            Opacity for secondary data gaps (0.0 to 1.0)

        axis_space : float, default 0.03
            Space between multiple y-axes (0.0 to 1.0)

        y_title_standoff : int or float, default 5
            Distance between y-axis title and axis in pixels

        verbose : bool, default False
            Whether to print information about pipeline steps during processing

        use_resampler : bool, default False
            Whether to use plotly-resampler for dynamic data aggregation.
            Requires the optional dependency ``plotly-resampler``
            (``pip install python-tide[resampler]``).
            When enabled, the figure dynamically resamples data on zoom/pan,
            making it practical for large datasets (e.g. 1-minute resolution over a year).
            Dynamic resampling requires a live Python server to respond to zoom events —
            it does **not** work with a static ``fig.show()`` call.
            In a Jupyter environment (with an active kernel), returns a
            ``FigureWidgetResampler`` — display the figure by evaluating ``fig`` in a cell
            (do **not** call ``fig.show()``).
            In a non-Jupyter environment, returns a ``FigureResampler``; call
            ``fig.show_dash()`` to launch the interactive Dash server.

        Returns
        -------
        go.Figure or FigureResampler or FigureWidgetResampler
            A plotly Figure object containing the plot with:
            - Multiple y-axes if y_axis_level is specified
            - Interactive features (zoom, pan, hover information)
            - Legend with all series
            - Optional gap highlighting
            - Customizable styling
            - Dynamic resampling on zoom/pan if use_resampler=True

        Examples
        --------
        >>> from tide.plumbing import Plumber
        >>> import pandas as pd
        >>> # Create sample data
        >>> data = pd.DataFrame(
        ...     {
        ...         "temp__°C__zone1": [20, 21, np.nan, 23],
        ...         "humid__%HR__zone1": [50, 55, 60, np.nan],
        ...         "power__kW__hvac": [1.5, 1.8, 1.6, 1.7],
        ...     },
        ...     index=pd.date_range("2023", freq="h", periods=4, tz="UTC"),
        ... )
        >>> # Initialize Plumber
        >>> plumber = Plumber(data)
        >>> # Create basic plot with automatic y-axes
        >>> fig = plumber.plot(y_axis_level="unit")
        >>> fig.show()
        >>> # Create plot with custom styling and gap highlighting
        >>> fig = plumber.plot(
        ...     select=["temp__°C__zone1", "power__kW__hvac"],
        ...     data_mode="lines+markers",
        ...     plot_gaps=True,
        ...     gaps_lower_td="1h",
        ...     title="Temperature and Power Data",
        ... )
        >>> fig.show()
        >>> # Create plot comparing raw and processed data
        >>> fig = plumber.plot(
        ...     steps="pre_processing",
        ...     steps_2=None,
        ...     data_mode="lines",
        ...     data_2_mode="markers",
        ...     title="Raw vs Processed Data",
        ... )
        >>> fig.show()
        >>> # Use dynamic resampling for large datasets (e.g. 1-min data over 1 year)
        >>> # In a Jupyter notebook:
        >>> fig = plumber.plot(use_resampler=True)
        >>> fig  # displays as an interactive widget with on-the-fly resampling
        >>> # Outside Jupyter:
        >>> fig = plumber.plot(use_resampler=True)
        >>> fig.show_dash()  # launches a local Dash server for interactive resampling
        """
        # A bit dirty. Here we assume that if you ask a selection
        # that is not found in original data columns, it is because it
        # has not yet been computed (using ExpressionCombine processor
        # for example) So we just process the whole data hoping to find the result
        # after.
        select_corr = (
            self.data.columns if not tide_request(self.data, select) else select
        )

        data_1 = self.get_corrected_data(select_corr, start, stop, steps, verbose)
        if steps_2 is not None:
            data_2 = self.get_corrected_data(select_corr, start, stop, steps_2)
            data_2.columns = [f"data_2->{col}" for col in data_2.columns]
        else:
            data_2 = pd.DataFrame()

        cols = pd.concat([data_1, data_2], axis=1).columns
        col_axes_map, axes_col_map, y_labels = get_cols_axis_maps_and_labels(
            cols, y_axis_level, y_tag_list
        )
        conf_dict_list = []
        conf_dict_list.append({col: {"name": f"{col}"} for col in cols})
        conf_dict_list.append(col_axes_map)
        conf_dict_list.append(
            {col: {"mode": data_mode} for col in data_1}
            | {col: {"mode": data_2_mode} for col in data_2}
        )
        conf_dict_list.append({col: dict(line=dict(width=lines_width)) for col in cols})
        conf_dict_list.append(
            {col: dict(marker=dict(opacity=markers_opacity)) for col in cols}
        )

        scatter_config = {}

        for d in conf_dict_list:
            for key in d:
                scatter_config[key] = {**scatter_config.get(key, {}), **d[key]}

        if use_resampler:
            try:
                from plotly_resampler import FigureResampler, FigureWidgetResampler
            except ImportError:
                raise ImportError(
                    "plotly-resampler is required when use_resampler=True. "
                    "Install it with: pip install python-tide[resampler]"
                )
            try:
                ip = __import__("IPython").get_ipython()
                in_jupyter = ip is not None and hasattr(ip, "kernel")
            except (ImportError, AttributeError):
                in_jupyter = False

            if not in_jupyter:
                import warnings

                warnings.warn(
                    "Dynamic resampling requires a live Python server. "
                    "Call fig.show_dash() on the returned figure to enable zoom-triggered resampling.",
                    UserWarning,
                    stacklevel=2,
                )

            fig = (
                FigureWidgetResampler(go.Figure())
                if in_jupyter
                else FigureResampler(go.Figure())
            )
            for col in data_1:
                fig.add_trace(
                    go.Scattergl(**scatter_config[col]),
                    hf_x=data_1.index,
                    hf_y=data_1[col],
                )
            if steps_2 is not None:
                for col in data_2:
                    fig.add_trace(
                        go.Scattergl(**scatter_config[col]),
                        hf_x=data_2.index,
                        hf_y=data_2[col],
                    )
        else:
            fig = go.Figure()
            for col in data_1:
                fig.add_scattergl(x=data_1.index, y=data_1[col], **scatter_config[col])
            if steps_2 is not None:
                for col in data_2:
                    fig.add_scattergl(
                        x=data_2.index, y=data_2[col], **scatter_config[col]
                    )

        yaxis_min_max = get_yaxis_min_max(
            pd.concat([data_1, data_2], axis=1), y_axis_level, y_tag_list
        )

        def gap_dict_config(data, lower_td, rgb, alpha):
            gaps_list = []
            for col in data:
                col_configs = get_gap_scatter_dict(
                    data[col], yaxis_min_max, col_axes_map, lower_td, rgb, alpha
                )
                if col_configs:
                    gaps_list += col_configs
            return gaps_list

        gap_conf_list = []
        if plot_gaps:
            gap_conf_list += gap_dict_config(
                data_1, gaps_lower_td, gaps_rgb, gaps_alpha
            )

        if plot_gaps_2:
            gap_conf_list += gap_dict_config(
                data_2, gaps_2_lower_td, gaps_2_rgb, gaps_2_alpha
            )

        for gap in gap_conf_list:
            if use_resampler:
                fig.add_trace(go.Scattergl(**gap))
            else:
                fig.add_scattergl(**gap)

        layout_dict = {
            "legend": dict(
                orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5
            ),
            "title": title,
            "yaxis": dict(
                title=y_labels[0] if y_labels is not None else None,
                side="left",
                title_standoff=y_title_standoff,
            ),
        }

        nb_right_y_axis = len(y_labels) - 1
        x_right_space = 1 - axis_space * nb_right_y_axis
        fig.update_xaxes(domain=(0, x_right_space))

        for i in range(nb_right_y_axis):
            layout_dict[f"yaxis{i + 2}"] = dict(
                title=y_labels[1 + i] if y_labels is not None else None,
                overlaying="y",
                side="right",
                position=x_right_space + i * axis_space,
                title_standoff=y_title_standoff,
            )

        fig.update_layout(layout_dict)

        return fig

    def plot_dash(
        self,
        select: str | pd.Index | list[str] = None,
        start: str | dt.datetime | pd.Timestamp = None,
        stop: str | dt.datetime | pd.Timestamp = None,
        y_axis_level: str = None,
        y_tag_list: list[str] = None,
        steps: None | str | list[str] | slice = slice(None),
        data_mode: str = "lines",
        steps_2: None | str | list[str] | slice = None,
        data_2_mode: str = "markers",
        markers_opacity: float = 0.8,
        lines_width: float = 2.0,
        title: str = None,
        plot_gaps: bool = False,
        gaps_lower_td: str | pd.Timedelta | dt.timedelta = None,
        gaps_rgb: tuple[int, int, int] = (31, 73, 125),
        gaps_alpha: float = 0.5,
        plot_gaps_2: bool = False,
        gaps_2_lower_td: str | pd.Timedelta | dt.timedelta = None,
        gaps_2_rgb: tuple[int, int, int] = (254, 160, 34),
        gaps_2_alpha: float = 0.5,
        axis_space: float = 0.03,
        y_title_standoff: int | float = 5,
        verbose: bool = False,
        port: int = 8050,
    ):
        """Launch an interactive Dash application for time series exploration.

        Provides the same visualisation options as :meth:`plot` with two
        additions:

        * A collapsible sidebar listing all available columns with per-series
          visibility toggles and colour pickers.
        * When ``plotly-resampler`` is installed, traces are dynamically
          downsampled on zoom/pan so arbitrarily large datasets remain
          responsive.

        The server runs in a background daemon thread and the default browser
        opens automatically. It stays alive as long as the Python process runs.
        Calling this method multiple times on different ports is supported.

        Parameters
        ----------
        select : str or pd.Index or list[str], optional
            Columns pre-selected (visible) when the app opens. Accepts a tide
            tag query string, an explicit list of column names, or a
            ``pd.Index``. Defaults to all columns.
        start : str or datetime or Timestamp, optional
            Start of the displayed time range.
        stop : str or datetime or Timestamp, optional
            End of the displayed time range.
        y_axis_level : str, optional
            Tag level used to group columns onto separate y-axes
            (``"unit"``, ``"bloc"``, etc.).
        y_tag_list : list[str], optional
            Explicit list of tag values for y-axis grouping, overrides
            ``y_axis_level``.
        steps : None or str or list[str] or slice, default slice(None)
            Pipeline step(s) applied to the primary data before plotting.
        data_mode : str, default "lines"
            Plotly trace mode for the primary data (``"lines"``,
            ``"markers"``, ``"lines+markers"``).
        steps_2 : None or str or list[str] or slice, optional
            Pipeline step(s) for the optional secondary dataset overlay.
        data_2_mode : str, default "markers"
            Plotly trace mode for the secondary data.
        markers_opacity : float, default 0.8
            Opacity of markers (0.0 – 1.0).
        lines_width : float, default 2.0
            Width of line traces in pixels.
        title : str, optional
            Figure title.
        plot_gaps : bool, default False
            Highlight gaps in the primary data.
        gaps_lower_td : str or Timedelta or timedelta, optional
            Minimum gap duration to highlight (e.g. ``"1h"``).
        gaps_rgb : tuple[int, int, int], default (31, 73, 125)
            RGB colour for primary-data gap rectangles.
        gaps_alpha : float, default 0.5
            Opacity for primary-data gap rectangles.
        plot_gaps_2 : bool, default False
            Highlight gaps in the secondary data.
        gaps_2_lower_td : str or Timedelta or timedelta, optional
            Minimum gap duration to highlight in the secondary data.
        gaps_2_rgb : tuple[int, int, int], default (254, 160, 34)
            RGB colour for secondary-data gap rectangles.
        gaps_2_alpha : float, default 0.5
            Opacity for secondary-data gap rectangles.
        axis_space : float, default 0.03
            Horizontal space reserved per additional right-side y-axis.
        y_title_standoff : int or float, default 5
            Distance between a y-axis title and the axis line (pixels).
        verbose : bool, default False
            Print pipeline processing information.
        port : int, default 8050
            TCP port for the Dash server.

        Returns
        -------
        None
            The method launches a Dash server as a side effect; it does not
            return a figure object.

        Examples
        --------
        >>> plumber.plot_dash()

        >>> # Pre-select columns and separate axes by unit
        >>> plumber.plot_dash(
        ...     select=["temp__°C__zone1", "power__kW__hvac"],
        ...     y_axis_level="unit",
        ...     title="Zone 1 monitoring",
        ... )

        >>> # Compare raw vs processed data on port 8051
        >>> plumber.plot_dash(
        ...     steps=slice(None),
        ...     steps_2=None,
        ...     data_2_mode="markers",
        ...     port=8051,
        ... )
        """
        try:
            from dash import Dash, dcc, html
        except ImportError:
            raise ImportError(
                "dash is required for plot_dash. "
                "Install it with: pip install python-tide[resampler]"
            )

        import threading
        import time as _time
        import webbrowser

        # --- All processed data (for sidebar tree) ---
        all_data_1 = self.get_corrected_data(
            self.data.columns, start, stop, steps, verbose
        )

        # --- Selected data (for figure) ---
        select_corr = (
            self.data.columns if not tide_request(self.data, select) else select
        )
        data_1 = self.get_corrected_data(select_corr, start, stop, steps, verbose)

        if steps_2 is not None:
            all_data_2_raw = self.get_corrected_data(
                self.data.columns, start, stop, steps_2
            )
            all_data_2 = all_data_2_raw.copy()
            all_data_2.columns = [f"data_2->{c}" for c in all_data_2.columns]

            data_2_raw = self.get_corrected_data(select_corr, start, stop, steps_2)
            data_2 = data_2_raw.copy()
            data_2.columns = [f"data_2->{c}" for c in data_2.columns]
        else:
            all_data_2 = pd.DataFrame()
            data_2 = pd.DataFrame()

        # --- Pre-assign colors (over all columns so they stay stable) ---
        n = len(_DASH_COLORS)
        col_colors_1 = {
            col: _DASH_COLORS[i % n] for i, col in enumerate(all_data_1.columns)
        }
        col_colors_2 = {
            col: _DASH_COLORS[(len(col_colors_1) + i) % n]
            for i, col in enumerate(
                [c.replace("data_2->", "") for c in all_data_2.columns]
            )
        }

        # --- Build scatter config for ALL columns ---
        all_fig_cols = pd.concat([all_data_1, all_data_2], axis=1).columns
        col_axes_map, axes_col_map, y_labels = get_cols_axis_maps_and_labels(
            all_fig_cols, y_axis_level, y_tag_list
        )
        conf_dict_list = [
            {col: {"name": col} for col in all_fig_cols},
            col_axes_map,
            {col: {"mode": data_mode} for col in all_data_1}
            | {col: {"mode": data_2_mode} for col in all_data_2},
            {col: dict(line=dict(width=lines_width)) for col in all_fig_cols},
            {col: dict(marker=dict(opacity=markers_opacity)) for col in all_fig_cols},
        ]
        scatter_config = {}
        for d in conf_dict_list:
            for key in d:
                scatter_config[key] = {**scatter_config.get(key, {}), **d[key]}

        # Inject pre-assigned colors so sidebar inputs match traces
        for col in all_data_1.columns:
            scatter_config[col]["line"]["color"] = col_colors_1[col]
            scatter_config[col]["marker"]["color"] = col_colors_1[col]
        for col in all_data_2.columns:
            orig = col.replace("data_2->", "")
            scatter_config[col]["line"]["color"] = col_colors_2[orig]
            scatter_config[col]["marker"]["color"] = col_colors_2[orig]

        # --- Initial visibility sets ---
        selected_1 = set(data_1.columns)
        selected_2 = {c.replace("data_2->", "") for c in data_2.columns}

        # --- Try to use FigureResampler for dynamic downsampling on zoom ---
        try:
            from plotly_resampler import FigureResampler as _FigureResampler

            _use_resampler = True
        except ImportError:
            _use_resampler = False

        # --- Build figure with ALL traces (unselected ones hidden) ---
        fig = _FigureResampler(go.Figure()) if _use_resampler else go.Figure()
        for col in all_data_1:
            is_vis = col in selected_1
            if _use_resampler:
                fig.add_trace(
                    go.Scattergl(
                        visible=is_vis, showlegend=is_vis, **scatter_config[col]
                    ),
                    hf_x=all_data_1.index,
                    hf_y=all_data_1[col],
                )
            else:
                fig.add_scattergl(
                    x=all_data_1.index,
                    y=all_data_1[col],
                    visible=is_vis,
                    showlegend=is_vis,
                    **scatter_config[col],
                )
        if steps_2 is not None:
            for col in all_data_2:
                orig = col.replace("data_2->", "")
                is_vis = orig in selected_2
                if _use_resampler:
                    fig.add_trace(
                        go.Scattergl(
                            visible=is_vis, showlegend=is_vis, **scatter_config[col]
                        ),
                        hf_x=all_data_2.index,
                        hf_y=all_data_2[col],
                    )
                else:
                    fig.add_scattergl(
                        x=all_data_2.index,
                        y=all_data_2[col],
                        visible=is_vis,
                        showlegend=is_vis,
                        **scatter_config[col],
                    )

        # Gaps (only on initially visible data)
        yaxis_min_max = get_yaxis_min_max(
            pd.concat([data_1, data_2], axis=1), y_axis_level, y_tag_list
        )

        def _gap_dict_config(data, lower_td, rgb, alpha):
            result = []
            for col in data:
                configs = get_gap_scatter_dict(
                    data[col], yaxis_min_max, col_axes_map, lower_td, rgb, alpha
                )
                if configs:
                    result += configs
            return result

        gap_conf_list = []
        if plot_gaps:
            gap_conf_list += _gap_dict_config(
                data_1, gaps_lower_td, gaps_rgb, gaps_alpha
            )
        if plot_gaps_2:
            gap_conf_list += _gap_dict_config(
                data_2, gaps_2_lower_td, gaps_2_rgb, gaps_2_alpha
            )
        for gap in gap_conf_list:
            if _use_resampler:
                fig.add_trace(go.Scattergl(**gap))
            else:
                fig.add_scattergl(**gap)

        # --- Initial layout based on initially visible columns ---
        init_visible = selected_1 | {f"data_2->{c}" for c in selected_2}
        init_active_axes = {
            col_axes_map[c]["yaxis"] for c in init_visible if c in col_axes_map
        }
        all_axes_order = ["y"] + [f"y{i + 2}" for i in range(len(y_labels) - 1)]
        init_active_right = [a for a in all_axes_order[1:] if a in init_active_axes]
        x_right_space = 1.0 - axis_space * len(init_active_right)

        layout_dict = {
            "legend": dict(
                orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5
            ),
            "title": title,
            "yaxis": dict(
                title=y_labels[0] if y_labels is not None else None,
                side="left",
                title_standoff=y_title_standoff,
                visible="y" in init_active_axes,
            ),
            "margin": dict(l=60, r=20, t=50, b=80),
        }
        fig.update_xaxes(domain=(0, x_right_space))
        right_idx = 0
        for i, ax in enumerate(all_axes_order[1:]):
            is_active = ax in init_active_axes
            pos = x_right_space + right_idx * axis_space if is_active else 0
            layout_dict[f"yaxis{i + 2}"] = dict(
                title=y_labels[1 + i] if y_labels is not None else None,
                overlaying="y",
                side="right",
                position=pos,
                title_standoff=y_title_standoff,
                visible=is_active,
            )
            if is_active:
                right_idx += 1
        fig.update_layout(layout_dict)

        # --- Metadata store for callbacks ---
        metadata = {
            "col_ax_map": {col: v["yaxis"] for col, v in col_axes_map.items()},
            "y_labels": list(y_labels) if y_labels is not None else [],
            "axis_space": axis_space,
            "y_title_standoff": y_title_standoff,
        }

        # --- Build sidebar ---
        sidebar = _build_dash_sidebar(
            all_data_1.columns,
            all_data_2.columns,
            col_colors_1,
            col_colors_2,
            selected_1,
            selected_2,
        )

        # --- Dash app ---
        from dash import ALL, Input, Output, State, ctx

        app = Dash(__name__)
        _layout_children = [
            dcc.Store(id="metadata-store", data=metadata),
            dcc.Graph(
                id="main-graph",
                figure=fig,
                style={"flex": "1", "minWidth": "0"},
                config={"responsive": True},
            ),
            html.Div(
                style={
                    "width": "1px",
                    "backgroundColor": "#ddd",
                    "flexShrink": "0",
                }
            ),
            html.Div(
                [
                    html.Div(
                        "Données disponibles",
                        style={
                            "padding": "10px 8px 6px",
                            "fontWeight": "600",
                            "borderBottom": "1px solid #eee",
                            "fontSize": "0.9em",
                            "color": "#333",
                            "flexShrink": "0",
                        },
                    ),
                    html.Div(
                        sidebar,
                        style={"overflowY": "auto", "flex": "1"},
                    ),
                ],
                style={
                    "width": "280px",
                    "flexShrink": "0",
                    "display": "flex",
                    "flexDirection": "column",
                    "background": "#f9f9f9",
                },
            ),
        ]
        app.layout = html.Div(
            _layout_children,
            style={
                "display": "flex",
                "height": "100vh",
                "fontFamily": "sans-serif",
            },
        )

        # --- Callbacks ---
        # Both callbacks use Patch() to send only diffs to the browser so that
        # FigureResampler's own callback (which outputs the full figure on zoom)
        # never conflicts with our visibility / colour changes.
        from dash import Patch
        from dash.exceptions import PreventUpdate

        @app.callback(
            Output("main-graph", "figure"),
            Input({"type": "col-check", "index": ALL}, "value"),
            State("metadata-store", "data"),
            prevent_initial_call=True,
        )
        def _update_visibility(check_values, meta):
            visible_cols = {v for vals in check_values for v in vals}

            patched = Patch()
            for i, col in enumerate(all_fig_cols):
                is_vis = col in visible_cols
                patched["data"][i]["visible"] = is_vis
                patched["data"][i]["showlegend"] = is_vis

            col_ax_map = meta["col_ax_map"]
            _y_labels = meta["y_labels"]
            _ax_space = meta["axis_space"]

            active_axes = {col_ax_map[c] for c in visible_cols if c in col_ax_map}
            _all_axes = ["y"] + [f"y{j + 2}" for j in range(len(_y_labels) - 1)]
            active_right = [a for a in _all_axes[1:] if a in active_axes]
            x_right = 1.0 - _ax_space * len(active_right)

            patched["layout"]["xaxis"]["domain"] = [0, x_right]
            _right_idx = 0
            for _ax in _all_axes:
                yax_key = "yaxis" if _ax == "y" else f"yaxis{_ax[1:]}"
                _is_active = _ax in active_axes
                patched["layout"][yax_key]["visible"] = _is_active
                if _ax != "y":
                    _pos = x_right + _right_idx * _ax_space if _is_active else 0
                    patched["layout"][yax_key]["position"] = _pos
                    if _is_active:
                        _right_idx += 1

            return patched

        @app.callback(
            Output("main-graph", "figure", allow_duplicate=True),
            Input({"type": "col-color", "index": ALL}, "value"),
            prevent_initial_call=True,
        )
        def _update_color(color_values):
            col_name = ctx.triggered_id["index"] if ctx.triggered_id else None
            if not col_name:
                raise PreventUpdate

            color_idx = next(
                (
                    i
                    for i, inp in enumerate(ctx.inputs_list[0])
                    if inp.get("id", {}).get("index") == col_name
                ),
                None,
            )
            if color_idx is None:
                raise PreventUpdate

            new_color = color_values[color_idx]
            if not new_color:
                raise PreventUpdate

            trace_idx = next(
                (i for i, col in enumerate(all_fig_cols) if col == col_name),
                None,
            )
            if trace_idx is None:
                raise PreventUpdate

            patched = Patch()
            patched["data"][trace_idx]["line"]["color"] = new_color
            patched["data"][trace_idx]["marker"]["color"] = new_color
            return patched

        if _use_resampler:
            fig.register_update_graph_callback(app, "main-graph")

        # --- Launch in background thread ---
        thread = threading.Thread(
            target=lambda: app.run(port=port, debug=False, use_reloader=False),
            daemon=True,
        )
        thread.start()
        _time.sleep(1.0)
        webbrowser.open(f"http://localhost:{port}")
