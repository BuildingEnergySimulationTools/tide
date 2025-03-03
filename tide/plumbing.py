import datetime as dt

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from tide.utils import (
    parse_request_to_col_names,
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
        requested_col = parse_request_to_col_names(data_columns, req)
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


class Plumber:
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
        Get statistical description of gaps durations in the data.

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
            Whether to include statistics for gaps present in any column

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
                    durations.append(gap[-1] - gap[0])
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
        row_order = ["data_presence_%"] + [idx for idx in stats_df.index if idx != "data_presence_%"]
        return stats_df.reindex(row_order)

    def set_data(self, data: pd.Series | pd.DataFrame):
        self.data = check_and_return_dt_index_df(data)
        self.root = data_columns_to_tree(data.columns)

    def select(
        self,
        select: str | pd.Index | list[str] = None,
    ):
        return parse_request_to_col_names(self.data, select)

    def get_pipeline(
        self,
        select: str | pd.Index | list[str] = None,
        steps: None | str | list[str] | slice = slice(None),
        verbose: bool = False,
    ) -> Pipeline:
        if self.data is None:
            raise ValueError("data is required to build a pipeline")
        selection = parse_request_to_col_names(self.data, select)
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
        if self.data is None:
            raise ValueError("Cannot get corrected data. data are missing")
        select = parse_request_to_col_names(self.data, select)
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
    ):
        # A bit dirty. Here we assume that if you ask a selection
        # that is not found in original data columns, it is because it
        # has not yet been computed (using ExpressionCombine processor
        # for example) So we just process the whole data hoping to find the result
        # after.
        select_corr = (
            self.data.columns
            if not parse_request_to_col_names(self.data, select)
            else select
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

        fig = go.Figure()
        for col in data_1:
            fig.add_scattergl(x=data_1.index, y=data_1[col], **scatter_config[col])

        if steps_2 is not None:
            for col in data_2:
                fig.add_scattergl(x=data_2.index, y=data_2[col], **scatter_config[col])

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
