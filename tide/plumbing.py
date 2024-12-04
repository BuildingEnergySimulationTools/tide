import datetime as dt

import pandas as pd

import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from tide.utils import (
    parse_request_to_col_names,
    check_and_return_dt_index_df,
    data_columns_to_tree,
    get_data_level_names,
    get_data_blocks,
    get_outer_timestamps,
    NamedList,
)
from tide.plot import (
    plot_gaps_heatmap,
    add_multi_axis_scatter,
    get_cols_axis_maps_and_labels,
    get_gap_scatter_dict,
    get_yaxis_min_max,
)
import tide.processing as pc


def _get_pipe_from_proc_list(proc_list: list) -> Pipeline:
    proc_units = [
        getattr(pc, proc[0])(
            *proc[1] if len(proc) > 1 and isinstance(proc[1], list) else (),
            **proc[1] if len(proc) > 1 and isinstance(proc[1], dict) else {},
        )
        for proc in proc_list
    ]
    return make_pipeline(*proc_units)


def _get_column_wise_transformer(
    proc_dict, data_columns: pd.Index | list[str], process_name: str = None
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
                    _get_pipe_from_proc_list(proc_list),
                    requested_col,
                )
            )

    if not col_trans_list:
        return None
    else:
        return ColumnTransformer(
            col_trans_list,
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")


def get_pipeline_from_dict(
    data_columns: pd.Index | list[str], pipe_dict: dict = None, verbose: bool = False
):
    if pipe_dict is None:
        return Pipeline([("Identity", pc.Identity())], verbose=verbose)
    else:
        steps_list = []
        for step, op_conf in pipe_dict.items():
            if isinstance(op_conf, list):
                operation = _get_pipe_from_proc_list(op_conf)

            elif isinstance(op_conf, dict):
                operation = _get_column_wise_transformer(op_conf, data_columns, step)

            else:
                raise ValueError(f"{op_conf} is an invalid operation config")

            if operation is not None:
                steps_list.append((step, operation))

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
                for lvl_name in get_data_level_names(self.root, tag_levels[tag]):
                    rep_str += f"{lvl_name}\n"
                rep_str += "\n"
            return rep_str
        else:
            return super().__repr__()

    def _check_config_data_pipe(self):
        if self.data is None or self.pipe_dict is None:
            raise ValueError("data and pipe_dict are required")

    def show(self):
        if self.root is not None:
            self.root.show()

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
        steps: str | list[str] | slice = slice(None),
        verbose: bool = False,
    ) -> Pipeline:
        self._check_config_data_pipe()
        selection = parse_request_to_col_names(self.data, select)
        if steps is None:
            dict_to_pipe = None
        else:
            pipe_named_keys = NamedList(list(self.pipe_dict.keys()))
            selected_steps = pipe_named_keys[steps]
            dict_to_pipe = {key: self.pipe_dict[key] for key in selected_steps}

        return get_pipeline_from_dict(selection, dict_to_pipe, verbose)

    def get_corrected_data(
        self,
        select: str | pd.Index | list[str] = None,
        start: str | dt.datetime | pd.Timestamp = None,
        stop: str | dt.datetime | pd.Timestamp = None,
        steps: str | list[str] | slice = slice(None),
        verbose: bool = False,
    ) -> pd.DataFrame:
        self._check_config_data_pipe()
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
        steps: str | list[str] | slice = slice(None),
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
        steps_1: str | list[str] | slice = slice(None),
        data_1_mode: str = "lines",
        steps_2: str | list[str] | slice = None,
        data_2_mode: str = "markers",
        markers_opacity: float = 0.8,
        lines_width: float = 2.0,
        title: str = None,
        plot_gaps_1: bool = False,
        gaps_1_lower_td: str | pd.Timedelta | dt.timedelta = None,
        gaps_1_rgb: tuple[int, int, int] = (31, 73, 125),
        gaps_1_alpha: float = 0.5,
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

        data_1 = self.get_corrected_data(select_corr, start, stop, steps_1, verbose)
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
            {col: {"mode": data_1_mode} for col in data_1}
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
        if plot_gaps_1:
            gap_conf_list += gap_dict_config(
                data_1, gaps_1_lower_td, gaps_1_rgb, gaps_1_alpha
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

    def plot_legacy(
        self,
        select: str | pd.Index | list[str] = None,
        start: str | dt.datetime | pd.Timestamp = None,
        stop: str | dt.datetime | pd.Timestamp = None,
        until_step_1: str = None,
        until_step_2: str = None,
        data_1_mode: str = "lines",
        data_2_mode: str = "markers",
        plot_gaps_1: bool = False,
        gaps_1_lower_td: str | pd.Timedelta | dt.timedelta = None,
        plot_gaps_2: bool = False,
        gaps_2_lower_td: str | pd.Timedelta | dt.timedelta = None,
        gaps_alpha: float = 0.5,
        y_axis_level: str = None,
        y_axis_tag: [str] = None,
        title: str = None,
        markers_opacity: float = 0.8,
        lines_width: float = 2.0,
    ):
        def get_gaps_start_end(df, lower_td):
            gaps = get_data_blocks(
                df,
                is_null=True,
                lower_td_threshold=lower_td,
                return_combination=False,
            )

            gaps = {col: gp for col, gp in gaps.items() if gp}

            gp_dict = {
                col: [get_outer_timestamps(idx, df.index) for idx in gaps[col]]
                for col in gaps.keys()
            }

            return {} if all([not gap for gap in gp_dict.values()]) else gp_dict

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

        data_1 = self.get_corrected_data(select_corr, start, stop, until_step_1)
        mode_dict = {col: data_1_mode for col in data_1.columns}

        gaps_dict = {} if plot_gaps_1 or plot_gaps_2 else None

        if plot_gaps_1:
            gaps_dict = gaps_dict | get_gaps_start_end(data_1, gaps_1_lower_td)

        if until_step_2 is not None:
            data_2 = self.get_corrected_data(select_corr, start, stop, until_step_2)
            data_2.columns = [f"data_2->{col}" for col in data_2.columns]
            mode_dict = mode_dict | {col: data_2_mode for col in data_2.columns}
            data = pd.concat([data_1, data_2], axis=1)
            if plot_gaps_2:
                gaps_dict = gaps_dict | get_gaps_start_end(data_2, gaps_2_lower_td)

        else:
            data = data_1

        # Get back only what we wanted
        cols = parse_request_to_col_names(data, select)
        if not cols:
            raise ValueError(
                f"Invalid selection: '{select}' not found in the "
                f"DataFrame columns after processing."
            )
        data = data.loc[:, cols]

        if y_axis_tag:
            y_tags = y_axis_tag
        else:
            root = data_columns_to_tree(data.columns)
            level = y_axis_level if y_axis_level else "unit"
            y_tags = get_data_level_names(root, level)

        col_axes_map = {}
        axes_col_map = {}
        for i, tag in enumerate(y_tags):
            selected_cols = parse_request_to_col_names(data.columns, tag)
            axes_col_map["y" if i == 0 else f"y{i + 1}"] = selected_cols
            for col in selected_cols:
                col_axes_map[col] = "y" if i == 0 else f"y{i + 1}"

        axes_min_max = {
            ax: (
                float(data[axes_col_map[ax]].min().min()),
                float(data[axes_col_map[ax]].max().max()),
            )
            for ax in axes_col_map.keys()
        }

        fig = go.Figure()

        add_multi_axis_scatter(
            fig,
            data,
            title,
            y_tags,
            col_axes_map,
            mode_dict,
            markers_opacity=markers_opacity,
            lines_width=lines_width,
        )

        if gaps_dict:
            for col, gap_list in gaps_dict.items():
                for gap in gap_list:
                    fig.add_trace(
                        go.Scattergl(
                            x=[gap[0], gap[0], gap[1], gap[1]],
                            y=[
                                axes_min_max[col_axes_map[col]][0],
                                axes_min_max[col_axes_map[col]][1],
                                axes_min_max[col_axes_map[col]][1],
                                axes_min_max[col_axes_map[col]][0],
                            ],
                            mode="none",
                            fill="toself",
                            showlegend=False,
                            fillcolor=f"rgba(128, 128, 128, {gaps_alpha})",
                            yaxis=col_axes_map[col],
                        )
                    )

        return fig
