import datetime as dt

import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from tide.utils import (
    parse_request_to_col_names,
    check_and_return_dt_index_df,
    data_columns_to_tree,
    get_data_level_names,
)
from tide.plot import plot_gaps_heatmap, plot
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
    data_columns: pd.Index | list[str], pipe_dict: dict, verbose: bool = False
):
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

    return (
        Pipeline([("Identity", pc.Identity())], verbose=verbose)
        if not steps_list
        else Pipeline(steps_list, verbose=verbose)
    )


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

    def get_pipeline(
        self, select: str | pd.Index | list[str] = None, until_step: str = None
    ) -> Pipeline:
        self._check_config_data_pipe()
        selection = parse_request_to_col_names(self.data, select)
        if until_step is None:
            dict_to_pipe = self.pipe_dict
        else:
            dict_to_pipe = {}
            for key, value in self.pipe_dict.items():
                dict_to_pipe[key] = value
                if key == until_step:
                    break

        return get_pipeline_from_dict(selection, dict_to_pipe)

    def get_corrected_data(
        self,
        select: str | pd.Index | list[str] = None,
        start: str | dt.datetime | pd.Timestamp = None,
        stop: str | dt.datetime | pd.Timestamp = None,
        until_step: str = None,
    ) -> pd.DataFrame:
        self._check_config_data_pipe()
        select = parse_request_to_col_names(self.data, select)
        data = self.data.loc[
            start or self.data.index[0] : stop or self.data.index[-1], select
        ].copy()
        if until_step == "":
            return data
        else:
            return self.get_pipeline(select, until_step).fit_transform(data)

    def plot_gaps_heatmap(
        self,
        select: str | pd.Index | list[str] = None,
        start: str | dt.datetime | pd.Timestamp = None,
        stop: str | dt.datetime | pd.Timestamp = None,
        until_step: str = None,
        time_step: str | pd.Timedelta | dt.timedelta = None,
        title: str = None,
    ):
        data = self.get_corrected_data(select, start, stop, until_step)
        return plot_gaps_heatmap(data, time_step=time_step, title=title)

    def plot(
        self,
        select: str | pd.Index | list[str] = None,
        start: str | dt.datetime | pd.Timestamp = None,
        stop: str | dt.datetime | pd.Timestamp = None,
        until_step_1: str = None,
        until_step_2: str = None,
        plot_gaps_1: bool = False,
        plot_gaps_2: bool = False,
        data_1_mode: str = "lines",
        data_2_mode: str = "markers",
        y_axis_level: str = None,
        y_axis_tag: [str] = None,
        title: str = None,
        markers_opacity: float = 0.8,
        lines_width: float = 2.0,
    ):
        # A bit dirty. Here we assume that if you ask a selection
        # that is not found in original data columns, it is because it
        # has not yet been computed (using ExpressionCombine processor
        # for example) So we just process the whole data
        select_corr = (
            self.data.columns
            if not parse_request_to_col_names(self.data, select)
            else select
        )

        data_1 = self.get_corrected_data(select_corr, start, stop, until_step_1)
        mode_dict = {col: data_1_mode for col in data_1.columns}

        if until_step_2 is not None:
            data_2 = self.get_corrected_data(select_corr, start, stop, until_step_2)
            data_2.columns = [f"data_2->{col}" for col in data_2.columns]
            mode_dict = mode_dict | {col: data_2_mode for col in data_2.columns}
            data = pd.concat([data_1, data_2], axis=1)
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

        axis_dict = {}
        for i, tag in enumerate(y_tags):
            selected_cols = parse_request_to_col_names(data.columns, tag)
            for col in selected_cols:
                axis_dict[col] = "y" if i == 0 else f"y{i + 1}"

        fig = plot(
            data,
            title,
            y_tags,
            axis_dict,
            mode_dict,
            markers_opacity=markers_opacity,
            lines_width=lines_width
        )

        return fig
