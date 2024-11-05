import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from tide.utils import (
    parse_request_to_col_names,
    check_and_return_dt_index_df,
    data_columns_to_tree,
    get_data_level_names,
)
import tide.processing as pc


def _select_to_data_columns(
    data: pd.DataFrame = None, select: str | pd.Index | list[str] = None
):
    return (
        parse_request_to_col_names(data.columns, select)
        if isinstance(select, str) or select is None
        else select
    )


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


def get_pipeline_from_dict(data_columns: pd.Index | list[str], pipe_dict: dict):
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

    return Pipeline(steps_list)


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

    def get_pipeline(self, select: str | pd.Index | list[str]) -> Pipeline:
        select = _select_to_data_columns(select)
        return get_pipeline_from_dict(select, self.pipe_dict)

    def get_corrected_data(self, select: str | pd.Index | list[str]):
        select = _select_to_data_columns(select)
        return self.get_pipeline(select).fit_transform(self.data[select].copy())
