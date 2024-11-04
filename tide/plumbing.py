import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from tide.utils import (
    parse_request_to_col_names,
    check_and_return_dt_index_df,
    data_columns_to_tree,
)
import tide.processing as pc


def _select_to_data_columns(select: str | pd.Index | list[str]):
    return parse_request_to_col_names(select) if isinstance(select, str) else select


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
        self.data = check_and_return_dt_index_df(data)
        self.pipe_dict = pipe_dict

    def __repr__(self):
        if self.data is not None:
            root = data_columns_to_tree(self.data.columns)
            root.show()

    def get_pipeline(self, select: str | pd.Index | list[str]) -> Pipeline:
        select = _select_to_data_columns(select)
        return get_pipeline_from_dict(select, self.pipe_dict)

    def get_corrected_data(self, select: str | pd.Index | list[str]):
        select = _select_to_data_columns(select)
        return self.get_pipeline(select).fit_transform(self.data[select].copy())
