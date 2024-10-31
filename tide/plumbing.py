import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from tide.utils import parse_request_to_col_names
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


def get_pipeline_from_dict(data_index: pd.Index | list[str], pipe_dict: dict):
    steps_list = []
    for step, op_conf in pipe_dict.items():
        if isinstance(op_conf, list):
            operation = _get_pipe_from_proc_list(op_conf)

        elif isinstance(op_conf, dict):
            operation = _get_column_wise_transformer(op_conf, data_index, step)

        else:
            raise ValueError(f"{op_conf} is an invalid operation config")

        if operation is not None:
            steps_list.append((step, operation))

    return Pipeline(steps_list)
