import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from tide import AGG_METHOD_MAP
from tide.utils import data_columns_to_tree, parse_request_to_col_names
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

def _get_resampler(arg_list:list, data_columns:pd.Index | list[str]):
    rule = arg_list[0]

    if len(arg_list) == 1:
        return pc.Resampler(rule=rule, method=AGG_METHOD_MAP["MEAN"])

    elif isinstance(arg_list[1], str):
        return pc.Resampler(rule=rule, method=AGG_METHOD_MAP[arg_list[1]])

    else:
        column_config_list = [
            (parse_request_to_col_names(data_columns, req), AGG_METHOD_MAP[method])
            for req, method in arg_list[1].items()
        ]

        return pc.ColumnResampler(
            rule=rule,
            columns_method=column_config_list,
            remainder=AGG_METHOD_MAP["MEAN"],
        )

def get_pipeline_from_dict(data_index: pd.Index | list[str], pipe_dict: {}):
    data_root = data_columns_to_tree(data_index)
    return None
