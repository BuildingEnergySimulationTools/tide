from enum import Enum
from tide.math import time_integrate
import tide.processing as pc


class AggMethod(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"
    CUMSUM = "CUMSUM"
    DIFF = "DIFF"
    TIME_INTEGRATE = "TIME_INTEGRATE"


AGG_METHOD_MAP = {
    "MEAN": "mean",
    "SUM": "sum",
    "CUMSUM": "cusmsum",
    "DIFF": "diff",
    "TIME_INTEGRATE": time_integrate,
}