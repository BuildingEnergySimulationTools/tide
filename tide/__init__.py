from enum import Enum
import tide.processing as pc


class Processor(Enum):
    DROPNA = "DROPNA"
    RENAME_COLUMNS = "RENAME_COLUMNS"
    SK_TRANSFORMER = "SK_TRANSFORMER"
    DROP_THRESHOLD = "DROP_THRESHOLD"
    DROP_TIME_GRADIENT = "DROP_TIME_GRADIENT"
    APPLY_EXPRESSION = "APPLY_EXPRESSION"
    TIME_GRADIENT = "TIME_GRADIENT"
    FILL_NA = "FILL_NA"
    BFILL = "BFILL"
    FFILL = "FFILL"
    RESAMPLE = "RESAMPLE"
    INTERPOLATE = "INTERPOLATE"
    GAUSSIAN_FILTER = "GAUSSIAN_FILTER"
    REPLACE_DUPLICATED = "REPLACE_DUPLICATED"
    STL_ERROR_FILTER = "STL_ERROR_FILTER"
    FILL_GAPS_AR = "FILL_GAPS_AR"


PROCESSOR_MAP = {
    "DROPNA": pc.Dropna,
    "RENAME_COLUMNS": pc.RenameColumns,
    "SK_TRANSFORMER": pc.SkTransformer,
    "DROP_THRESHOLD": pc.DropThreshold,
    "DROP_TIME_GRADIENT": pc.DropTimeGradient,
    "APPLY_EXPRESSION": pc.ApplyExpression,
    "TIME_GRADIENT": pc.TimeGradient,
    "FILL_NA": pc.FillNa,
    "BFILL": pc.Bfill,
    "FFILL": pc.Ffill,
    "RESAMPLE": pc.Resampler,
    "INTERPOLATE": pc.Interpolate,
    "GAUSSIAN_FILTER": pc.GaussianFilter1D,
    "REPLACE_DUPLICATED": pc.ReplaceDuplicated,
    "STL_ERROR_FILTER": pc.STLFilter,
    "FILL_GAPS_AR": pc.FillGapsAR,
}
