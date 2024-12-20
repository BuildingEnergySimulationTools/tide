import pandas as pd
import numpy as np
import pytest

from tide.base import BaseProcessing


class DumbProcessor(BaseProcessing):
    def __init__(self, required_columns, keep_required):
        BaseProcessing.__init__(self, required_columns)
        self.keep_required = keep_required

    def _fit_implementation(self, X, y=None):
        if not self.keep_required:
            self.removed_columns = self.required_columns
        self.added_columns = ["new_col__°C__meteo"]
        return self

    def _transform_implementation(self, X):
        X[self.added_columns[0]] = X[self.required_columns] * 2
        X.drop(self.removed_columns, axis=1, inplace=True)
        return X


class TestBase:
    def test_base_processing(self):
        data = pd.DataFrame(
            {
                "text__°C__meteo": np.random.randn(24),
                "hr__%hr__meteo": np.random.randn(24),
            },
            index=pd.date_range("2024-12-05 00:00:00", freq="h", periods=24),
        )

        dp = DumbProcessor(required_columns=["text__°C__meteo"], keep_required=False)

        dp.fit(data.copy())
        assert dp.get_feature_names_in() == list(data.columns)
        assert dp.get_feature_names_out() == ["hr__%hr__meteo", "new_col__°C__meteo"]

        res = dp.transform(data.copy())

        with pytest.raises(ValueError):
            dp.transform(res)

        dp = DumbProcessor(required_columns=["text__°C__meteo"], keep_required=True)
        dp.fit(data)
        assert dp.get_feature_names_out() == [
            "text__°C__meteo",
            "hr__%hr__meteo",
            "new_col__°C__meteo",
        ]
