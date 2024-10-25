from pathlib import Path

import pandas as pd

from tide.classifiers import STLEDetector

RESOURCES_PATH = Path(__file__).parent / "resources"


class TestErrorDetection:
    def test_stl_e_detector(self):
        # A temperature timeseries with artificial errors (+0.7°C) at given time steps
        data = pd.read_csv(
            RESOURCES_PATH / "stl_data.csv", index_col=0, parse_dates=True
        )
        data = data.asfreq("15min")

        stl = STLEDetector(
            period="24h",
            trend="1d",
            stl_kwargs={"robust": True},
            absolute_threshold=0.6,
        )

        stl.fit(data)
        res = stl.predict(data)

        pd.testing.assert_index_equal(res.index, data.index)
        pd.testing.assert_index_equal(res.columns, data.columns)

        # Check that the 3 errors are found
        assert res.sum().iloc[0] == 3
