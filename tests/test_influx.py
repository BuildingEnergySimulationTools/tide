import datetime as dt
from unittest.mock import MagicMock, patch

import pandas as pd

from tide.influx import get_influx_data, push_influx_data


def mock_influx_client_query(query):
    ref_query = """
            from(bucket: "my_bucket")
        |> range(start: 2009-01-01T00:00:00Z, 
                 stop: 2009-01-01T03:00:00Z)
        |> filter(fn: (r) => r["_measurement"] == "log")
        |> map(fn: (r) => ({r with tide: r.Name
     + "__" + r.Unit + "__" + r.bloc}))
        |> keep(columns: ["_time", "_value", "tide"])
        |> pivot(rowKey: ["_time"], columnKey: ["tide"], valueColumn: "_value")
        |> sort(columns: ["_time"])
    """

    assert query.strip().replace("\n", "").replace(
        " ", ""
    ) == ref_query.strip().replace("\n", "").replace(" ", "")

    return [
        MagicMock(
            records=[
                MagicMock(
                    values={
                        "_time": "2023-01-01T00:00:00Z",
                        "temp__°C__bloc1": 0.0,
                        "hr__%hr__bloc1": 10.0,
                        "table": 0,
                        "result": "_result",
                    }
                ),
                MagicMock(
                    values={
                        "_time": "2023-01-01T02:00:00Z",
                        "temp__°C__bloc1": 1.0,
                        "hr__%hr__bloc1": 20.0,
                        "table": 0,
                        "result": "_result",
                    }
                ),
                MagicMock(
                    values={
                        "_time": "2023-01-03T01:00:00Z",
                        "temp__°C__bloc1": 2.0,
                        "hr__%hr__bloc1": 30.0,
                        "table": 0,
                        "result": "_result",
                    }
                ),
            ]
        )
    ]


URL = "https://influx_db.com:3000"
INFLUX_ORG = "my_org"
INFLUX_TOKEN = "my_tok"
TIDE_USED_TAGS = ["Name", "Unit", "bloc"]
BUCKET = "my_bucket"
MEASUREMENT = "log"
START = dt.datetime(2009, 1, 1, 0)
STOP = dt.datetime(2009, 1, 1, 3)


class TestInflux:
    @patch("influxdb_client.InfluxDBClient.query_api")
    def test_get_influx_data(self, mock_query_api):
        mock_query_api.return_value.query = MagicMock(
            side_effect=mock_influx_client_query
        )

        df = get_influx_data(
            START,
            STOP,
            BUCKET,
            MEASUREMENT,
            TIDE_USED_TAGS,
            URL,
            INFLUX_ORG,
            INFLUX_TOKEN,
        )

        assert isinstance(df, pd.DataFrame)

    @patch("tide.influx.InfluxDBClient")
    def test_push_influx_data(self, mock_influx_client):
        mock_write_api = MagicMock()
        mock_influx_client.return_value.__enter__.return_value.write_api.return_value.__enter__.return_value = mock_write_api

        data = pd.DataFrame(
            {
                "name1__°C__bloc1": [1.0, 2.0],
                "name2__W__bloc1": [3.0, 4.0],
            },
            index=pd.to_datetime(["2009-01-01T00:00:00Z", "2009-01-01T01:00:00Z"]),
        )

        push_influx_data(
            data=data,
            tide_tags=TIDE_USED_TAGS,
            bucket=BUCKET,
            url=URL,
            org=INFLUX_ORG,
            token=INFLUX_TOKEN,
            measurement=MEASUREMENT,
        )

        # Assertions to verify interactions with the mock
        mock_influx_client.assert_called_once_with(
            url=URL, token=INFLUX_TOKEN, org=INFLUX_ORG
        )  # Ensure write_api was used

        # Ensure InfluxDBClient was initialized
        mock_influx_client.assert_called_once_with(
            url="https://influx_db.com:3000", token="my_tok", org="my_org"
        )

        mock_write_api.write.assert_called_once()
        call_args = mock_write_api.write.call_args[1]
        assert call_args["bucket"] == BUCKET
        assert call_args["data_frame_measurement_name"] == MEASUREMENT
        assert call_args["data_frame_tag_columns"] == TIDE_USED_TAGS

        written_df = call_args["record"]
        assert set(written_df.columns) == {"_value", "Name", "Unit", "bloc"}
        assert written_df["_value"].tolist() == [1.0, 3.0, 2.0, 4.0]
        assert written_df["Unit"].tolist() == ["°C", "W", "°C", "W"]
        assert written_df["bloc"].tolist() == ["bloc1"] * 4
