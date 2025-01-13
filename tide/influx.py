import datetime as dt

import pandas as pd
from influxdb_client import InfluxDBClient

from tide.utils import check_and_return_dt_index_df


def date_objects_tostring(date):
    if isinstance(date, dt.datetime) or isinstance(date, pd.Timestamp):
        return date.strftime("%Y-%m-%d %H:%M")
    else:
        return date


def get_influx_data(
    start: str | pd.Timestamp | dt.datetime,
    stop: str | pd.Timestamp | dt.datetime,
    bucket: str,
    measurement: str,
    tide_tags: list[str],
    url: str,
    org: str,
    token: str,
):
    client = InfluxDBClient(url=url, org=org, token=token)

    query_api = client.query_api()
    query = f"""
        from(bucket: {bucket})
        |> range(start: {date_objects_tostring(start)}, stop: {date_objects_tostring(stop)})
        |> filter(fn: (r) => r["_measurement"] == {measurement})
        |> map(fn: (r) => ({{r with tide: r.{tide_tags[0]} + "__" + r.{tide_tags[1]} + "__" + r.{tide_tags[2]} + "__" + r.{tide_tags[3]}}}))
        |> keep(columns: ["_time", "_value", "tide"])
        |> pivot(rowKey: ["_time"], columnKey: ["tide"], valueColumn: "_value")
        |> sort(columns: ["_time"])
    """

    tables = query_api.query(query)

    records = []
    for table in tables:
        for record in table.records:
            records.append(record.values)

    df = pd.DataFrame(records)
    df["_time"] = pd.to_datetime(df["_time"])
    df.set_index("_time", inplace=True)
    df.drop(["result", "table"], axis=1, inplace=True)

    return df


def push_influx_data(
    data: pd.DataFrame,
    tide_tags: list[str],
    bucket: str,
    url: str,
    org: str,
    token: str,
    measurement: str = "tide",
):
    data = check_and_return_dt_index_df(data)
    influx_df_list = []
    for time, row in data.iterrows():
        df = row.reset_index()
        df.columns = ["full_index", "_value"]
        df[tide_tags] = df["full_index"].str.split("__", expand=True)
        df = df.drop("full_index", axis=1)
        df.index = pd.to_datetime([time] * df.shape[0])
        influx_df_list.append(df)

    influx_df = pd.concat(influx_df_list).dropna()

    with InfluxDBClient(url=url, token=token, org=org) as client:
        with client.write_api() as write_client:
            write_client.write(
                bucket=bucket,
                record=influx_df,
                data_frame_measurement_name=measurement,
                data_frame_tag_columns=tide_tags,
            )
