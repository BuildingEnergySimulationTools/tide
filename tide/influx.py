import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from influxdb_client import InfluxDBClient

from tide.utils import check_and_return_dt_index_df


def _date_objects_tostring(date: dt.datetime | pd.Timestamp, tz_info=None):
    if date.tzinfo is None:
        if tz_info is None:
            raise ValueError("tz_info must be provided for naive datetime objects.")
        date = date.replace(tzinfo=ZoneInfo(tz_info))

    date_utc = date.astimezone(ZoneInfo("UTC"))
    return date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _single_influx_request(
    start: str | pd.Timestamp | dt.datetime,
    stop: str | pd.Timestamp | dt.datetime,
    bucket: str,
    measurement: str,
    tide_tags: list[str],
    url: str,
    org: str,
    token: str,
    tz_info: str = "UTC",
) -> pd.DataFrame:
    client = InfluxDBClient(url=url, org=org, token=token)
    query_api = client.query_api()
    query = f"""
        from(bucket: "{bucket}")
        |> range(start: {_date_objects_tostring(start, tz_info)}, 
                 stop: {_date_objects_tostring(stop, tz_info)})
        |> filter(fn: (r) => r["_measurement"] == "{measurement}")
        |> map(fn: (r) => ({{r with tide: r.{tide_tags[0]}
    """
    if len(tide_tags) > 1:
        for tag in tide_tags[1:]:
            query += f' + "__" + r.{tag}'
    query += "}))"
    query += """
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


def get_influx_data(
    start: str | pd.Timestamp | dt.datetime,
    stop: str | pd.Timestamp | dt.datetime,
    bucket: str,
    measurement: str,
    tide_tags: list[str],
    url: str,
    org: str,
    token: str,
    split_td: str | dt.timedelta | pd.Timedelta = None,
    tz_info: str = "UTC",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fetches data from an InfluxDB instance for the specified time range,
    bucket, and measurement, optionally splitting the request into smaller time
    intervals.

    Parameters
    ----------
    start : str, pd.Timestamp, or datetime.datetime
        The start of the time range for the query. Can be:
        - A relative time string (e.g., "-1d", "-2h").
        - A `pd.Timestamp` or `datetime.datetime` object.

    stop : str, pd.Timestamp, or datetime.datetime
        The end of the time range for the query.
        Accepts the same formats as `start`.

    bucket : str
        The name of the InfluxDB bucket to query data from.

    measurement : str
        The _measurement name within the InfluxDB bucket to filter data.

    tide_tags : list[str]
        A list of fields or tags in Influx that correspond to Tide tags.
        Must be specified in the following order name__unit__bloc__sub_bloc.

    url : str
        The URL of the InfluxDB instance (e.g., "http://localhost:8086").

    org : str
        The organization name in the InfluxDB instance.

    token : str
        The authentication token for accessing the InfluxDB instance.

    split_td : str, datetime.timedelta, or pd.Timedelta, optional
        The time interval for splitting the query into smaller chunks
        (e.g., "1d", "12h"). If `None`, the query will not be split.

    tz_info : str, optional
        The timezone for interpreting the start and stop times.
        Defaults to "UTC".

    verbose : bool, optional
        If `True`, prints progress messages for each time chunk being fetched.
        Defaults to `False`.

    """

    if isinstance(start, str) and isinstance(stop, str):
        start = dt.datetime.now() + pd.Timedelta(start)
        stop = dt.datetime.now() + pd.Timedelta(stop)

    if split_td is not None:
        dates_index = pd.date_range(start, stop, freq=split_td)
    else:
        dates_index = pd.Index([start, stop])

    df_list = []
    for i in range(len(dates_index) - 1):
        if verbose:
            print(f"Getting period {i + 1} / {len(dates_index) - 1}")
        df_list.append(
            _single_influx_request(
                start=dates_index[i],
                stop=dates_index[i + 1],
                bucket=bucket,
                measurement=measurement,
                tide_tags=tide_tags,
                url=url,
                org=org,
                token=token,
                tz_info=tz_info,
            )
        )

    return df_list[0] if len(df_list) == 1 else pd.concat(df_list)


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
