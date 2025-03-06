import time
import datetime as dt
from zoneinfo import ZoneInfo
from urllib3.exceptions import ReadTimeoutError

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
    if not df.empty:
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
    max_retry: int = 5,
    waited_seconds_at_retry: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fetch time series data from an InfluxDB instance.

    This function retrieves data from InfluxDB and formats it according to Tide's
    hierarchical column naming convention. It supports:

        - Flexible time range specification
        - Automatic query splitting for large time ranges
        - Retry mechanism for handling timeouts
        - Timezone-aware data handling

    Parameters
    ----------
    start : str or pd.Timestamp or datetime.datetime
        Start time for the query. Can be:
            - A relative time string (e.g., "-1d", "-2h")
            - A pandas Timestamp
            - A datetime object
        If using relative time strings, they are interpreted relative to the current time.

    stop : str or pd.Timestamp or datetime.datetime
        End time for the query. Accepts the same formats as start.

    bucket : str
        Name of the InfluxDB bucket to query.

    measurement : str
        Name of the InfluxDB measurement to filter data.

    tide_tags : list[str]
        List of InfluxDB fields/tags to combine into Tide column names.
        Must be specified in order: [name, unit, bloc, sub_bloc].
        Example: ["name", "unit", "location", "room"] will create columns like
        "temperature__°C__zone1__room1"

    url : str
        URL of the InfluxDB instance (e.g., "http://localhost:8086")

    org : str
        InfluxDB organization name

    token : str
        Authentication token for InfluxDB access

    split_td : str or datetime.timedelta or pd.Timedelta, optional
        Time interval for splitting large queries into smaller chunks.
        Useful for handling large time ranges or rate limits.
        Example: "1d" for daily chunks, "12h" for half-day chunks.
        If None, queries the entire time range at once.

    tz_info : str, default "UTC"
        Timezone for interpreting start and stop times.
        Must be a valid timezone name from the IANA Time Zone Database.

    max_retry : int, default 5
        Maximum number of retry attempts for failed queries.
        Only applies to ReadTimeoutError exceptions.

    waited_seconds_at_retry : int, default 5
        Number of seconds to wait between retry attempts.

    verbose : bool, default False
        Whether to print progress information during data fetching.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the fetched data with:
            - Datetime index in UTC
            - Columns named according to Tide's convention (name__unit__bloc__sub_bloc)
            - Values from the InfluxDB _value field

    Raises
    ------
    ReadTimeoutError
        If all retry attempts fail to fetch data
    ValueError
        If tz_info is required but not provided for naive datetime objects

    Examples
    --------
    >>> from tide import get_influx_data
    >>> import pandas as pd
    >>> # Fetch last 24 hours of data
    >>> df = get_influx_data(
    ...     start="-24h",
    ...     stop="now",
    ...     bucket="my_bucket",
    ...     measurement="sensors",
    ...     tide_tags=["name", "unit", "location"],
    ...     url="http://localhost:8086",
    ...     org="my_org",
    ...     token="my_token",
    ... )
    >>> # Fetch specific time range with daily splitting
    >>> df = get_influx_data(
    ...     start="2023-01-01",
    ...     stop="2023-01-07",
    ...     bucket="my_bucket",
    ...     measurement="sensors",
    ...     tide_tags=["name", "unit", "location", "room"],
    ...     url="http://localhost:8086",
    ...     org="my_org",
    ...     token="my_token",
    ...     split_td="1d",
    ...     verbose=True,
    ... )
    >>> # Fetch data with custom timezone
    >>> df = get_influx_data(
    ...     start="2023-01-01T00:00:00",
    ...     stop="2023-01-01T23:59:59",
    ...     bucket="my_bucket",
    ...     measurement="sensors",
    ...     tide_tags=["name", "unit", "location"],
    ...     url="http://localhost:8086",
    ...     org="my_org",
    ...     token="my_token",
    ...     tz_info="Europe/Paris",
    ... )
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
        for attempt in range(max_retry):
            try:
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
                break
            except ReadTimeoutError:
                if attempt < max_retry - 1:
                    if verbose:
                        print(f"Attempt {attempt + 1} failed")
                    time.sleep(waited_seconds_at_retry)
                else:
                    if verbose:
                        print("Max retries reached. Unable to get data.")
                    raise

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
    """
    Pushes data from a pandas DataFrame to an InfluxDB bucket.

    This function processes a DataFrame indexed by datetime and writes the data
    to an InfluxDB bucket. Each row in the DataFrame is expanded based on Tide tags
    extracted from a specific column and written to InfluxDB with corresponding
    timestamp and tag values.

    Parameters:
        data (pd.DataFrame): Input DataFrame with a datetime index and
            one or more columns of values.

        tide_tags (list[str]): List of tag names to extract from the
            "full_index" column after splitting it. For exemple : ["Name", "Unit",
            "bloc", "sub_bloc"

        bucket (str): InfluxDB bucket name where the data will be written.

        url (str): URL of the InfluxDB instance.

        org (str): InfluxDB organization name.

        token (str): Authentication token for the InfluxDB instance.

        measurement (str, optional): Name of the measurement to use in InfluxDB.
            Defaults to "tide".

    Raises:
        ValueError: If the input `data` is not a DataFrame with a datetime index.

    Example:
        >>> data = pd.DataFrame(
            {
                "name1__°C__bloc1": [1.0, 2.0],
                "name2__W__bloc1": [3.0, 4.0],
            },
            index=pd.to_datetime(["2009-01-01T00:00:00Z", "2009-01-01T01:00:00Z"]),
        )

        >>> push_influx_data(
                data=data,
                tide_tags=['Name', 'Unit', "bloc"],
                bucket='my-bucket',
                url='http://localhost:8086',
                org='my-org',
                token='my-token'
            )
    """

    data = check_and_return_dt_index_df(data)
    influx_df_list = []
    for t, row in data.iterrows():
        df = row.reset_index()
        df.columns = ["full_index", "_value"]
        df[tide_tags] = df["full_index"].str.split("__", expand=True)
        df = df.drop("full_index", axis=1)
        df.index = pd.to_datetime([t] * df.shape[0])
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
