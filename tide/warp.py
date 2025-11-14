import json
import requests

import datetime as dt

import pandas as pd
import numpy as np

def get_warp_data(
    endpoint: str,
    token: str,
    class_names: str | list[str],
    labels: str | list[str],
    start: str | pd.Timestamp | dt.datetime,
    stop: str | pd.Timestamp | dt.datetime,
    timestep: str = None,
    tz_info: str = "UTC",
    count: int = None,
):
    class_names = [class_names] if isinstance(class_names, str) else class_names
    class_regex = f"~({ '|'.join(class_names) })"

    start = pd.to_datetime(start) if isinstance(start, str) else start
    stop = pd.to_datetime(stop) if isinstance(stop, str) else stop

    # Format pour TSELEMENTS-> : [année, mois, jour, heure, minute, seconde]
    start_list = (f"[ "
                  f"{start.year} "
                  f"{start.month:02d} "
                  f"{start.day:02d} "
                  f"{start.hour:02d} "
                  f"{start.minute:02d} "
                  f"{start.second:02d} ]"
                  )

    stop_list = (f"[ "
                 f"{stop.year} "
                 f"{stop.month:02d} "
                 f"{stop.day:02d} "
                 f"{stop.hour:02d} "
                 f"{stop.minute:02d} "
                 f"{stop.second:02d} ]"
                 )

    query = f"""
    {{
      'token'  '{token}'
      'class'  '{class_regex}'
      'labels' {{ {" ".join(f"'{lab}'" for lab in labels)} }}
      'end'    {stop_list} TSELEMENTS->
      'start'  {start_list} TSELEMENTS->
    """

    if timestep:
        query += f"""
        'timestep'  '{timestep}'
        """

    if count:
        query += f"""'count' {count}
        """

    query += "} FETCH"
    # print(query)
    response = requests.post(endpoint, data=query)

    if response.ok:
        data = json.loads(response.content)
        dfs = {}
        for gts in data[0]:
            arr = np.array(gts["v"])
            dfs[gts["c"]] = pd.Series(
                arr[:, 1], index=pd.to_datetime(arr[:, 0], unit="us"), name=gts["c"]
            )

        res = pd.concat(dfs.values(), axis=1)
        res.index = res.index.tz_localize(tz_info)
        return res

    raise ValueError("❌ Error", response.status_code, response.text)
