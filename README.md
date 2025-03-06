<p align="center">
  <img src="https://raw.githubusercontent.com/BuildingEnergySimulationTools/tide/main/tide_logo.svg" alt="CorrAI" width="200"/>
</p>



[![PyPI](https://img.shields.io/pypi/v/python-tide?label=pypi%20package)](https://pypi.org/project/python-tide/)
[![Static Badge](https://img.shields.io/badge/python-3.10_%7C_3.12-blue)](https://pypi.org/project/python-tide/)
[![codecov](https://codecov.io/gh/BuildingEnergySimulationTools/tide/branch/main/graph/badge.svg?token=F51O9CXI61)](https://codecov.io/gh/BuildingEnergySimulationTools/tide)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/python-tide/badge/?version=latest)](https://python-tide.readthedocs.io/en/latest/?badge=latest)

## Pipeline Development and Data Visualization for Time Series in Physical Measurements

Tide is a tool for developing data processing pipelines
and visualizing time series data,
particularly suited for physical measurements.
Key features include:

- __Efficient Data Management__
    - Organize and select data using a tagging system

- __Pipeline Construction__
    - Store and retrieve pipelines easily with JSON-based dictionary structures
    - Build dynamic pipelines that adjust based on the selected data

- __Interactive Visualization__
    - Create interactive plots to explore data ([plotly](https://plotly.com/))
    - Visualize pipeline or slices of pipelines effects on data

- Custom Data Enrichment
    - Integrate external weather data sources
    - Implement autoregressive models for gaps filling
    - Develop and incorporate custom data processors

Uses [pandas](https://pandas.pydata.org/) DataFrames and Series for robust data
handling.
[bigtree](https://github.com/kayjan/bigtree) for tags and data selection.
[Scikit-learn](https://scikit-learn.org/stable/)'s API for pipeline construction.

## Getting started
### 1- Install ⬇️
````
pip install python-tide
````

### 2- Load and format data 🌲

To begin, load your time series data into a pandas DataFrame, ensuring the index is a
DateTimeIndex:

```python
df = pd.read_csv(
    "https://raw.githubusercontent.com/BuildingEnergySimulationTools/tide/main/tutorials/getting_started_ts.csv",
    parse_dates=True,
    index_col=0
)
```

Rename columns using Tide's tagging system.
The format is:
<code>name__unit__bloc__sub_bloc</code> with tags separated by double underscores.
The order of the tags matters.
The order of tags is important, and you can use "OTHER" as a placeholder
You can use one or several tags.

```python
df.columns = ["Tin__°C__Building", "Text__°C__Outdoor", "Heat__W__Building"]
```

Plumber objects are used to help us with pipelines building and data visualization

```python
from tide.plumbing import Plumber

plumber = Plumber(df)
```

Display the data organization as a tree:

```python
plumber.show()
```

Select data using tags:

```python
plumber.get_corrected_data("°C")
plumber.get_corrected_data("Building")
plumber.get_corrected_data("Tin")
```

### 3- Visualizing  data 📈

Show data availability:

```python
plumber.plot_gaps_heatmap(time_step='d')
```

Plot time series with missing data highlighted:

```python
fig = plumber.plot(plot_gaps=True)
fig.show(None)
```

### 4- Building and testing Pipelines 🛠️

Create a pipeline dictionary:

````python
pipe_dict = {
    "step_1": [["Common_proc_1"], ["Common_proc_2", ["arg1", "arg2"]]],
    "step_2": {
        "selection_1": [["Proc_selection_1", {"arg": "arg_value"}]]
    }
}
````

__Pipeline Rules:__

- Use dictionaries for pipeline description
- Keys represent pipeline steps ex. <code>"step_1"</code>
- Step values can be lists (apply to all columns) or dicts (filter columns)
- Processing objects are listed as [class_name, arguments]

__Example Pipeline:__

- Resample data to 15-minute intervals
- Interpolate temperature gaps ≤ 3 hours
- Fill large Tin gaps using Autoregressive STLForecast

````python
pipe_dict = {
    "resample_15min": [["Resample", ["15min"]]],
    "interpolate_temps": {
        "°C": [["Interpolate", {"gaps_lte": "3h"}]]
    },
    "ar_tin": {
        "Tin": [
            [
                "FillGapsAR",
                {
                    "model_name": "Prophet",
                    "resample_at_td": "1h",
                    "gaps_gte": "3h",
                    "gaps_lte": "3d"
                }
            ]
        ]
    }
}

plumber.pipe_dict = pipe_dict
````

Get pipeline using <code>get_pipeline</code> method.

````python
plumber.get_pipeline(verbose=True)
````

Get pipelines for specific columns

````python
plumber.get_pipeline(select="Building", verbose=True)
````

Visualize pipeline effects:

````python
plumber.plot(
    steps=None,
    plot_gaps=True,
    steps_2=slice(None, "interpolate_temps"),
    plot_gaps_2=True,
    verbose=True
)
````

__Step Arguments:__

- <code>None</code>: No operation (Identity)
- <code>str</code>: Process until named step
- <code>list[str]</code>: Perform specified steps
- <code>slice</code>: Process a slice of the pipeline

Compare full pipeline to raw data:

````python
plumber.plot(
    steps=None,
    plot_gaps=True,
    steps_2=slice(None),
    plot_gaps_2=True,
    verbose=True
)
````

### Sponsors
<table style="border-collapse: collapse;">
<tr style="border: 1px solid transparent;">
<td width="150" >
<img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg" alt="eu_flag" width="150"/>
</td>
<td>
The development of this library has been supported by ENSNARE Project, which
has received funding from the European Union's Horizon 2020 Research and Innovation
Programme under Grant Agreement No. 953193. The sole responsibility for the content of
this library lies entirely with the author's view. The European Commission is not
responsible for any use that may be made of the information it contains. 
</td>
</tr>
</table>



