<p align="center">
  <img src="https://raw.githubusercontent.com/BuildingEnergySimulationTools/tide/main/tide_logo.svg" alt="CorrAI" width="200"/>
</p>



[![PyPI](https://img.shields.io/pypi/v/corrai?label=pypi%20package)](https://pypi.org/project/corrai/)
[![Static Badge](https://img.shields.io/badge/python-3.10_%7C_3.12-blue)](https://pypi.org/project/corrai/)
[![codecov](https://codecov.io/gh/BuildingEnergySimulationTools/tide/branch/main/graph/badge.svg?token=F51O9CXI61)](https://codecov.io/gh/BuildingEnergySimulationTools/tide)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Pipeline Development and Data Visualization for Time Series in Physical Measurements

Tide is a powerful tool for developing data processing pipelines and visualizing time series data, particularly suited for physical measurements. Key features include:

 - __Efficient Data Management__
    - Organize and select data using a tagging system
        
 - __Pipeline Construction__
   - Store and retrieve pipelines easily with JSON-based dictionary structures
   - Build dynamic pipelines that adjust based on the selected data
 
 - __Interactive Visualization__
   - Create interactive plots to explore data trends and patterns
   - Visualize pipeline effects on data in real-time
        
 - Custom Data Enrichment
   - Integrate external weather data sources
   - Implement autoregressive models for time series analysis
   - Develop and incorporate custom data processors
 
 Uses pandas DataFrames and Series for robust data handling. big-tree for tags and data selection.
 Scikit-learn's API for pipeline construction.

## Getting started
### 1- Load and format data 🌲
Load time series using pandas DataFrame, make sure the index is a DateTimeIndex:

```python
df = pd.read_csv(
    "https://raw.githubusercontent.com/BuildingEnergySimulationTools/tide/main/tutorials/getting_started_ts.csv",
    parse_dates=True,
    index_col=0
)
```

Rename columns using tags. Tides allows you to use the folowing tags : <code>name__unit__bloc__sub_bloc</code>
tags are separeted by a double under score. The order of the tags matters. You can use one or sevral tags.
If you d'ont want to specify a tag for a given columns, you can replace it with <code>"OTHER"</code>. ex: <code>"temperature__°C__OTHER__Outdoor"</code>
Following the example :

```python
df.columns = ["Tin__°C__Building", "Text__°C__Outdoor", "Heat__W__Building"]
```

We use a Plumber object to help us build pipelines and visualise the data
```python
from tide.plumbing import Plumber
plumber = Plumber(df)
```

The tags organise the data as a tree that can be displayed:
```python
plumber.show()
```

Data can now be selected using one or severall tags. You can try the following command.

```python
plumber.get_corrected_data("°C")
plumber.get_corrected_data("Building")
plumber.get_corrected_data("Tin")
```

Note that no pipeline description have been given. <code>get_corrected_data</code> returns
raw data for selected columns.

### 2- Plot data 📈
Data availability can be shown:
```python
plumber.plot_gaps_heatmap(time_step='d')
```

Time series can be plot using Plumber <code>plot</code> method. By default, it will
organize the y axis by time series units. The argument <code>plot_gaps_1 = True</code>
will display the missing data using transparent blue surfaces.

```python
fig = plumber.show(plot_gaps_1=True)
fig.show()
```

### 3- Building and testing Pipelines 🔧
Pipelines description must follow the following rules :
   - Pipelines are written as dictionaries
   - Dictionaries keys are the steps of the pipeline
   - For each steps :
     - If 

