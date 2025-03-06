Welcome to python-tide's documentation!
====================================

.. image:: ../tide_logo.svg
   :width: 200px
   :align: center


python-tide is a Python library for time series data visualization and pipeline creation, 
with a focus on building data processing pipelines and analyzing data gaps.

GitHub Repository
----------------

The source code for python-tide is available on GitHub:
`github.com/yourusername/python-tide <https://github.com/BuildingEnergySimulationTools/tide/tree/24-add-readthedoc>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api_reference/index
   tutorials/index
   contributing
   changelog

Features
--------

- Hierarchical column naming system (name__unit__bloc__sub_bloc)
- Flexible data selection using tags
- Configurable data processing pipelines
- Advanced gap analysis and visualization
- Interactive time series plotting with multiple y-axes
- Integration with scikit-learn transformers

Quick Example
------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from tide.plumbing import Plumber

   # Create sample data
   data = pd.DataFrame({
       "temp__°C__zone1": [20, 21, np.nan, 23],
       "humid__%HR__zone1": [50, 55, 60, np.nan]
   }, index=pd.date_range("2023", freq="h", periods=4))

   # Define pipeline
   pipe_dict = {
       "pre_processing": {"°C": [["ReplaceThreshold", {"upper": 25}]]},
       "common": [["Interpolate", ["linear"]]]
   }

   # Create plumber and process data
   plumber = Plumber(data, pipe_dict)
   corrected = plumber.get_corrected_data()

   # Analyze gaps
   gaps = plumber.get_gaps_description()

   # Plot data
   fig = plumber.plot(plot_gaps=True)
   fig.show()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 