Installation
============

python-tide requires Python 3.10 or later.

Using pip
---------

The recommended way to install python-tide is via pip:

.. code-block:: bash

   pip install python-tide

This will install python-tide and all its dependencies.

From Source
----------

To install python-tide from source, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/BuildingEnergySimulationTools/tide.git
   cd tide
   pip install -e .

Development Installation
----------------------

For development, you'll want to install additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This will install all development dependencies including:

- pytest for testing
- sphinx for documentation
- pre-commit for code quality
- other development tools

Dependencies
-----------

Core Dependencies
~~~~~~~~~~~~~~~

- numpy>=1.22.4
- pandas>=2.0.0
- scipy>=1.9.1
- bigtree>=0.21.3
- scikit-learn>=1.2.2
- statsmodels>=0.14.4
- matplotlib>=3.5.1
- plotly>=5.3.1
- requests>=2.32.3
- influxdb-client>=1.48.0
- prophet>=1.1.6

Optional Dependencies
~~~~~~~~~~~~~~~~~~~

For development and documentation:

- pytest
- sphinx
- sphinx-rtd-theme
- sphinx-autodoc-typehints
- myst-parser
- nbsphinx
- sphinx-copybutton
- pre-commit
- bump2version 