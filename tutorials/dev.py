import pandas as pd

# %%
# Dataframe names :
# - Name|group|unit
# - If no pipe, columns string is name
# - If Unit but no group, None must be specified as "group"
# - If Name and group specified without "unit", unit is set to dimensionless

test_df = pd.DataFrame(
    {
        "Tin__°C__building": [10.0, 20.0, 30.0],
        "Text__°C__outdoor": [-1.0, 5.0, 4.0],
        "radiation__W/m2__outdoor": [50, 100, 400],
        "Humidity__%HR": [10, 15, 13],
        "Humidity_2": [10, 15, 13],
        "light__DIMENSIONLESS__building": [100, 200, 300],
        "mass_flwr__m3/h__hvac": [300, 500, 600],
    },
    index=pd.date_range("2009", freq="h", periods=3),
)

# %%

# ALL, REMAINDERS, RESAMPLER are protected keywords

pipe_dict = [
    [
        "pre_processing",
        {
            "°C": ["DropThresHold", {"upper_value": 100}],
            "outdoor|W/m2": ["DropDerivative", {"lower_value": -100}],
        },
    ],
    ["common", {"ALL": ["Interpolate", ["linear"]]}],
    ["RESAMPLER", "15min", {"energy": "sum"}],
    [
        "Compute_energy",
        {
            "DataCombiner": [
                ["T1", "T2", "M"],
                '("T1"-"T2") * M * 2001',
                "Air_flow_energy|hvac|J",
                True,
            ],
        },
    ],
]
