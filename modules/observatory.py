
import pandas as pd

def fake_indicator():
    return pd.DataFrame({
        "year":[2019,2020,2021,2022,2023],
        "value":[6.5,7.1,6.8,6.0,5.9]
    })
