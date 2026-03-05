
import pandas as pd
import io
import pyreadstat

def export_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def export_stata(df):
    buffer = io.BytesIO()
    pyreadstat.write_dta(df, buffer)
    buffer.seek(0)
    return buffer
