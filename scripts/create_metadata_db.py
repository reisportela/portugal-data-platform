
import sqlite3
import pandas as pd

conn = sqlite3.connect("metadata.db")
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS datasets(id INTEGER PRIMARY KEY,name TEXT,institution TEXT)")

df = pd.read_csv("dataset_catalog.csv")

for _,r in df.iterrows():
    c.execute("INSERT INTO datasets(name,institution) VALUES (?,?)",(r["dataset_name"],r["institution"]))

conn.commit()
conn.close()
