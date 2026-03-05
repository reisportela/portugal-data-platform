
import sqlite3
import pandas as pd

class Metadata:

    def __init__(self,db="metadata.db"):
        self.conn = sqlite3.connect(db)

    def datasets(self):
        return pd.read_sql("SELECT * FROM datasets",self.conn)

    def search(self,keyword):
        q = "SELECT * FROM datasets WHERE name LIKE ?"
        return pd.read_sql(q,self.conn,params=[f"%{keyword}%"])
