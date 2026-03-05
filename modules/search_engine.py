
import pandas as pd
from rapidfuzz import process

class SearchEngine:

    def __init__(self,file):
        self.df = pd.read_csv(file)

    def search(self,query):
        titles = self.df["dataset_name"].tolist()
        matches = process.extract(query,titles,limit=5)
        results=[]
        for m in matches:
            row = self.df[self.df["dataset_name"]==m[0]].iloc[0]
            results.append(row)
        return results
