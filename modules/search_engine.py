
import pandas as pd
from rapidfuzz import process

class SearchEngine:

    def __init__(self, file):
        self.index = pd.read_csv(file)

    def search(self, query):
        titles = self.index["dataset_name"].tolist()
        matches = process.extract(query, titles, limit=5)

        results = []
        for m in matches:
            row = self.index[self.index["dataset_name"] == m[0]].iloc[0]
            results.append(row)

        return results
