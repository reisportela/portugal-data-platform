
import pandas as pd

class SearchEngine:

    def __init__(self, file):
        self.df = pd.read_csv(file)

    def search(self, query):

        query = query.lower()

        mask = (
            self.df["dataset_name"].str.lower().str.contains(query, na=False)
            | self.df["domain"].str.lower().str.contains(query, na=False)
            | self.df["institution"].str.lower().str.contains(query, na=False)
        )

        results = self.df[mask]

        return results
