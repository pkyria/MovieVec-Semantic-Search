import pandas as pd
import numpy as np
from config import PATH_TO_DATA

class DataLoader:
    def __init__(self, path=PATH_TO_DATA):
        self.path = path
        self.df = pd.read_csv(self.path)

    def load_data(self):
        """Load the dataset from the specified path and return as a list of texts."""
        try:
            print(f"Loading data...")
            self.df = pd.read_csv(self.path)
            # Combine relevant text fields into a single text field for embedding
            self.df['text'] = self.df['title'].fillna("").astype(str)  # start with title
            # Fill NaNs with empty strings and ensure all are strings
            for col in ["overview", "keywords", "genres", "cast_crew"]:
                self.df[col] = self.df[col].fillna("").astype(str)   # ensure all strings
                self.df["text"] = self.df["text"] + ". " + self.df[col]
            data = self.df['text'].tolist() # Convert to list for embedding

            print(f"Loaded {len(data)} movie records.")

            return data
        
        except FileNotFoundError:
            print("File not found. Please check the path.")
            return None
