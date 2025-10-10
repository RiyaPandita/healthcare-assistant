# tests/test_interactions.py
import pandas as pd

def test_interactions_csv_load():
    df = pd.read_csv("data/interactions.csv")
    assert {"drug_a","drug_b","level","note"}.issubset(df.columns)