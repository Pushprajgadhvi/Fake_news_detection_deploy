import pandas as pd
df = pd.read_csv("WELFake_Dataset.csv", nrows=30)
print(df[["title", "label"]].to_string())
print("\nLabel value counts:", df["label"].value_counts().to_dict())
