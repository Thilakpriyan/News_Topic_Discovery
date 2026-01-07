import pandas as pd
from src.preprocessing import preprocess_texts
import os
# Correct dataset path
df = pd.read_csv("data/raw/bbc_news/bbc-text.csv")

# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_texts)

# Save cleaned data
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/clean_bbc_text.csv", index=False)

print(df[["text", "clean_text"]].head())

