# drop_and_save.py

import pandas as pd

# Load your enriched dataset (make sure the path matches your actual file)
df = pd.read_csv("enriched_books.csv")

# Keep only the relevant columns
columns_to_keep = ["goodreads_title", "goodreads_author", "processed_description"]
clean_df = df[columns_to_keep].dropna()

# Save to a new CSV file
clean_df.to_csv("cleaned_books_dataset.csv", index=False)
print("Saved cleaned dataset to 'cleaned_books_dataset.csv'")
