import pandas as pd

# Load datasets
cleaned_df = pd.read_csv("cleaned_books_dataset.csv")
books_df = pd.read_csv("data/books.csv")
ratings_df = pd.read_csv("ratings.csv")

# Normalize case and strip whitespace for matching
cleaned_df['goodreads_title'] = cleaned_df['goodreads_title'].str.strip().str.lower()
cleaned_df['goodreads_author'] = cleaned_df['goodreads_author'].str.strip().str.lower()
books_df['title'] = books_df['title'].str.strip().str.lower()
books_df['authors'] = books_df['authors'].str.strip().str.lower()

# Merge cleaned books and books dataset on title and author
merged_books_df = pd.merge(
    cleaned_df,
    books_df,
    left_on=["goodreads_title", "goodreads_author"],
    right_on=["title", "authors"],
    how="left"
)

# Merge the above result with ratings dataset on book_id
merged_df = pd.merge(
    merged_books_df,
    ratings_df,
    left_on="book_id",
    right_on="book_id",
    how="left"
)

# Select relevant columns for recommender
user_data = merged_df[["user_id", "book_id", "rating", "goodreads_title", "goodreads_author", "processed_description"]]

# Drop rows where book_id or user_id is missing
user_data = user_data.dropna(subset=["book_id", "user_id"])

# Save to new file
user_data.to_csv("user_data.csv", index=False)

print("user_data.csv created with ratings and book information.")
