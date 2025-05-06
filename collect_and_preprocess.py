import requests
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

GOOGLE_BOOKS_API_KEY = "AIzaSyDuGgORd_2tGky5eLBdnOskzXccsOh_srE" 

# Function to fetch book info from Google Books API
def fetch_google_books_data(title, author=None):
    query = f'intitle:{title}'
    if author:
        query += f'+inauthor:{author}'
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&key={GOOGLE_BOOKS_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        items = response.json().get("items")
        if items:
            volume_info = items[0]["volumeInfo"]
            return {
                "title": volume_info.get("title", ""),
                "authors": volume_info.get("authors", []),
                "description": volume_info.get("description", ""),
                "categories": volume_info.get("categories", []),
                "averageRating": volume_info.get("averageRating", None),
                "ratingsCount": volume_info.get("ratingsCount", None),
                "pageCount": volume_info.get("pageCount", None),
                "publishedDate": volume_info.get("publishedDate", "")
            }
    return {}

# Function to enrich Goodreads data using Google Books API
def enrich_goodreads_data(df, max_books=100):
    enriched_books = []
    for idx, row in df.head(max_books).iterrows():
        print(f"Fetching {row['title']} by {row['authors']}")
        book_data = fetch_google_books_data(row["title"], row["authors"])
        enriched_books.append({
            "goodreads_title": row["title"],
            "goodreads_author": row["authors"],
            "google_info": book_data,
        })
    return pd.DataFrame(enriched_books)

# Function to preprocess book descriptions
def preprocess_description(text):
    if not text:
        return ""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(words)

# Main data collection and preprocessing pipeline
def main():
    goodreads_path = "data/books.csv"
    
    # Step 1: Load Goodreads data
    print("Loading Goodreads dataset...")
    goodreads_df = pd.read_csv(goodreads_path)
    print("Goodreads columns:", goodreads_df.columns)

    # Step 2: Enrich with Google Books API
    print("Enriching data with Google Books API...")
    enriched_df = enrich_goodreads_data(goodreads_df, max_books=10000)

    # Step 3: Extract and preprocess descriptions
    print("Preprocessing descriptions...")
    enriched_df["processed_description"] = enriched_df["google_info"].apply(
        lambda info: preprocess_description(info.get("description", ""))
    )

    # Step 4: Save to CSV
    output_path = "enriched_books.csv"
    enriched_df.to_csv(output_path, index=False)
    print(f"Saved enriched and preprocessed data to {output_path}")

if __name__ == "__main__":
    main()
