import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load all data from user_data.csv
user_data_df = pd.read_csv('user_data.csv')

# Get all unique books with their descriptions
books_df = user_data_df.drop_duplicates(subset='book_id')[['book_id', 'processed_description']].dropna().reset_index(drop=True)
book_id_to_index = {book_id: idx for idx, book_id in enumerate(books_df['book_id'])}

# Vectorize book descriptions for content similarity (done once globally)
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
tfidf_matrix = tfidf.fit_transform(books_df['processed_description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Predict ratings using user-specific cosine similarity to previously rated books
def predict_cf_ratings(user_id, user_data_df, books_df, cosine_sim, top_n=10):
    user_ratings = user_data_df[user_data_df['user_id'] == user_id][['book_id', 'rating']]
    rated_books = user_ratings['book_id'].tolist()
    rated_indices = [book_id_to_index[bid] for bid in rated_books if bid in book_id_to_index]

    if not rated_indices:
        return []

    # Average similarity to rated books for all others
    sim_scores = np.mean(cosine_sim[rated_indices], axis=0)

    # Filter out already rated books
    candidate_indices = [i for i in range(len(sim_scores)) if books_df.iloc[i]['book_id'] not in rated_books]
    predictions = [(books_df.iloc[i]['book_id'], sim_scores[i]) for i in candidate_indices]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]

# Content-based recommendation using cosine similarity of descriptions
def get_content_based_recommendations(book_id, cosine_sim, books_df, top_n=5):
    if book_id not in book_id_to_index:
        return []
    idx = book_id_to_index[book_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices]['book_id'].tolist()

# Hybrid recommender combining CF and content-based

def hybrid_recommender(user_id, user_data_df, books_df, cosine_sim, top_n=10, cf_weight=0.7, content_weight=0.3):
    cf_recs = predict_cf_ratings(user_id, user_data_df, books_df, cosine_sim, top_n=top_n*2)
    user_ratings_sorted = user_data_df[user_data_df['user_id'] == user_id].sort_values(by='rating', ascending=False)
    rated_books = user_ratings_sorted['book_id'].tolist()
    rated_indices = [book_id_to_index[bid] for bid in rated_books[:3] if bid in book_id_to_index]

    if rated_indices:
        content_scores_array = np.mean(cosine_sim[rated_indices], axis=0)
    else:
        content_scores_array = np.zeros(len(books_df))

    final_scores = []
    breakdowns = []
    max_cf = max([score for _, score in cf_recs], default=1)
    for book_id, cf_score in cf_recs:
        idx = book_id_to_index[book_id]
        content_score = content_scores_array[idx]  # direct similarity
        scaled_cf = cf_score / max_cf if max_cf != 0 else 0
        final_score = cf_weight * scaled_cf + content_weight * content_score
        final_scores.append((book_id, final_score))
        breakdowns.append((book_id, scaled_cf, content_score, final_score))

    sorted_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_n]
    breakdown_dict = {
        book_id: {
            'cf': cf_part,
            'content': content_part,
            'raw': score
        }
        for (book_id, cf_part, content_part, score) in breakdowns
    }

    return sorted_scores, breakdown_dict

# Main execution
if __name__ == "__main__":
    user_id = int(input("Enter user ID: "))

    recommendations, breakdowns = hybrid_recommender(user_id, user_data_df, books_df, cosine_sim, top_n=10)

    print("\nBooks the user has read:")
    user_books = user_data_df[user_data_df['user_id'] == user_id][['book_id', 'goodreads_title']].drop_duplicates()
    for _, row in user_books.iterrows():
        print(f"{row['goodreads_title']} (Book ID: {row['book_id']})")

    print("\nTop Recommendations:")
    for book_id, score in recommendations:
        title = user_data_df[user_data_df['book_id'] == book_id].iloc[0]['goodreads_title']
        details = breakdowns[book_id]
        print(f"{title} (Book ID: {book_id}) — Score: {score:.2f} [CF: {details['cf']:.2f}, Content: {details['content']:.2f}]")
