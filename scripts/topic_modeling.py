import csv

import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm

from scripts.utility import game_selection


def generate_topics(model, feature_names, num_top_words,output_file):
    fields = [
        "topic_number","word","weight"
    ]
    with (open(output_file, "w", newline="", encoding="utf-8") as file):
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for topic_idx, topic in enumerate(model.components_):
            #Heaviest feature first
            for i in topic.argsort()[:-num_top_words - 1:-1]:
                word = feature_names[i]  # Prendi il nome della parola
                if word != "game":
                    weight = topic[i]
                    writer.writerow({"topic_number": topic_idx, "word": word, "weight": weight})


if __name__ == "__main__":
    foldername = game_selection()
    df = pd.read_csv(f'../data/{foldername}/Reviews.csv')
    output_file = f'../data/{foldername}/Topic_modeling.csv'

    # Preprocessing
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()


    # Load dataset and drop null reviews
    print("Loading data...")
    reviews = df['review'].dropna()


    print("Cleaning data...")
    # Clean, remove stop words, and lemmatize
    cleaned_reviews= [' '.join([lemmatizer.lemmatize(word) for word in review.lower().split() if word not in stop_words])
                      for review in reviews]

    # Term frequency
    print("Calculating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cleaned_reviews)

    # LDA per topic modeling
    n_components = 3
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42, verbose=1)
    for _ in tqdm(range(1), desc="Fitting LDA Model", unit=" iteration"):
        lda.fit(tfidf_matrix)

    print(f"perplexity:{lda.perplexity(tfidf_matrix)}")

    generate_topics(lda, vectorizer.get_feature_names_out(), 11,output_file)

