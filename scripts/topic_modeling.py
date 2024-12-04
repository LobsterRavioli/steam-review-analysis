import csv

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm


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





def choose_version():
    """
    Prompts the user to select between a mini version (quick test) or a full version
    (complete dataset). Returns the selected version as a string ('mini' or 'full').
    """
    print("Please choose the version to run:")
    print("1. Run Mini Version (for a quick test with a small dataset [~3000 entries])")
    print("2. Run Full Version (for running with a complete dataset [~120000 entries])")
    while True:
        # Get user input and strip any surrounding whitespace
        choice = input("Enter '1' for Mini Version or '2' for Full Version: ").strip()
        # Validate the input and return the corresponding version
        if choice == '1':
            print(f"You have selected the \"mini\" version.")
            df = pd.read_csv('../data/mini_reviews.csv')
            output_file = "../data/miniTopic_reviews.csv"
            break
        elif choice == '2':
            print(f"You have selected the \"full\" version.")
            df = pd.read_csv('../data/binding_of_isaac_reviews.csv')
            output_file = "../data/Topic_reviews.csv"
            break
        else:
            # Provide feedback on invalid input
            print("Invalid input. Please choose '1' for Mini Version or '2' for Full Version.")
    return df,output_file

if __name__ == "__main__":
    df,output_file=choose_version()

    # Preprocessing
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Load dataset
    print("Loading data...")
    reviews = df['review'].dropna()

    # Data cleaning and tokenization
    print("Cleaning data...")
    cleaned_reviews = [' '.join([word for word in review.lower().split() if word not in stop_words]) for review in
                       reviews]

    # Term frequency
    print("Calculating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cleaned_reviews)

    # LDA per topic modeling
    n_components = 3
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42, verbose=1)
    for _ in tqdm(range(1), desc="Fitting LDA Model", unit=" iteration"):
        lda.fit(tfidf_matrix)



    generate_topics(lda, vectorizer.get_feature_names_out(), 11,output_file)

