import csv

import pandas
from tqdm import tqdm
from transformers import pipeline,AutoTokenizer
import re

def sentiment_analysis(output_file,):
    fields = [
        "recommendationid","review", "sentiment", "emotion"
    ]
    print("Loading Data")

    df = pandas.read_csv("../data/binding_of_isaac_reviews.csv")
    # df = pandas.read_csv("../data/mini_reviews.csv")
    reviews = df["review"]
    ids = df["recommendationid"]

    #Load the emotion analysis model and its tokenizer
    print("Initializing Emotion Model..")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    emotion_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

    #Load the emotion sentiment model and its tokenizer
    print("Initializing Sentiment Model..")
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    sentiment_tokenizer= AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    #Load the summarization model and its tokenizer
    print("Initializing Summarization Model..")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_tokenizer= AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    with (open(output_file, "w", newline="", encoding="utf-8") as file):
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        #Process bar
        with tqdm(total=len(reviews), desc="Processing emotions", unit=" Reviews Analyzed") as pbar:
            for i, review in enumerate(reviews):
                try:
                    # Check if review is a not null string
                    if pandas.notna(review) and isinstance(review, str):
                        # Replaces any sequence of spaces with a single space
                        review = re.sub(r'\s+', ' ', review).strip()

                        # Tokenize the text and calculate the number of tokens
                        tokens = summary_tokenizer.encode(review)
                        #Max number of token accepted by the summarizer model
                        if len(tokens) <= 1024 :
                            e_tokens = emotion_tokenizer.encode(review)
                            s_tokens = sentiment_tokenizer.encode(review)

                            # If the number of tokens exceeds the limit, summarize the text
                            if len(e_tokens) > 512 or len(s_tokens) > 512:
                                # Incompatibility between models
                                if len(tokens) < 150:
                                    summary = summarizer(review, max_length=10, min_length=1, do_sample=False)
                                else:
                                    summary = summarizer(review, max_length=150, min_length=20, do_sample=False)
                                review=summary[0]["summary_text"]

                            row = {
                                "recommendationid": ids[i],
                                "review": review,
                                # Emotion and sentiment evaluation
                                "sentiment": sentiment_analyzer(review),
                                "emotion": emotion_model(review),
                            }
                            writer.writerow(row)
                    else:
                        #If the review is too lenghty, drop it from the dataframe
                        ids.drop(i, inplace=True)
                        reviews.drop(i, inplace=True)
                except Exception as e:
                    ids.drop(i, inplace=True)
                    reviews.drop(i, inplace=True)

                pbar.update(1)





sentiment_analysis("../data/Output_SentimentAnalysis.csv")