import pandas as pd
import matplotlib.pyplot as plt
import json
from wordcloud import WordCloud
import seaborn as sns
from pandas.plotting import parallel_coordinates
import os
import ast

def load_and_process_data(data):
    """
    Load and process the dataset for analysis.
    Parses 'sentiment' and 'emotion' columns.
    """
    df = pd.DataFrame(data)
    df['sentiment_label'] = df['sentiment'].apply(lambda x: ast.literal_eval(x)[0]['label'])
    df['sentiment_score'] = df['sentiment'].apply(lambda x: ast.literal_eval(x)[0]['score'])
    df['emotion_label'] = df['emotion'].apply(lambda x: ast.literal_eval(x)[0]['label'])
    df['emotion_score'] = df['emotion'].apply(lambda x: ast.literal_eval(x)[0]['score'])
    return df

def plot_sentiment_trend_over_time(df, output_dir):
    """
    Plot the average sentiment score over time and save it locally.
    Assumes a 'timestamp' column in the dataset with datetime format.
    """
    if 'timestamp' not in df:
        print("The dataset does not contain a 'timestamp' column for temporal analysis.")
        return

    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_sentiment = df.groupby('date')['sentiment_score'].mean()
    plt.figure(figsize=(10, 6))
    daily_sentiment.plot()
    plt.title('Sentiment Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sentiment_trend_over_time.png'))
    plt.close()

def plot_sentiment_trend_over_time(df, output_dir):
    """
    Plot the average sentiment score over time and save it locally.
    Assumes a 'timestamp' column in the dataset with datetime format.
    """
    if 'timestamp' not in df:
        print("The dataset does not contain a 'timestamp' column for temporal analysis.")
        return

    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_sentiment = df.groupby('date')['sentiment_score'].mean()
    plt.figure(figsize=(10, 6))
    daily_sentiment.plot()
    plt.title('Sentiment Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sentiment_trend_over_time.png'))
    plt.close()

def plot_sentiment_vs_review_length(df, output_dir):
    """
    Create a scatter plot for sentiment scores vs review length.
    """
    df['review_length'] = df['review'].apply(len)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['review_length'], df['sentiment_score'], alpha=0.5)
    plt.title('Sentiment Score vs Review Length')
    plt.xlabel('Review Length')
    plt.ylabel('Sentiment Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sentiment_vs_review_length.png'))
    plt.close()


def plot_sentiment_vs_review_length(df, output_dir):
    """
    Create a scatter plot for sentiment scores vs review length.
    """
    df['review_length'] = df['review'].apply(len)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['review_length'], df['sentiment_score'], alpha=0.5)
    plt.title('Sentiment Score vs Review Length')
    plt.xlabel('Review Length')
    plt.ylabel('Sentiment Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sentiment_vs_review_length.png'))
    plt.close()


def plot_positive_negative_ratio(df, output_dir):
    """
    Create a pie chart showing the ratio of positive to negative reviews.
    Assumes 'sentiment_label' has values that can be categorized as positive or negative.
    """
    positive_labels = ['4 star', '5 star']
    negative_labels = ['1 star', '2 star']

    df['sentiment_type'] = df['sentiment_label'].apply(
        lambda x: 'Positive' if x in positive_labels else ('Negative' if x in negative_labels else 'Neutral')
    )
    sentiment_counts = df['sentiment_type'].value_counts()
    plt.figure(figsize=(8, 8))
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, explode=[0.1 if i == 'Positive' else 0 for i in sentiment_counts.index])
    plt.title('Positive vs Negative Reviews Ratio')
    plt.ylabel('')
    plt.savefig(os.path.join(output_dir, 'positive_negative_ratio.png'))
    plt.close()


def plot_emotion_sentiment_heatmap(df, output_dir):
    """
    Create a heatmap showing the frequency of emotion and sentiment combinations.
    """
    heatmap_data = df.groupby(['sentiment_label', 'emotion_label']).size().unstack(fill_value=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues')
    plt.title('Heatmap of Emotion and Sentiment Frequencies')
    plt.xlabel('Emotion')
    plt.ylabel('Sentiment')
    plt.savefig(os.path.join(output_dir, 'emotion_sentiment_heatmap.png'))
    plt.close()

def plot_sentiment_vs_emotion_scores(df, output_dir):
    """
    Create a scatter plot for sentiment scores vs emotion scores and save it locally.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(df['sentiment_score'], df['emotion_score'], alpha=0.7)
    plt.title('Sentiment Scores vs Emotion Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Emotion Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sentiment_vs_emotion_scores.png'))
    plt.close()

def plot_average_emotion_scores_by_sentiment(df, output_dir):
    """
    Create a bar chart for average emotion scores by sentiment and save it locally.
    """
    avg_scores = df.groupby('sentiment_label')['emotion_score'].mean()
    plt.figure(figsize=(8, 5))
    avg_scores.plot(kind='bar')
    plt.title('Average Emotion Scores by Sentiment')
    plt.xlabel('Sentiment Label')
    plt.ylabel('Average Emotion Score')
    plt.savefig(os.path.join(output_dir, 'average_emotion_scores_by_sentiment.png'))
    plt.close()



def plot_emotion_scores_by_sentiment(df, output_dir):
    """
    Create a box plot for emotion scores grouped by sentiment labels and save it locally.
    """
    plt.figure(figsize=(10, 6))
    df.boxplot(column='emotion_score', by='sentiment_label', grid=False)
    plt.title('Emotion Scores by Sentiment')
    plt.suptitle('')  # Remove the automatic title
    plt.xlabel('Sentiment Label')
    plt.ylabel('Emotion Score')
    plt.savefig(os.path.join(output_dir, 'emotion_scores_by_sentiment.png'))
    plt.close()





def plot_sentiment_distribution(df, output_dir):
    """
    Create a bar chart for sentiment distribution and save it locally.
    """
    plt.figure(figsize=(8, 5))
    df['sentiment_label'].value_counts().plot(kind='bar')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()

def plot_emotion_distribution(df, output_dir):
    """
    Create a bar chart for emotion distribution and save it locally.
    """
    plt.figure(figsize=(8, 5))
    df['emotion_label'].value_counts().plot(kind='bar')
    plt.title('Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
    plt.close()

def plot_sentiment_scores(df, output_dir):
    """
    Create a histogram for sentiment scores and save it locally.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(df['sentiment_score'], bins=10, alpha=0.7)
    plt.title('Sentiment Scores Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'sentiment_scores_distribution.png'))
    plt.close()

def plot_emotion_scores(df, output_dir):
    """
    Create a histogram for emotion scores and save it locally.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(df['emotion_score'], bins=10, alpha=0.7)
    plt.title('Emotion Scores Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'emotion_scores_distribution.png'))
    plt.close()

def plot_emotion_by_sentiment(df, output_dir):
    """
    Create a grouped bar chart for emotion by sentiment and save it locally.
    """
    plt.figure(figsize=(10, 6))
    df.groupby(['sentiment_label', 'emotion_label']).size().unstack().plot(kind='bar', stacked=True)
    plt.title('Emotion by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.legend(title="Emotion")
    plt.savefig(os.path.join(output_dir, 'emotion_by_sentiment.png'))
    plt.close()

# emotion and sentiment

def emotion_pie_chart(df, output_dir):
    # Pie chart for emotion labels
    emotion_counts = df['emotion_label'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Emotion Distribution')
    plt.savefig(os.path.join(output_dir, 'emotion_scores_by_sentiment.png'))
    plt.close()






def plot_topic_word_heatmap(df, output_dir, num_words=10):
    """
    Plots a heatmap of top words and their weights for each topic.
    """
    # Filter the top words per topic
    top_words = df.groupby('topic_number').apply(lambda x: x.nlargest(num_words, 'weight')).reset_index(drop=True)

    # Pivot the data for heatmap
    heatmap_data = top_words.pivot(index='word', columns='topic_number', values='weight').fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=False, cmap='Blues', cbar_kws={'label': 'Weight'})
    plt.title(f'Topic-Word Heatmap (Top {num_words} Words)')
    plt.xlabel('Topic Number')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_word_heatmap.png'))


def plot_word_weights_scatter(df, output_dir, topic_number ):
    """
    Creates a scatter plot of word weights for a specific topic.
    """

    topic_df = df[df['topic_number'] == topic_number]

    plt.figure(figsize=(10, 6))
    plt.scatter(topic_df['word'], topic_df['weight'], color='coral', alpha=0.7)
    plt.title(f'Scatter Plot of Word Weights for Topic {topic_number}')
    plt.xlabel('Words')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'topic_{topic_number}_word_weights_scatter.png'))



def plot_parallel_coordinates(df, output_dir, num_words=5):
    """
    Plots a parallel coordinate chart for top words across topics.
    """
    top_words = df.groupby('topic_number').apply(lambda x: x.nlargest(num_words, 'weight')).reset_index(drop=True)
    pivot_df = top_words.pivot(index='word', columns='topic_number', values='weight').fillna(0)
    pivot_df['word'] = pivot_df.index

    plt.figure(figsize=(12, 6))
    parallel_coordinates(pivot_df, class_column='word', colormap='viridis', alpha=0.7)
    plt.title(f'Parallel Coordinates Plot for Top {num_words} Words')
    plt.xlabel('Topic Number')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parallel_coordinates.png'))
    plt.close()



def plot_topic_distribution(df, output_dir):
    """
    Plots the distribution of total weights across topics.
    """
    topic_weights = df.groupby('topic_number')['weight'].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    topic_weights.plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title('Topic Distribution by Total Weight')
    plt.xlabel('Topic Number')
    plt.ylabel('Total Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_distribution.png'))
    plt.close()


def plot_cumulative_weight_distribution(df, topic_number, output_dir='topic_'):
    """
    Plots the cumulative distribution of word weights for a specific topic.
    """
    topic_df = df[df['topic_number'] == topic_number].sort_values(by='weight', ascending=False)
    topic_df['cumulative_weight'] = topic_df['weight'].cumsum() / topic_df['weight'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(topic_df['word'], topic_df['cumulative_weight'], marker='o', color='green')
    plt.title(f'Cumulative Weight Distribution for Topic {topic_number}')
    plt.xlabel('Words')
    plt.ylabel('Cumulative Weight')
    plt.xticks(rotation=45)
    plt.axhline(0.8, color='red', linestyle='--', label='80% Weight')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'topic_{topic_number}_topic_distribution.png'))
    plt.close()



# Plot Top Words per Topic
def plot_top_words_per_topic(df, topic_number, output_dir, n=10):
    """
    Plots the top N words for a specific topic.
    """
    topic_df = df[df['topic_number'] == topic_number].nlargest(n, 'weight')

    plt.figure(figsize=(10, 6))
    plt.bar(topic_df['word'], topic_df['weight'], color='skyblue')
    plt.title(f'Top {n} Words for Topic {topic_number}')
    plt.xlabel('Word')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'topic_{topic_number}_topwords.png'))
    plt.close()


# Generate Word Cloud for a Topic
def generate_wordcloud(df, topic_number,output_dir):
    """
    Generates a word cloud for a specific topic.
    """
    topic_df = df[df['topic_number'] == topic_number]
    word_freq = dict(zip(topic_df['word'], topic_df['weight']))

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Topic {topic_number}')
    plt.savefig(os.path.join(output_dir, f'topic_{topic_number}_word_cloud.png'))
    plt.close()


def plot_emotion_distribution_by_rating(df, output_dir):
    """
    Create a grouped bar chart showing emotion distribution by star rating.
    Assumes 'sentiment_label' represents star ratings and 'emotion_label' represents emotions.
    """
    emotion_rating_data = df.groupby(['sentiment_label', 'emotion_label']).size().unstack(fill_value=0)
    emotion_rating_data_percentage = emotion_rating_data.div(emotion_rating_data.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(12, 8))
    emotion_rating_data_percentage.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Emotion Distribution by Star Rating')
    plt.xlabel('Star Rating')
    plt.ylabel('Percentage of Emotions')
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_distribution_by_rating.png'))
    plt.close()

def merge_datasets(reviews_file, timestamps_file, output_file):
    """
    Unisce due dataset basandosi sulla colonna 'recommendationid'.

    Args:
        reviews_file (str): Path al file CSV contenente le recensioni.
        timestamps_file (str): Path al file CSV contenente i timestamp e altre informazioni.
        output_file (str): Path dove salvare il dataset unito.

    Returns:
        pd.DataFrame: Il dataset unito.
    """
    # Carica il dataset delle recensioni
    dataset_reviews = pd.read_csv(reviews_file)

    # Carica il dataset con i timestamp
    dataset_timestamps = pd.read_csv(timestamps_file)

    # Seleziona solo le colonne necessarie dal dataset dei timestamp
    columns_of_interest = ['recommendationid', 'timestamp_created']
    dataset_timestamps = dataset_timestamps[columns_of_interest]

    # Esegui il merge basato sulla colonna 'recommendationid'
    merged_dataset = pd.merge(dataset_reviews, dataset_timestamps, on='recommendationid', how='inner')

    # Converte il timestamp in formato leggibile
    merged_dataset['timestamp'] = pd.to_datetime(merged_dataset['timestamp_created'], unit='s')

    # Salva il dataset unito
    merged_dataset.to_csv(output_file, index=False)

    print(f"Dataset unito salvato in: {output_file}")
    return merged_dataset



def configure_paths(
    sentiment_results='../data/Output_SentimentAnalysis.csv',
    topic_results='../data/Topic_reviews.csv',
    timestamps_file='../data/binding_of_isaac_reviews.csv',
    sentiment_output_dir='../plots/sentiment_emotion_plots',
    topic_output_dir='../plots/topic_modelling_plots',
    merged_output_file='merged_reviews_with_timestamps.csv'
):
    """
    Configura i percorsi dei file e delle directory di output.

    Returns:
        dict: Un dizionario con i percorsi configurati.
    """
    os.makedirs(sentiment_output_dir, exist_ok=True)
    os.makedirs(topic_output_dir, exist_ok=True)

    return {
        "sentiment_results": sentiment_results,
        "topic_results": topic_results,
        "timestamps_file": timestamps_file,
        "sentiment_output_dir": sentiment_output_dir,
        "topic_output_dir": topic_output_dir,
        "merged_output_file": merged_output_file,
    }

def analyze_sentiment_and_emotions(df, output_dir):
    """
    Esegue le analisi relative al sentiment e alle emozioni.

    Args:
        df (pd.DataFrame): DataFrame contenente i dati elaborati.
        output_dir (str): Directory dove salvare i grafici.
    """
    plot_sentiment_trend_over_time(df, output_dir)
    plot_positive_negative_ratio(df, output_dir)
    plot_emotion_distribution_by_rating(df, output_dir)
    plot_sentiment_distribution(df, output_dir)
    plot_emotion_distribution(df, output_dir)
    plot_sentiment_scores(df, output_dir)
    plot_emotion_scores(df, output_dir)
    plot_emotion_by_sentiment(df, output_dir)
    plot_emotion_sentiment_heatmap(df, output_dir)
    plot_emotion_scores_by_sentiment(df, output_dir)
    plot_sentiment_vs_emotion_scores(df, output_dir)
    plot_average_emotion_scores_by_sentiment(df, output_dir)
    emotion_pie_chart(df, output_dir)


def analyze_topics(df, output_dir, n_top_words=10):
    """
    Esegue le analisi relative alla modellazione degli argomenti.

    Args:
        df (pd.DataFrame): DataFrame contenente i risultati della modellazione degli argomenti.
        output_dir (str): Directory dove salvare i grafici.
        n_top_words (int): Numero di parole principali per ogni argomento.
    """
    unique_topics = df['topic_number'].unique()
    for topic in unique_topics:
        plot_top_words_per_topic(df, output_dir=output_dir, topic_number=topic, n=n_top_words)
        generate_wordcloud(df, output_dir=output_dir, topic_number=topic)

    plot_topic_word_heatmap(df, num_words=n_top_words, output_dir=output_dir)
    plot_topic_distribution(df, output_dir=output_dir)

    for topic in unique_topics:
        plot_word_weights_scatter(df, output_dir=output_dir, topic_number=topic)
        plot_cumulative_weight_distribution(df, output_dir=output_dir, topic_number=topic)

    plot_parallel_coordinates(df, output_dir=output_dir, num_words=n_top_words)

def run_analysis(config, n_top_words=10):
    """
    Esegue l'intera pipeline di analisi.

    Args:
        config (dict): Configurazione dei percorsi dei file e delle directory.
        n_top_words (int): Numero di parole principali per ogni argomento (default: 10).
    """
    # Unisci i dataset e carica i dati elaborati
    merged_dataset = merge_datasets(
        reviews_file=config['sentiment_results'],
        timestamps_file=config['timestamps_file'],
        output_file=config['merged_output_file']
    )
    sentiment_df = load_and_process_data(merged_dataset)

    # Analisi del sentiment ed emozionioutpud_dir
    analyze_sentiment_and_emotions(sentiment_df, config['sentiment_output_dir'])

    # Caricamento dei dati della modellazione degli argomenti
    topic_df = pd.read_csv(config['topic_results'])

    # Analisi degli argomenti
    analyze_topics(topic_df, config['topic_output_dir'], n_top_words=n_top_words)

if __name__ == '__main__':
    # Configurazione dei percorsi
    config = configure_paths(
        sentiment_results='../data/Output_SentimentAnalysis.csv',
        topic_results='../data/Topic_reviews.csv',
        timestamps_file='../data/binding_of_isaac_reviews.csv',
        sentiment_output_dir='../plots/sentiment_emotion_plots',
        topic_output_dir='../plots/topic_modelling_plots',
        merged_output_file='merged_reviews_with_timestamps.csv'
    )

    # Esecuzione dell'analisi
    run_analysis(config, n_top_words=10)








