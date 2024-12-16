import pandas as pd
import matplotlib.pyplot as plt
import json
from wordcloud import WordCloud
import seaborn as sns
from pandas.plotting import parallel_coordinates
import os
import ast

from scripts.utility import game_selection


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


def plot_star_rating_trend_over_time(df, output_dir):
    """
    Plot the average star rating trend over time and save it locally with enhanced readability.

    Args:
    - df (pd.DataFrame): DataFrame containing 'timestamp' and 'sentiment_label'.
    - output_dir (str): Directory where the plot will be saved.

    Returns:
    - None: Saves the trend plot to the specified directory.
    """
    # Check if required columns are in the DataFrame
    if 'timestamp' not in df or 'sentiment_label' not in df:
        print("Error: 'timestamp' and/or 'sentiment_label' column(s) are missing in the dataset.")
        return

    # Convert timestamp to datetime format
    try:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date'] = df['datetime'].dt.date
    except Exception as e:
        print(f"Error in converting 'timestamp' to datetime: {e}")
        return

    # Extract numeric star ratings from 'sentiment_label'
    try:
        df['star_rating'] = pd.to_numeric(df['sentiment_label'].str.extract(r'(\d+)')[0], errors='coerce')
    except Exception as e:
        print(f"Error in extracting star ratings: {e}")
        return

    # Drop rows with missing or invalid star ratings
    df = df.dropna(subset=['star_rating'])

    # Calculate the average star rating per day
    daily_star_rating = df.groupby('date')['star_rating'].mean()

    # Plot the trend with enhanced aesthetics
    plt.figure(figsize=(12, 6))
    plt.plot(daily_star_rating.index, daily_star_rating.values, marker='o', linestyle='-', linewidth=2, alpha=0.8)
    plt.title('Star Rating Trend Over Time', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Star Rating', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'star_rating_trend_over_time.png')

    # Save the plot
    plt.savefig(output_path, dpi=300)
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




def emotion_accuracy(df, output_dir):
    """
    Create a bar chart for the model's accuracy by sentiment and save it locally.
    Args:
    - df (pd.DataFrame): DataFrame containing 'sentiment_label' and 'accuracy' columns.
    - output_dir (str): Directory where the plot will be saved.

    Returns:
    - None: Saves the bar chart to the specified directory.
    """

    # Ensure the column names align with the intention
    avg_accuracy = df.groupby('emotion_label')['emotion_score'].mean()

    # Create the bar chart
    plt.figure(figsize=(8, 8))
    avg_accuracy.plot(kind='bar')
    plt.title('Accuracy of the model')
    plt.xlabel('Prediciton on Emotion')
    plt.ylabel('Accuracy Mean')
    plt.xticks(rotation=45)

    # Save the plot
    output_path = os.path.join(output_dir, 'accuracy_of_emotion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def sentiment_accuracy(df, output_dir):
    """
    Create a bar chart for the model's accuracy by sentiment and save it locally.
    Args:
    - df (pd.DataFrame): DataFrame containing 'sentiment_label' and 'accuracy' columns.
    - output_dir (str): Directory where the plot will be saved.

    Returns:
    - None: Saves the bar chart to the specified directory.
    """

    # Ensure the column names align with the intention
    avg_accuracy = df.groupby('sentiment_label')['sentiment_score'].mean()

    # Create the bar chart
    plt.figure(figsize=(8, 8))
    avg_accuracy.plot(kind='bar')
    plt.title('Accuracy of the model')
    plt.xlabel('Prediciton on Star Rating')
    plt.ylabel('Accuracy Mean')
    plt.xticks(rotation=45)

    # Save the plot
    output_path = os.path.join(output_dir, 'accuracy_of_sentiment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def sentiment_distribution_piechart(df, output_dir):
    """
    Create a pie chart showing the distribution of tuples by sentiment_label and save it locally.
    Args:
        df (pd.DataFrame): DataFrame containing 'sentiment_label' column.
        output_dir (str): Directory where the plot will be saved.
    Returns:
    None: Saves the pie chart to the specified directory."""  # Define the order of the stars


    star_order = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

    # Count the number of tuples per sentiment label and sort based on the star order
    counts = df['sentiment_label'].value_counts()
    counts = counts.reindex(star_order, fill_value=0)  # Ensure all stars are included and ordered

    # Define a color map to match the order
    colors = plt.cm.viridis([0.1, 0.3, 0.5, 0.7, 0.9])  # Adjust color shades for a consistent gradient

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        legend=True
    )
    plt.title('Distribution of Rating Stars')
    plt.ylabel('')  # Remove y-axis label for a cleaner pie chart

    # Save the chart
    output_path = os.path.join(output_dir, 'sentiment_label_distribution_pie_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sentiment_distribution(df, output_dir):
    """
    Create a bar chart for sentiment distribution and save it locally.
    """
    plt.figure(figsize=(10, 9))
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
    plt.figure(figsize=(10, 9))
    df['emotion_label'].value_counts().plot(kind='bar')
    plt.title('Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
    plt.close()




def plot_emotion_by_sentiment(df, output_dir):
    """
    Create a grouped bar chart for emotion by sentiment and save it locally.
    """
    plt.figure(figsize=(10, 11))
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
    topic_weights.plot(kind='bar', alpha=0.8)
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
    plt.bar(topic_df['word'], topic_df['weight'])
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

    plot_positive_negative_ratio(df, output_dir)
    plot_emotion_distribution_by_rating(df, output_dir)
    plot_sentiment_distribution(df, output_dir)
    plot_emotion_distribution(df, output_dir)
    plot_emotion_by_sentiment(df, output_dir)
    plot_emotion_sentiment_heatmap(df, output_dir)
    sentiment_distribution_piechart(df, output_dir)
    emotion_accuracy(df, output_dir)
    sentiment_accuracy(df, output_dir)
    emotion_pie_chart(df, output_dir)
    plot_star_rating_trend_over_time(df,output_dir)


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
    foldername = game_selection()
    # Configurazione dei percorsi
    config = configure_paths(
        sentiment_results=f'../data/{foldername}/Output_SentimentAnalysis.csv',
        topic_results=f'../data/{foldername}/Topic_modeling.csv',
        timestamps_file=f'../data/{foldername}/Reviews.csv',
        sentiment_output_dir=f'../data/{foldername}/plots/sentiment_emotion_plots',
        topic_output_dir=f'../data/{foldername}/plots/topic_modelling_plots',
        merged_output_file=f'../data/{foldername}/merged_reviews_with_timestamps.csv'
    )

    # Esecuzione dell'analisi
    run_analysis(config, n_top_words=10)








