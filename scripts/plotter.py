
import pandas as pd
import matplotlib.pyplot as plt
import json
from wordcloud import WordCloud
import seaborn as sns
from pandas.plotting import parallel_coordinates



def emotion_pie_chart(df, path_to_save='../plots/emotion_plots/emotion_pie_chart.png'):

    # Convert 'sentiment' and 'emotion' columns from strings to dictionaries
    df['sentiment'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'", '"'))[0])
    df['emotion'] = df['emotion'].apply(lambda x: json.loads(x.replace("'", '"'))[0])

    # Extract sentiment and emotion details
    df['sentiment_score'] = df['sentiment'].apply(lambda x: x['score'])
    df['sentiment_label'] = df['sentiment'].apply(lambda x: x['label'])
    df['emotion_label'] = df['emotion'].apply(lambda x: x['label'])
    df['emotion_score'] = df['emotion'].apply(lambda x: x['score'])


    # Pie chart for emotion labels
    emotion_counts = df['emotion_label'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Emotion Distribution')
    plt.savefig(path_to_save)  # Save the plot






def plot_topic_word_heatmap(df, num_words=10, path_to_save='../plots/topic/topic_word_heatmap.png'):
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
    plt.savefig(path_to_save)



def plot_topic_word_heatmap(df, num_words=10, path_to_save='../plots/topic_modelling_plots/topic_word_heatmap.png'):
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
    plt.savefig(path_to_save)





def plot_word_weights_scatter(df, topic_number, path_to_save='../plots/topic_'):
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
    plt.savefig(f'{path_to_save}{topic_number}_scatter.png')


def plot_word_weights_scatter(df, topic_number, path_to_save='../plots/topic_modelling_plots/topic_'):
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
    plt.savefig(f'{path_to_save}_{topic_number}_scatter.png')



def plot_parallel_coordinates(df, num_words=5, path_to_save='../plots/topic_modelling_plots/topic_parallel_coordinates.png'):
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
    plt.savefig(path_to_save)


def plot_topic_distribution(df, path_to_save='../plots/topic_modelling_plots/topic_distribution.png'):
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
    plt.savefig(path_to_save)


def plot_cumulative_weight_distribution(df, topic_number, path_to_save='../plots/topic_modelling_plots/topic_'):
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
    plt.savefig(f'{path_to_save}{topic_number}_cumulative_distribution.png')



# Plot Top Words per Topic
def plot_top_words_per_topic(df, topic_number, n=10, path_to_save='../plots/topic_modelling_plots/topic_'):
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
    plt.savefig(f'{path_to_save}{topic_number}_top_words.png')  # Save plot as an image


# Generate Word Cloud for a Topic
def generate_wordcloud(df, topic_number,path_to_save='../plots/topic_modelling_plots/topic_'):
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
    plt.savefig(f'{path_to_save}_{topic_number}_wordcloud.png')  # Save word cloud as an image


SENTIMENT_ANALYSIS_RESULTS = '../data/Output_SentimentAnalysis.csv'
TOPIC_MODELING_RESULTS = '../data/Topic_reviews.csv'

if __name__ == '__main__':
    # Sentiment analysis charts
    df = pd.read_csv(SENTIMENT_ANALYSIS_RESULTS)
    emotion_pie_chart(df)
    df = pd.read_csv(TOPIC_MODELING_RESULTS)
    # Plotting for all topics
    unique_topics = df['topic_number'].unique()
    for topic in unique_topics:
        # Plot top words per topic
        plot_top_words_per_topic(df, topic_number=topic, n=10)
        # Generate and save word cloud per topic
        generate_wordcloud(df, topic_number=topic)

    plot_topic_word_heatmap(df, num_words=10)
    plot_topic_distribution(df)

    plot_word_weights_scatter(df, topic_number=0)
    plot_word_weights_scatter(df, topic_number=1)
    plot_word_weights_scatter(df, topic_number=2)
    plot_word_weights_scatter(df, topic_number=3)
    plot_word_weights_scatter(df, topic_number=4)
    plot_cumulative_weight_distribution(df, topic_number=0)
    plot_cumulative_weight_distribution(df, topic_number=1)
    plot_cumulative_weight_distribution(df, topic_number=2)
    plot_cumulative_weight_distribution(df, topic_number=3)
    plot_cumulative_weight_distribution(df, topic_number=4)
    plot_parallel_coordinates(df, num_words=5)




