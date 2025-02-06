
# Steam Review Analysis

This project involves scraping comments, performing sentiment and emotion analysis, topic modeling, and visualizing the results. Below are the steps to set up and run the project.

## Prerequisites

Before you start, make sure you have the following:

- Python 3.8 or later
- Pip (Python package manager)

## Steps

### 1. Clone the Project

First, clone the repository to your local machine:

```bash
git clone https://github.com/LobsterRavioli/steam-review-analysis.git
cd <project_directory>
```

### 2. Install Requirements

Install the necessary dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Run `driver.py` to Fetch Comments

Use `driver.py` to fetch comments from a specified source. For an example, choose "mini" as the data source.

```bash
python driver.py
```

This will scrape the comments and save them for further analysis.

### 4. Run `sentiment_emotion_analysis.py`

Once the comments are fetched, run the `sentiment_emotion_analysis.py` script to perform sentiment and emotion analysis on the comments.

```bash
python sentiment_emotion_analysis.py
```

This script will analyze the comments and output the sentiment (star-score from 1 to 5) and associated emotions.

### 5. Run `topic_modelling.py`

After performing sentiment and emotion analysis, run `topic_modelling.py` to extract the main topics from the comments.

```bash
python topic_modelling.py
```

This script will identify the most frequent topics from the fetched comments and display the results.

### 6. Run `plotter.py` for Visualization

Finally, run `plotter.py` to visualize the results of sentiment analysis, emotion analysis, and topic modeling.

```bash
python plotter.py
```

This will generate the relevant plots and charts to help interpret the results visually.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
