import requests
import time
import csv


def fetch_reviews_to_csv(app_id, output_file, max_reviews=None):
    """
    Fetch all reviews for a Steam app ID using the Steam Reviews API and save them in a CSV file.

    :param app_id: The app ID of the game (e.g., 250900 for The Binding of Isaac: Rebirth).
    :param output_file: Path to save the reviews (CSV format).
    :param max_reviews: Maximum number of reviews to fetch. If None, fetch all available.
    """
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
    cursor = "*"  # Initial cursor for pagination
    total_reviews_fetched = 0
    fields = [
        "recommendationid", "steamid", "num_games_owned", "num_reviews",
        "playtime_forever", "playtime_last_two_weeks", "playtime_at_review",
        "last_played", "language", "review", "timestamp_created", "timestamp_updated",
        "voted_up", "votes_up", "votes_funny", "weighted_vote_score",
        "comment_count", "steam_purchase", "received_for_free", "written_during_early_access"
    ]

    # Open CSV file and write header
    with (open(output_file, "w", newline="", encoding="utf-8") as file):
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        while True:
            params = {
                "json": 1,
                "filter": "recent",  # Fetch all reviews
                "language": "english",  # Language of reviews
                "cursor": cursor,  # Pagination cursor
                "num_per_page": 100,  # Maximum reviews per request
            }
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                break

            data = response.json()
            if not data.get("success"):
                print("Failed to fetch reviews. Exiting.")
                break

            reviews = data.get("reviews", [])
            if len(reviews)==0:
                return

            # Process each review and write to CSV
            for review in reviews:
                row = {
                    "recommendationid": review.get("recommendationid"),
                    "steamid": review["author"].get("steamid"),
                    "num_games_owned": review["author"].get("num_games_owned"),
                    "num_reviews": review["author"].get("num_reviews"),
                    "playtime_forever": review["author"].get("playtime_forever"),
                    "playtime_last_two_weeks": review["author"].get("playtime_last_two_weeks"),
                    "playtime_at_review": review["author"].get("playtime_at_review"),
                    "last_played": review["author"].get("last_played"),
                    "language": review.get("language"),
                    "review": review.get("review"),
                    "timestamp_created": review.get("timestamp_created"),
                    "timestamp_updated": review.get("timestamp_updated"),
                    "voted_up": review.get("voted_up"),
                    "votes_up": review.get("votes_up"),
                    "votes_funny": review.get("votes_funny"),
                    "weighted_vote_score": review.get("weighted_vote_score"),
                    "comment_count": review.get("comment_count"),
                    "steam_purchase": review.get("steam_purchase"),
                    "received_for_free": review.get("received_for_free"),
                    "written_during_early_access": review.get("written_during_early_access"),
                }


                writer.writerow(row)

            total_reviews_fetched += len(reviews)
            print(f"Fetched {len(reviews)} reviews (Total: {total_reviews_fetched})")


            # Update cursor for the next batch
            cursor = data.get("cursor")
            # Respect API rate limits
            time.sleep(1)

    print(f"Completed. Total reviews fetched: {total_reviews_fetched}")




def choose_version():
    """
    Prompts the user to select between a mini version (quick test) or a full version
    (complete dataset). Returns the selected version as a string ('mini' or 'full').
    """
    print("Please choose the version to run:")
    print("1. Run Mini Version (for a quick test with a small dataset [~3500 entries])")
    print("2. Run Full Version (for running with a complete dataset [~120000 entries])")
    version = ""
    while True:
        # Get user input and strip any surrounding whitespace
        choice = input("Enter '1' for Mini Version or '2' for Full Version: ").strip()
        # Validate the input and return the corresponding version
        if choice == '1':
            print(f"You have selected the \"mini\" version.")
            fetch_reviews_to_csv(app_id=1272320, output_file="../data/mini_reviews.csv", max_reviews=None)
            return
        elif choice == '2':
            print(f"You have selected the \"full\" version.")
            fetch_reviews_to_csv(app_id=250900, output_file="../data/binding_of_isaac_reviews.csv", max_reviews=None)
            return
        else:
            # Provide feedback on invalid input
            print("Invalid input. Please choose '1' for Mini Version or '2' for Full Version.")



choose_version()
