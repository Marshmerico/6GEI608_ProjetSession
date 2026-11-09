import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]

K_NEIGHBORS = 20

GENRE_WEIGHT = 0.5 

def recommendation(input_path, input_track, input_artist, input_genre, input_features, k_neighbors):
    df = pd.read_csv("data/playlist_1_full_regrouped.csv")
    df = df.drop_duplicates(["track_name", "artists"]).reset_index(drop=True)

    scaler = StandardScaler()
    standard = scaler.fit_transform(df[FEATURES])

    #encode genre
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    standard = np.hstack([standard, GENRE_WEIGHT * encoder.fit_transform(df[["track_genre"]])])

    #search criteria
    #check if song in list
    match = df["track_name"].str.lower() == INPUT_TRACK.lower()
    match &= df["artists"].str.contains(INPUT_ARTIST, case=False, regex=False)
    query_index = match.idxmax()

    query = standard[[query_index]]
    query_label = f"{df.at[query_index, 'track_name']} – {df.at[query_index, 'artists']}"

    #kneighbors
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine").fit(standard)
    distances, indices = nn.kneighbors(query)
    
    results = df.loc[indices[0], ["track_name", "artists", "track_genre"]].copy()
    results["similarity"] = 1 - distances[0]

    if query_index is not None:
        results = results.drop(index=query_index, errors="ignore")  # dont return same song
    results = results.head(k_neighbors).reset_index(drop=True)
    results.index += 1
    
    print(f"Songs most similar to: {query_label}\n")
    print(results.to_string(float_format="{:.3f}".format))


    return results, query_label, k_neighbors


    
def graphic_results(results,song_label, kneighbors):
    labels = results["track_name"] + " – " + results["artists"]
    plt.figure(figsize=(9, 0.45 * len(results) + 1.5))
    plt.barh(labels[::-1], results["similarity"][::-1], color="darkorange")
    plt.xlabel("cosine similarity")
    plt.xlim(results["similarity"].min() - 0.02, 1)
    plt.title(f"{kneighbors} nearest neighbors to {song_label}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    INPUT_TRACK = "Comedy"
    INPUT_ARTIST = "Gen Hoshino"  
    INPUT_GENRE = "Pop"   

    INPUT_FEATURES = {
        "danceability": 0.70, "energy": 0.80, "loudness": -5.0, "speechiness": 0.05,
        "acousticness": 0.10, "instrumentalness": 0.0, "liveness": 0.15,
        "valence": 0.60, "tempo": 120.0,
    }


    results,song_label, kneighbors = recommendation("data/playlist_1_full_regrouped.csv", INPUT_TRACK, INPUT_ARTIST, INPUT_GENRE, INPUT_FEATURES, K_NEIGHBORS)

    graphic_results(results,song_label, kneighbors)