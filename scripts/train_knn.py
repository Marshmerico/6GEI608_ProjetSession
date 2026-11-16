import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]

K_NEIGHBORS = 20

GENRE_WEIGHT = 0.5 

VARIANCE_THRESHOLD = 0.80   

def feature_selection(input_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates(["track_name", "artists"]).reset_index(drop=True)

    correlation = df[FEATURES].corr()

    plt.figure(figsize=(8, 7))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
    plt.yticks(range(len(FEATURES)), FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            plt.text(j, i, f"{correlation.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Correlation matrix of the features")
    plt.tight_layout()
    plt.savefig("figure/correlation_features")
    plt.show()
    

    scaler = StandardScaler()
    standard = scaler.fit_transform(df[FEATURES])

    pca = PCA().fit(standard)

    variance = pca.explained_variance_ratio_
    
    total = 0
    ncomponents = 0;
    cumulative = []
    names = []

    for i in range(len(variance)):
        total += variance[i] 
        if not total >= VARIANCE_THRESHOLD:
            ncomponents += 1
            
        cumulative.append(total)
        names.append(f"PC{i + 1}")    

    colors = ["darkorange" if i < ncomponents else "lightgray" for i in range(len(variance))]
    plt.figure(figsize=(8, 5))
    plt.bar(names, variance, color=colors, label="variance of the component")
    plt.plot(names, cumulative, color="black", marker="o", label="cumulative variance")
    plt.axhline(VARIANCE_THRESHOLD, color="gray", linestyle="--", label=f"threshold ({VARIANCE_THRESHOLD:.0%})")
    plt.ylim(0, 1.05)
    plt.ylabel("explained variance ratio")
    plt.title(f"PCA: {ncomponents} components kept (gray = dropped)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure/PC_selection")
    plt.show()

    return ncomponents, PCA(n_components=ncomponents).fit(standard)


def recommendation(input_path, input_track, input_artist, input_genre, input_features, k_neighbors):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates(["track_name", "artists"]).reset_index(drop=True)

    scaler = StandardScaler()
    standard = scaler.fit_transform(df[FEATURES])

    #encode genre
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    standard = np.hstack([standard, GENRE_WEIGHT * encoder.fit_transform(df[["track_genre"]])])

    #search criteria
    #check if song in list

    match = df["track_name"].str.lower() == input_track.lower()
    match &= df["artists"].str.contains(input_artist, case=False, regex=False)

    if match.any():
        query_index = match.idxmax()
        query = standard[[query_index]]
        query_label = f"{df.at[query_index, 'track_name']} – {df.at[query_index, 'artists']}"

    else:
        query_index = None
        query = scaler.transform(pd.DataFrame([input_features])[FEATURES])
        genre = encoder.transform(pd.DataFrame({"track_genre": [input_genre]}))
        query = np.hstack([query, GENRE_WEIGHT * genre])
        query_label = f"input_track_name - input_artist"

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

    INPUT_TRACK = "The adults are talking2"
    INPUT_ARTIST = "The Strokes"  
    INPUT_GENRE = "Rock & Metal"   

    INPUT_FEATURES = {
        "danceability": 0.596, "energy": 0.749, "loudness": -5.67, "speechiness": 0.0475,
        "acousticness": 0.0113, "instrumentalness": 0.106, "liveness": 0.314,
        "valence": 0.649, "tempo": 164.959,
    }

    input_path = "data/playlist_1_full_regrouped.csv"

    ncomponents, pca = feature_selection(input_path)

    results,song_label, kneighbors = recommendation("data/playlist_1_full_regrouped.csv", INPUT_TRACK, INPUT_ARTIST, INPUT_GENRE, INPUT_FEATURES, K_NEIGHBORS)

    graphic_results(results,song_label, kneighbors)