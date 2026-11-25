import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score


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

    pca = PCA(0.8).fit(standard)

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

def confusion_matrix(input_path, ncomponents):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates(["track_name", "artists"]).reset_index(drop=True)

    n = df["track_genre"].value_counts().min()
    df = df.groupby("track_genre").sample(n=n, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["track_genre"], test_size=0.2, stratify=df["track_genre"], random_state=0)

    
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=ncomponents).fit(scaler.transform(X_train))

    X_train = pca.transform(scaler.transform(X_train))
    X_test = pca.transform(scaler.transform(X_test))

    knn = KNeighborsClassifier(n_neighbors=50, metric="cosine").fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true", values_format=".2f",
                                            cmap="Oranges", xticks_rotation=45)
    plt.title(f"Confusion matrix (KNN, k=50, {ncomponents} principal components)")
    plt.gcf().set_size_inches(10, 9)
    plt.tight_layout()
    plt.savefig("figure/confusion_genre")
    plt.show()

def train():
    df = pd.read_csv(input_path)
    df = df.drop_duplicates(["track_name", "artists"]).reset_index(drop=True)

    scaler = StandardScaler()
    standard = scaler.fit_transform(df[FEATURES])

    #encode genre
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    standard = np.hstack([standard, GENRE_WEIGHT * encoder.fit_transform(df[["track_genre"]])])

    
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="cosine")
    nn.fit(standard)

    joblib.dump(nn,"knn_model.pkl")
    


def recommendation(input_path, input_track, input_artist, input_genre, input_features, k_neighbors):
    
    nn = joblib.load("knn_model.pkl")

    df = pd.read_csv(input_path)
    df = df.drop_duplicates(["track_name", "artists"]).reset_index(drop=True)

    scaler = StandardScaler()
    standard = scaler.fit_transform(df[FEATURES])
    
    #encode genre
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    standard = np.hstack([standard, GENRE_WEIGHT * encoder.fit_transform(df[["track_genre"]])])
    
   

    query_features = pd.DataFrame([input_features])[FEATURES]      
    query_genre = pd.DataFrame({"track_genre": [input_genre]})    
    query = np.hstack([
        scaler.transform(query_features),
        GENRE_WEIGHT * encoder.transform(query_genre),
    ])

    distances, indices = nn.kneighbors(query)
    
    results = df.loc[indices[0], ["track_name", "artists", "track_genre"]].copy()
    results["similarity"] = 1 - distances[0]

    
    results = results.head(k_neighbors).reset_index(drop=True)
    results.index += 1
    
    print(f"Songs most similar to: {input_track}\n")
    print(results.to_string(float_format="{:.3f}".format))


    return results, input_track, k_neighbors


    
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

    input_path = "data/playlist_1_full_regrouped_v2.csv"

    ncomponents, pca = feature_selection(input_path)

    confusion_matrix(input_path, ncomponents)
    
    train()

    results,song_label, kneighbors = recommendation( input_path, INPUT_TRACK, INPUT_ARTIST, INPUT_GENRE, INPUT_FEATURES, K_NEIGHBORS)

    graphic_results(results,song_label, kneighbors)