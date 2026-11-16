"""
Collecte des extraits audio de 30 s sur Deezer pour une liste de chansons.

Entrée  : un CSV de chansons (par défaut au format du jeu Kaggle « Spotify Tracks Dataset » :
          colonnes track_id, artists, track_name, track_genre, duration_ms).
Sortie  : Data/audio_deezer/<genre>/<track_id>.mp3
          Data/collecte_log.csv  (une ligne par chanson : statut, correspondance trouvée, score)

Le script peut être arrêté et relancé : les chansons déjà traitées (présentes dans le log)
sont sautées. Les liens d'extraits Deezer expirent : chaque extrait est donc téléchargé
immédiatement après la recherche.

Exemples :
    python collecte_deezer.py --input Data/chansons.csv --limit 20      # essai sur 20 chansons
    python collecte_deezer.py --input Data/chansons.csv                 # collecte complète
    python collecte_deezer.py --input Data/chansons.csv --retry         # réessaie les échecs

Seule la bibliothèque standard de Python est utilisée.
"""

import argparse
import csv
import difflib
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

API_SEARCH = "https://api.deezer.com/search"
USER_AGENT = "projet-6GEI608-UQAC/1.0"
API_DELAY = 0.15          # secondes entre deux appels (Deezer limite à 50 requêtes / 5 s)
QUOTA_PAUSE = 6.0         # pause quand Deezer répond « Quota limit exceeded »
MIN_SCORE = 0.80          # similarité minimale du titre pour accepter une correspondance
MAX_DURATION_GAP = 15     # écart de durée maximal accepté (secondes), si la durée est connue

LOG_FIELDS = [
    "track_id", "artist", "title", "genre", "statut",
    "deezer_id", "deezer_artist", "deezer_title", "score", "fichier", "raison",
]
# Mots qui signalent une autre version que l'originale
VERSION_WORDS = ("live", "remix", "acoustic", "acoustique", "karaoke", "instrumental",
                 "cover", "version", "edit", "mix", "demo", "sped up", "slowed",
                 "acapella", "a cappella", "a capella", "recorded at", "session",
                 "nightcore", "8d", "reprise", "unplugged", "orchestral")
# Parenthèses sans importance pour la correspondance
NEUTRAL_WORDS = ("feat", "ft.", "with", "remaster", "explicit", "from", "original")


# --------------------------------------------------------------------------- texte

def normalize(text):
    """Minuscules, sans accents, sans contenu entre parenthèses/crochets ni ' - Remastered…'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)          # (feat. X), [Remastered]
    text = re.sub(r"\s-\s.*$", " ", text)                  # « Titre - Remastered 2011 »
    text = re.sub(r"\b(feat|ft|featuring)\b.*$", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def is_other_version(candidate_title, original_title):
    """Vrai si le titre Deezer annonce une autre version (live, remix…) absente du titre d'origine."""
    cand, orig = candidate_title.lower(), original_title.lower()
    return any(w in cand and w not in orig for w in VERSION_WORDS)


def extra_parentheses(candidate_title, original_title):
    """Mentions entre parenthèses présentes sur Deezer mais absentes du titre d'origine
    (ex. « (Recorded at Hoxa Studios) ») ; les mentions neutres (feat., remaster…) sont ignorées."""
    def parts(t):
        return {normalize_light(p) for p in re.findall(r"\((.*?)\)|\[(.*?)\]", t) for p in p if p}
    extra = parts(candidate_title) - parts(original_title)
    return [e for e in extra if not any(w in e for w in NEUTRAL_WORDS)]


def normalize_light(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", text).strip()


def first_artist(artists):
    # Le jeu Kaggle sépare les artistes par « ; »
    return artists.split(";")[0].strip()


# --------------------------------------------------------------------------- Deezer

def http_get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def deezer_search(query, limit=10, retries=3):
    url = f"{API_SEARCH}?{urllib.parse.urlencode({'q': query, 'limit': limit})}"
    for attempt in range(retries):
        time.sleep(API_DELAY)
        try:
            data = json.loads(http_get(url))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"    erreur réseau ({exc}), nouvel essai…")
            time.sleep(2 * (attempt + 1))
            continue
        if "error" in data:
            if data["error"].get("code") == 4:          # Quota limit exceeded
                time.sleep(QUOTA_PAUSE)
                continue
            raise RuntimeError(f"Deezer : {data['error']}")
        return data.get("data", [])
    raise RuntimeError("Deezer ne répond pas après plusieurs essais")


def best_match(artist, title, duration_s):
    """Cherche la chanson et retourne (candidat, score, raison).
    raison explique un échec : aucun résultat, artiste différent, durée, score faible…"""
    # Titre de recherche sans « - Remastered 2011 », « (feat. X) », etc.
    query_title = re.sub(r"\(.*?\)|\[.*?\]|\s-\s.*$", " ", title).strip() or title
    candidates = deezer_search(f'artist:"{artist}" track:"{query_title}"')
    if not candidates:                                   # recherche plus souple
        candidates = deezer_search(f"{artist} {query_title}")

    if not candidates:
        return None, 0.0, "aucun résultat Deezer"

    best, best_score = None, 0.0
    rejects = {"pas d'extrait": 0, "artiste différent": 0, "durée différente": 0}
    for c in candidates:
        c_artist = c.get("artist", {}).get("name", "")
        c_title = c.get("title", "")
        if not c.get("preview"):
            rejects["pas d'extrait"] += 1
            continue
        if similarity(artist, c_artist) < 0.85 and normalize(artist) not in normalize(c_artist):
            rejects["artiste différent"] += 1            # reprise, homonyme
            continue
        if duration_s and abs(c.get("duration", 0) - duration_s) > MAX_DURATION_GAP:
            rejects["durée différente"] += 1             # autre version (live, édition radio…)
            continue
        score = similarity(title, c.get("title_short") or c_title)
        if is_other_version(c_title, title):
            score -= 0.3                                 # live, remix, acapella…
        elif extra_parentheses(c_title, title):
            score -= 0.1                                 # préférer la version sans mention ajoutée
        if score > best_score:
            best, best_score = c, score

    if best is not None and best_score >= MIN_SCORE:
        return best, best_score, ""
    if best is not None:
        return None, best_score, f"score faible ({best['artist']['name']} - {best['title']})"
    reason = ", ".join(f"{k} x{v}" for k, v in rejects.items() if v)
    return None, 0.0, f"{len(candidates)} résultats rejetés : {reason}"


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(http_get(url, timeout=30))
    os.replace(tmp, path)                                # pas de fichier à moitié écrit


# --------------------------------------------------------------------------- log

def read_log(log_path):
    if not os.path.exists(log_path):
        return {}
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = {row["track_id"]: row for row in csv.DictReader(f)}
    for row in rows.values():                            # anciens logs sans la colonne raison
        row.setdefault("raison", "")
        row.pop(None, None)
    return rows


def write_log(log_path, rows):
    tmp = log_path + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    os.replace(tmp, log_path)


# --------------------------------------------------------------------------- principal

def main():
    p = argparse.ArgumentParser(description="Collecte des extraits Deezer de 30 s")
    p.add_argument("--input", default="Data/chansons.csv", help="CSV de la liste de chansons")
    p.add_argument("--out-dir", default="Data/audio_deezer")
    p.add_argument("--log", default="Data/collecte_log.csv")
    p.add_argument("--limit", type=int, default=None, help="ne traiter que N chansons (essai)")
    p.add_argument("--retry", action="store_true", help="réessayer les chansons en échec")
    p.add_argument("--col-id", default="track_id")
    p.add_argument("--col-artist", default="artists")
    p.add_argument("--col-title", default="track_name")
    p.add_argument("--col-genre", default="track_genre")
    p.add_argument("--col-duration", default="duration_ms",
                   help="durée en millisecondes ; laisser vide si absente")
    args = p.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        songs = list(csv.DictReader(f))
    if args.limit:
        songs = songs[:args.limit]

    log = read_log(args.log)
    counts = {"ok": 0, "introuvable": 0, "erreur": 0, "deja_fait": 0}
    print(f"{len(songs)} chansons à traiter ({len(log)} déjà dans le log)")

    for i, song in enumerate(songs, 1):
        track_id = song[args.col_id]
        previous = log.get(track_id)
        # « exclu » = rejeté à la main (mauvaise version, extrait corrompu) : jamais retenté
        if previous and (previous["statut"] in ("ok", "exclu") or not args.retry):
            counts["deja_fait"] += 1
            continue

        artist = first_artist(song[args.col_artist])
        title = song[args.col_title]
        genre = song[args.col_genre]
        duration_s = None
        if args.col_duration and song.get(args.col_duration):
            duration_s = float(song[args.col_duration]) / 1000

        row = {"track_id": track_id, "artist": artist, "title": title, "genre": genre,
               "statut": "", "deezer_id": "", "deezer_artist": "", "deezer_title": "",
               "score": "", "fichier": "", "raison": ""}
        try:
            match, score, reason = best_match(artist, title, duration_s)
            row["score"] = f"{score:.2f}"
            if match is None:
                row["statut"] = "introuvable"
                row["raison"] = reason
            else:
                path = os.path.join(args.out_dir, genre, f"{track_id}.mp3")
                download(match["preview"], path)         # tout de suite : le lien expire
                row.update(statut="ok", deezer_id=match["id"],
                           deezer_artist=match["artist"]["name"],
                           deezer_title=match["title"], fichier=path)
        except Exception as exc:                          # on note l'erreur et on continue
            row["statut"] = "erreur"
            row["raison"] = str(exc)[:200]

        counts[row["statut"]] += 1
        log[track_id] = row
        detail = (f"  ->  {row['deezer_artist']} - {row['deezer_title']} ({row['score']})"
                  if row["statut"] == "ok" else f"  ({row['raison']})")
        print(f"[{i}/{len(songs)}] {row['statut']:11s} {artist} - {title}{detail}")

        if i % 20 == 0:                                   # sauvegarde régulière du log
            write_log(args.log, log)

    write_log(args.log, log)
    done = counts["ok"] + counts["introuvable"] + counts["erreur"]
    print("\nTerminé :", ", ".join(f"{k} {v}" for k, v in counts.items()))
    if done:
        print(f"Taux de réussite sur cette exécution : {counts['ok'] / done:.0%}")
    print(f"Log : {args.log}")


if __name__ == "__main__":
    main()
