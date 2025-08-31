import os
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPS
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ============================
# BENUTZERDEFINIERTE PARAMETER
# ============================

# Pfad zur Eingabedatei (Transkript oder Untertitel)
INPUT_FILE = "/Users/rsmirnov/Desktop/Heimatverteidiger/Battlefield_1_Avanti_Savoia_Text.txt"

# Automatische Erstellung des Ergebnisordners auf dem Desktop
OUTPUT_DIR = os.path.expanduser("~/Desktop/outputs_dr")

# Parameter für die Analyse
TOP_N = 50                   # Anzahl der häufigsten Wörter für Frequenzanalyse
N_TOPICS = 8                 # Anzahl der Themen für LDA
CHUNK_WORDS = 800            # Textsegmentierung für Topic Modeling
COOCCURRENCE_WINDOW = 2      # Fenstergröße für Kookkurrenzanalyse

# Sicherstellen, dass der Ausgabepfad existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# HILFSFUNKTIONEN
# ============================

def load_spacy():
    """Lädt das englische SpaCy-Modell"""
    try:
        return spacy.load("en_core_web_sm", disable=["tagger","parser","ner","attribute_ruler","lemmatizer"])
    except:
        return spacy.load("en_core_web_sm")

def normalize_whitespace(text: str) -> str:
    """Bereinigt den Text von überflüssigen Leerzeichen und Zeilenumbrüchen"""
    text = text.replace("\ufeff", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_text_basic(text: str) -> str:
    """Bereinigt Sonderzeichen und normalisiert Satzzeichen"""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\nA-Za-z0-9 ,\.\!\?\:\;\-\(\)\'\"]+", " ", text)
    return normalize_whitespace(text)

def sentences_spacy(text: str, nlp):
    """Segmentiert den Text in Sätze"""
    # Wenn das Modell keinen Sentencizer hat, fügen wir ihn hinzu
    if "sentencizer" not in nlp.pipe_names:
        try:
            nlp.add_pipe("sentencizer")
        except Exception as e:
            print("Konnte Sentencizer nicht hinzufügen:", e)
            return [text]  # Fallback: Rückgabe des gesamten Texts als ein Satz
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def tokenize_lemmatize(text: str, nlp):
    """Tokenisiert und lemmatisiert den Text"""
    if "lemmatizer" not in nlp.pipe_names:
        try:
            nlp.add_pipe("lemmatizer", config={"mode": "rule"}, last=True)
        except:
            pass
    doc = nlp(text)
    return [t.lemma_.lower().strip() if t.lemma_ else t.text.lower().strip() 
            for t in doc if not t.is_space and not t.is_punct]

def drop_stopwords(tokens, extra_stops=None, min_len=3):
    """Entfernt Stoppwörter und sehr kurze Tokens"""
    stops = set(SPACY_STOPS)
    if extra_stops:
        stops |= set(extra_stops)
    return [t for t in tokens if t not in stops and len(t) >= min_len and not t.isdigit()]

def chunk_by_words(words, chunk_size=1000):
    """Teilt Text in Segmente für Topic Modeling"""
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i+chunk_size]
        if len(chunk) >= max(50, chunk_size//4):
            chunks.append(" ".join(chunk))
    return chunks or [" ".join(words)]

def build_cooccurrence(sentences, top_terms, window=2):
    """Erstellt ein Kookkurrenznetzwerk der häufigsten Wörter"""
    top_set = set(top_terms)
    edges = Counter()
    for sent in sentences:
        words = [w for w in re.findall(r"[A-Za-z']+", sent.lower()) if len(w) >= 3]
        for i, w in enumerate(words):
            if w not in top_set:
                continue
            for j in range(i+1, min(i+1+window, len(words))):
                u, v = w, words[j]
                if u in top_set and v in top_set and u != v:
                    edge = tuple(sorted((u, v)))
                    edges[edge] += 1
    G = nx.Graph()
    for (u, v), w in edges.items():
        G.add_edge(u, v, weight=int(w))
    return G, edges

def lda_topics(docs, n_topics=8, max_df=0.9, min_df=2, max_features=5000, n_top_words=12):
    """Führt LDA-Topic-Modeling durch"""
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features,
                                 stop_words="english", ngram_range=(1,2))
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method="batch", random_state=42)
    lda.fit(X)
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = []
    for k, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[::-1][:n_top_words]
        topics.append({
            "topic_id": k,
            "terms": list(feature_names[top_idx]),
            "weights": list(map(float, topic[top_idx]))
        })
    doc_topic = lda.transform(X)
    return topics, doc_topic, vectorizer

# ============================
# HAUPTPROZESS
# ============================

print("Lese die Datei ein...")
raw = open(INPUT_FILE, "r", encoding="utf-8", errors="ignore").read()
cleaned = clean_text_basic(raw)
nlp = load_spacy()

# Tokenisierung und Satztrennung
sents = sentences_spacy(cleaned, nlp)
tokens = tokenize_lemmatize(cleaned, nlp)
tokens_clean = drop_stopwords(tokens)

# Wortfrequenzanalyse
freqs = Counter(tokens_clean)
df_top = pd.DataFrame(freqs.most_common(TOP_N), columns=["term", "count"])
df_top.to_csv(os.path.join(OUTPUT_DIR, "top_words.csv"), index=False)

# ====== VISUALISIERUNG 1: HÄUFIGSTE WÖRTER ======
plt.figure(figsize=(12,6))
plt.barh(df_top["term"], df_top["count"], color="skyblue")
plt.title("Top-{} häufigste Wörter".format(TOP_N))
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_words_plot.png"))
plt.show()

# ====== VISUALISIERUNG 2: SENTIMENT-ANALYSE ======
sia = SentimentIntensityAnalyzer()
sentiments = [sia.polarity_scores(s)["compound"] for s in sents]
plt.figure(figsize=(14,5))
plt.plot(sentiments, color="purple")
plt.axhline(0, color="black", linestyle="--")
plt.title("Sentiment-Analyse pro Satz")
plt.xlabel("Satzindex")
plt.ylabel("Compound Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sentiment_plot.png"))
plt.show()

# ====== VISUALISIERUNG 3: KOOCCURRENZ-NETZWERK ======
top_terms = [t for t, _ in freqs.most_common(TOP_N)]
G, edges = build_cooccurrence(sents, top_terms, window=COOCCURRENCE_WINDOW)
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)
nx.draw_networkx(G, pos, node_size=300, font_size=9, node_color="lightgreen")
plt.title("Kookkurrenznetzwerk der Top-Wörter")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cooccurrence_network.png"))
plt.show()

# ====== VISUALISIERUNG 4: TOPIC MODELING (LDA) ======
docs = chunk_by_words(tokens_clean, chunk_size=CHUNK_WORDS)
topics, doc_topic, vectorizer = lda_topics(docs, n_topics=N_TOPICS)
df_topics = pd.DataFrame([
    {"topic_id": t["topic_id"], "terms": ", ".join(t["terms"])} for t in topics
])
df_topics.to_csv(os.path.join(OUTPUT_DIR, "topics.csv"), index=False)

plt.figure(figsize=(12,6))
for i, topic in enumerate(topics):
    plt.bar([f"Topic {topic['topic_id']}"], [sum(topic["weights"])], label=f"Topic {topic['topic_id']}")
plt.title("Themenverteilung (LDA)")
plt.ylabel("Gesamtgewicht der Terme")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lda_topics_plot.png"))
plt.show()

print(f"Analyse abgeschlossen! Ergebnisse gespeichert in: {OUTPUT_DIR}")
