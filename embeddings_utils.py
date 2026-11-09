import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("cointegrated/rubert-tiny2")
    return _embedder

def compute_embeddings(texts, batch_size=64):
    model = get_embedder()
    return model.encode(
        [str(t) for t in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

def cluster_comments(df, n_clusters=5, only_negative=True):
    if only_negative:
        mask = df["sentiment"] == "Негатив"
        work = df[mask].copy()
    else:
        work = df.copy()

    if len(work) < n_clusters * 2:
        return None, {}

    emb = compute_embeddings(work["comment_text"].tolist())

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    work["cluster"] = km.fit_predict(emb)

    descriptions = {}
    for c in range(n_clusters):
        cluster_texts = work[work["cluster"] == c]["comment_text"].tolist()
        if len(cluster_texts) == 0:
            continue

        words = []
        for t in cluster_texts:
            words += re.findall(r"[а-яё]{4,}", str(t).lower())
        top_words = [w for w, _ in Counter(words).most_common(10)]

        examples = sorted(cluster_texts, key=len, reverse=True)[:3]

        descriptions[c] = {
            "size": len(cluster_texts),
            "top_words": top_words,
            "examples": examples,
        }

    return work, descriptions

def compute_post_comment_similarity(df):
    if "post_text" not in df.columns or df["post_text"].isna().all():
        return None

    df = df.copy()
    unique_posts = df["post_text"].dropna().unique()

    post_emb_map = {}
    for pt in unique_posts:
        post_emb_map[pt] = compute_embeddings([pt])[0]

    comment_embs = compute_embeddings(df["comment_text"].tolist())

    sims = []
    for i, row in enumerate(df.itertuples()):
        pt = getattr(row, "post_text", None)
        if pt in post_emb_map and pd.notna(pt):
            sim = float(np.dot(comment_embs[i], post_emb_map[pt]))
        else:
            sim = 0.0
        sims.append(sim)

    df["similarity"] = sims
    return df

def compute_trend(df):
    if "comment_date" not in df.columns:
        return None, 0, "flat"

    work = df.copy()
    work["comment_date"] = pd.to_datetime(work["comment_date"], errors="coerce")
    work = work.dropna(subset=["comment_date"])
    if len(work) < 3:
        return None, 0, "flat"

    work["date"] = work["comment_date"].dt.date
    daily = work.groupby("date").agg(
        total=("comment_text", "count"),
        negative=("sentiment", lambda x: (x == "Негатив").sum()),
    ).reset_index()
    daily["neg_share"] = daily["negative"] / daily["total"] * 100

    daily = daily.sort_values("date").reset_index(drop=True)
    x = np.arange(len(daily))
    y = daily["neg_share"].values
    if len(x) >= 2:
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    if slope > 1.0:
        direction = "up"
    elif slope < -1.0:
        direction = "down"
    else:
        direction = "flat"

    return daily, slope, direction

def compare_with_norm(filtered_df, full_df):
    if len(filtered_df) == 0 or len(full_df) == 0:
        return None

    filtered_neg = (filtered_df["sentiment"] == "Негатив").mean() * 100
    full_neg = (full_df["sentiment"] == "Негатив").mean() * 100

    diff = filtered_neg - full_neg
    ratio = filtered_neg / full_neg if full_neg > 0 else 0

    if diff > 10:
        verdict = "problem"
        message = "⚠️ Выборка содержит значительно больше негатива, чем в среднем по датасету"
    elif diff > 3:
        verdict = "slightly_problem"
        message = "⚠️ Негатива немного больше нормы"
    elif diff < -10:
        verdict = "positive"
        message = "✅ Выборка содержит значительно меньше негатива, чем в среднем"
    elif diff < -3:
        verdict = "slightly_positive"
        message = "✅ Негатива немного меньше нормы"
    else:
        verdict = "neutral"
        message = "ℹ️ Уровень негатива близок к среднему по датасету"

    return {
        "filtered_neg_pct": filtered_neg,
        "full_neg_pct": full_neg,
        "diff": diff,
        "ratio": ratio,
        "verdict": verdict,
        "message": message,
    }