# db_builder.py — пересборка instagram_data.db из data/instagram_mvd_dataset.json
import json
import sqlite3
from pathlib import Path
import pandas as pd

from analyzer import (
    analyze_sentiment,
    classify_theme,
    calculate_priority_score,
)

DATA_JSON = Path("../data/instagram_mvd_dataset.json")
DB_OUT = Path("instagram_data.db")


def main():
    print(f"Читаю {DATA_JSON}...")
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    posts = pd.DataFrame(data["posts"])
    comms = pd.DataFrame(data["comments"])
    print(f"  Постов: {len(posts)}, комментариев: {len(comms)}")

    # ---------- Справочник постов ----------
    post_id2url = dict(zip(posts["id"], posts.get("post_url", posts["id"])))
    post_id2text = dict(zip(posts["id"], posts["text"]))
    post_id2likes = dict(zip(posts["id"], posts.get("likesCount", [0] * len(posts))))

    # ---------- Комментарии ----------
    comms["post_url"] = comms["post_id"].map(post_id2url).fillna("")
    comms["post_text"] = comms["post_id"].map(post_id2text).fillna("")
    comms["likes_count"] = comms["post_id"].map(post_id2likes).fillna(0).astype(int)

    comms = comms.rename(columns={
        "id": "comment_id",
        "text": "comment_text",
        "ownerUsername": "comment_author",
        "timestamp": "comment_date",
    })

    # ---------- Тональность ----------
    print("Размечаю тональность через RuBERT...")
    comms["sentiment"] = analyze_sentiment(comms["comment_text"].tolist())

    # ---------- Темы ----------
    print("Классифицирую темы...")
    comms["theme"] = comms["comment_text"].apply(classify_theme)

    # ---------- Приоритет ----------
    print("Считаю priority...")
    comms["priority"] = comms.apply(calculate_priority_score, axis=1)

    # ---------- Финальная таблица ----------
    final = comms[[
        "comment_id", "comment_text", "comment_author", "comment_date",
        "post_url", "post_text", "likes_count",
        "sentiment", "theme", "priority",
    ]].copy()

    final = final.dropna(subset=["comment_text"])
    final = final[final["comment_text"].astype(str).str.strip().str.len() > 0]

    print(f"\nГотово: {len(final)} комментариев")
    print("\nРаспределение тональности:")
    print(final["sentiment"].value_counts())
    print("\nРаспределение тем:")
    print(final["theme"].value_counts())
    print(f"\nPriority min/max/mean: {final['priority'].min()}/"
          f"{final['priority'].max()}/{final['priority'].mean():.2f}")

    # ---------- Сохранение в SQLite ----------
    if DB_OUT.exists():
        DB_OUT.unlink()

    conn = sqlite3.connect(DB_OUT)
    final.to_sql("comments", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\n✅ БД создана: {DB_OUT.absolute()} ({DB_OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()