import json
import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def try_read_file(file_path):

    encodings = [
        "utf-8",
        "cp1251",
        "windows-1251",
        "latin-1"
    ]

    for enc in encodings:

        try:
            with open(
                file_path,
                "r",
                encoding=enc,
                errors="ignore"
            ) as f:

                content = f.read()

                return json.loads(content)

        except:
            continue

    return []


def load():

    rows = []

    for file in DATA_DIR.glob("*.json"):

        print(f"Loading: {file}")

        data = try_read_file(file)

        if isinstance(data, list):

            for item in data:

                rows.append({

                    "comment_id": item.get("id", ""),

                    "comment_text": item.get("text", ""),

                    "comment_author": item.get(
                        "ownerUsername",
                        item.get("author", "unknown")
                    ),

                    "comment_date": item.get(
                        "timestamp",
                        item.get("date", "")
                    ),

                    "post_url": item.get(
                        "postUrl",
                        item.get("post_url", "")
                    ),

                    "profile_pic": item.get(
                        "ownerProfilePicUrl",
                        ""
                    )
                })

    return pd.DataFrame(rows)


def create_db():

    df = load()

    print(df.head())

    conn = sqlite3.connect("instagram_data.db")

    df.to_sql(
        "comments",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    print("Database created")


if __name__ == "__main__":
    create_db()