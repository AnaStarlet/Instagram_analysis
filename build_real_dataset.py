import os
import glob
import pandas as pd
from analyzer import analyze_sentiment, classify_theme, calculate_priority_score

DATA_DIR = "real_data" 
OUTPUT_CSV = "real_comments_labeled.csv"
OUTPUT_DB = "real_data.db"

print(f"Ищу CSV в папке: {DATA_DIR}")

all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Найдено файлов: {len(all_files)}")

dfs = []
for file in all_files:
    try:
        for enc in ["utf-8", "utf-8-sig", "cp1251"]:
            try:
                df = pd.read_csv(file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"⚠️ Не удалось прочитать {file}")
            continue

        if "commentText" not in df.columns:
            print(f"⚠️ В файле {file} нет колонки commentText, пропускаю")
            continue

        post_name = os.path.basename(file).replace(".csv", "")
        df["post_url"] = f"https://www.instagram.com/p/{post_name}/"

        dfs.append(df)
        print(f"✅ {file}: {len(df)} комментариев")
    except Exception as e:
        print(f"❌ Ошибка в {file}: {e}")

if not dfs:
    print("Нет данных для обработки")
    exit()

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nВсего до обработки: {len(df_all)} комментариев")


df_all = df_all.dropna(subset=["commentText"])
df_all = df_all[df_all["commentText"].astype(str).str.strip().str.len() > 0]

df_all = df_all.drop_duplicates(subset=["username", "commentText", "publishedAt"])

print(f"После очистки (без пустых и дублей): {len(df_all)}")

print("\nРазмечаю тональность через RuBERT (это займёт пару минут)...")
texts = df_all["commentText"].astype(str).tolist()
df_all["sentiment"] = analyze_sentiment(texts)

print("Классифицирую темы...")
df_all["theme"] = df_all["commentText"].apply(classify_theme)

df_for_priority = df_all.rename(columns={"commentText": "comment_text"})
print("Считаю Priority Score...")
df_all["priority"] = df_for_priority.apply(calculate_priority_score, axis=1)

final = pd.DataFrame({
    "comment_id": df_all["id"],
    "comment_text": df_all["commentText"],
    "comment_author": df_all["username"],
    "comment_date": pd.to_datetime(df_all["publishedAt"], format="%d.%m.%Y, %H:%M:%S", errors="coerce"),
    "post_url": df_all["post_url"],
    "post_text": "Instagram post", 
    "likes_count": 0,             
    "sentiment": df_all["sentiment"],
    "theme": df_all["theme"],
    "priority": df_all["priority"],
})


final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ CSV сохранён: {OUTPUT_CSV}")


import sqlite3
if os.path.exists(OUTPUT_DB):
    os.remove(OUTPUT_DB)

conn = sqlite3.connect(OUTPUT_DB)
final.to_sql("comments", conn, if_exists="replace", index=False)
conn.close()
print(f"✅ SQLite сохранён: {OUTPUT_DB}")


print("\n" + "="*60)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*60)
print(f"\nВсего комментариев: {len(final)}")
print(f"\nРаспределение тональности:")
print(final["sentiment"].value_counts())
print(f"\nРаспределение тем:")
print(final["theme"].value_counts())
print(f"\nPriority: min={final['priority'].min()}, max={final['priority'].max()}, mean={final['priority'].mean():.2f}")

print("\n" + "="*60)
print("ПРИМЕРЫ (первые 10):")
print("="*60)
for i, row in final.head(10).iterrows():
    print(f"\n@{row['comment_author']}: {row['comment_text'][:100]}")
    print(f"   → {row['sentiment']} | {row['theme']} | {row['priority']}")