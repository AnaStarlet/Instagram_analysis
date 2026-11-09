import re
import torch
import pymorphy3
from transformers import AutoTokenizer, AutoModelForSequenceClassification

morph = pymorphy3.MorphAnalyzer()

SENT_MODEL = "blanchefort/rubert-base-cased-sentiment"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL).eval().to(device)
    ID2RU = {0: "Нейтрально", 1: "Позитив", 2: "Негатив"}
    MODEL_READY = True
    print(f"✅ Модель загружена на {device}")
except Exception as e:
    print(f"⚠️ Модель не загрузилась: {e}")
    MODEL_READY = False
    ID2RU = {}

BAD_TRIGGERS = [
    "ужас", "плохо", "кошмар", "обман", "отстой", "позор", "бесит",
    "глупо", "развод", "мошенники", "херня", "чушь", "вранье", "фу",
    "задолбали", "кидалово", "отвратительно", "кошмарно", "ужасно",
    "бардак", "хаос", "пробка", "очередь", "давка", "толпа", "цирк",
    "показуха", "галочка", "беспредел", "коррупция", "некомпетентн",
    "верните", "удалите", "уберите", "хватит", "надоело", "устал",
    "жаль", "проблема", "нарушение",
]

GOOD_TRIGGERS = [
    "спасибо", "благодарю", "супер", "класс", "ура", "лучше", "мощно",
    "красиво", "восторг", "круто", "талант", "умничка", "огонь", "браво",
    "обожаю", "рекомендую", "молодцы", "правильно", "поддерживаю",
    "❤️", "🔥", "👍", "🙏",
]

THEMES_MAP = {
    "Организация": [
        "очеред", "организац", "бардак", "хаос", "пробк", "перекр",
        "давк", "толп", "досмотр", "цирк", "показух", "галочк",
        "порядок", "подготовк", "логистик",
    ],
    "Безопасность": [
        "безопасн", "террор", "угроз", "экстремизм", "страшн",
        "боюсь", "опасн", "пресек", "полиц", "мвд", "росгвард",
        "охрана", "защит", "провокац",
    ],
    "Логистика": [
        "автобус", "проезд", "парков", "метро", "добрат",
        "перекрыт", "дорог", "транспорт", "разъехат", "пробка",
        "закрыт", "движени",
    ],
    "Эмоции": [
        "спасибо", "молодц", "ужас", "кошмар", "надоел",
        "устал", "ненавиж", "любл", "рад", "страх",
        "пережив", "волну", "верю", "надежд",
    ],
}


def clean_text(text):
    text = re.sub(r"http\S+", " ", str(text))
    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def get_lemmas(text):
    lemmas = []
    for word in clean_text(text).split():
        try:
            lemmas.append(morph.parse(word)[0].normal_form)
        except Exception:
            pass
    return lemmas


@torch.no_grad()
def analyze_sentiment(texts):
    if not MODEL_READY:
        return ["Нейтрально"] * len(texts)

    texts = [str(t)[:512] for t in texts]
    out = []

    BATCH = 64
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().numpy()

        for j, p in enumerate(preds):
            label = ID2RU[int(p)]
            txt = batch[j].lower()

            has_bad = any(w in txt for w in BAD_TRIGGERS)
            has_good = any(w in txt for w in GOOD_TRIGGERS)

            if has_bad and not has_good:
                label = "Негатив"
            elif has_good and label == "Нейтрально":
                label = "Позитив"

            out.append(label)

    return out


def classify_theme(text):
    text_l = clean_text(text)
    scores = {th: sum(1 for kw in kws if kw in text_l)
              for th, kws in THEMES_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Общее"


def calculate_priority_score(row):
    text = str(row.get("comment_text", "")).lower()
    sentiment = row.get("sentiment", "")
    theme = row.get("theme", "")

    score = 1.0

    if len(text) > 100:
        score += 1.0
    if "?" in text:
        score += 1.0
    if sentiment == "Негатив":
        score += 1.5
    if theme in ("Безопасность", "Организация"):
        score += 0.5

    return round(min(score, 5.0), 1)


def generate_insights(df, full_df=None):
    if df.empty:
        return {"summary": "Нет данных для анализа.", "blocks": []}

    insights = {"blocks": []}
    total = len(df)

    if full_df is not None and len(full_df) > total:
        from embeddings_utils import compare_with_norm
        norm = compare_with_norm(df, full_df)
        if norm is not None:
            insights["blocks"].append({
                "type": "norm",
                "title": "0. Сравнение с нормой по датасету",
                "items": [
                    f"Доля негатива в выборке: {norm['filtered_neg_pct']:.1f}%",
                    f"Доля негатива в среднем по датасету: {norm['full_neg_pct']:.1f}%",
                    f"Отклонение: {norm['diff']:+.1f} п.п. (соотношение ×{norm['ratio']:.2f})",
                ],
                "conclusion": norm["message"],
            })

    neg_pct = (df["sentiment"] == "Негатив").mean() * 100
    pos_pct = (df["sentiment"] == "Позитив").mean() * 100
    neu_pct = (df["sentiment"] == "Нейтрально").mean() * 100

    dominant = df["sentiment"].mode()[0] if not df.empty else "Нейтрально"

    insights["summary"] = (
        f"Проанализировано **{total}** комментариев. "
        f"Доминирующая тональность — **{dominant}** "
        f"({df['sentiment'].value_counts().iloc[0]/total*100:.1f}%). "
        f"Индекс поляризации (разница между позитивом и негативом): "
        f"**{abs(pos_pct - neg_pct):.1f}%**."
    )

    insights["blocks"].append({
        "type": "sentiment",
        "title": "1. Структура тональности",
        "items": [
            f"Позитив: {pos_pct:.1f}% ({len(df[df['sentiment']=='Позитив'])})",
            f"Нейтрально: {neu_pct:.1f}% ({len(df[df['sentiment']=='Нейтрально'])})",
            f"Негатив: {neg_pct:.1f}% ({len(df[df['sentiment']=='Негатив'])})",
        ],
        "conclusion": _sentiment_conclusion(neg_pct, pos_pct, total)
    })

    theme_counts = df["theme"].value_counts()
    top_theme = theme_counts.index[0] if len(theme_counts) > 0 else "-"

    negative_by_theme = []
    for theme in theme_counts.index:
        sub = df[df["theme"] == theme]
        if len(sub) < 3:
            continue
        neg_share = (sub["sentiment"] == "Негатив").mean() * 100
        negative_by_theme.append((theme, neg_share, len(sub)))

    negative_by_theme.sort(key=lambda x: -x[1])

    theme_items = [f"{theme}: {count} ({count/total*100:.1f}%)" for theme, count in theme_counts.head(5).items()]

    conclusion = f"Самая обсуждаемая тема — **{top_theme}**."
    if negative_by_theme:
        worst_theme, worst_share, worst_n = negative_by_theme[0]
        conclusion += (
            f" Наибольшая доля негатива — в теме **{worst_theme}** "
            f"({worst_share:.1f}%, {worst_n} комментов). "
            f"Это потенциальная зона риска."
        )

    insights["blocks"].append({
        "type": "themes",
        "title": "2. Тематическое распределение",
        "items": theme_items,
        "conclusion": conclusion
    })

    high_priority = df[df["priority"] >= 4.0]
    medium_priority = df[(df["priority"] >= 3.0) & (df["priority"] < 4.0)]

    priority_conclusion = (
        f"Комментариев с высоким приоритетом (≥4.0): **{len(high_priority)}** "
        f"({len(high_priority)/total*100:.1f}%). "
    )
    if len(high_priority) > 0:
        priority_conclusion += (
            f"Это критические сообщения, требующие первоочередного реагирования. "
            f"Среди них {len(high_priority[high_priority['sentiment']=='Негатив'])} негативных."
        )

    insights["blocks"].append({
        "type": "priority",
        "title": "3. Приоритизация реагирования",
        "items": [
            f"Высокий приоритет (4.0–5.0): {len(high_priority)}",
            f"Средний приоритет (3.0–4.0): {len(medium_priority)}",
            f"Обычный приоритет (<3.0): {len(df[df['priority'] < 3.0])}",
        ],
        "conclusion": priority_conclusion
    })

    if total > 50:
        avg_len = df["comment_text"].astype(str).str.len().mean()
        long_comments = df[df["comment_text"].astype(str).str.len() > avg_len * 1.5]

        if len(long_comments) > 0:
            long_neg_share = (long_comments["sentiment"] == "Негатив").mean() * 100
            general_neg_share = neg_pct

            diff = long_neg_share - general_neg_share
            trend = "выше" if diff > 0 else "ниже"

            insights["blocks"].append({
                "type": "correlation",
                "title": "4. Корреляция длины и тональности",
                "items": [
                    f"Средняя длина комментария: {avg_len:.0f} символов",
                    f"Длинных комментариев (>1.5× средней): {len(long_comments)}",
                    f"Доля негатива в длинных: {long_neg_share:.1f}%",
                ],
                "conclusion": (
                    f"В длинных комментариях доля негатива **{trend}** общей "
                    f"на {abs(diff):.1f}%. Это может указывать на то, что "
                    f"пользователи пишут развёрнуто, когда хотят выразить "
                    f"{'недовольство' if diff > 0 else 'поддержку'}."
                )
            })

    insights["conclusion"] = _final_conclusion(df, neg_pct, pos_pct, negative_by_theme)

    return insights


def _sentiment_conclusion(neg_pct, pos_pct, total):
    if neg_pct > 50:
        return "⚠️ Преобладание негатива. Требуется анализ причин и оперативное реагирование."
    elif neg_pct > 30:
        return "⚠️ Умеренный уровень негатива. Есть точки напряжения, требующие внимания."
    elif pos_pct > 50:
        return "✅ Преобладание позитива. Аудитория лояльна."
    else:
        return "ℹ️ Сбалансированная картина. Ярко выраженной полярности нет."


def _final_conclusion(df, neg_pct, pos_pct, negative_by_theme):
    total = len(df)
    parts = []

    if neg_pct > 30:
        parts.append(
            f"Выявлен повышенный уровень негатива ({neg_pct:.1f}%)."
        )
    elif pos_pct > 40:
        parts.append(
            f"Зафиксирована позитивная реакция аудитории ({pos_pct:.1f}% позитива)."
        )
    else:
        parts.append("Тональность нейтральная с умеренными отклонениями.")

    if negative_by_theme:
        worst_theme, worst_share, _ = negative_by_theme[0]
        parts.append(
            f"Ключевая зона риска — тема «{worst_theme}» "
            f"({worst_share:.1f}% негативных комментариев)."
        )

    high_priority = df[df["priority"] >= 4.0]
    if len(high_priority) > 0:
        parts.append(
            f"Обнаружено {len(high_priority)} критических сообщений "
            f"с высоким приоритетом."
        )

    return " ".join(parts)