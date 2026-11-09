import streamlit as st
import pandas as pd
import embeddings_utils as eu
import sqlite3
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import os
import time
import tempfile
from datetime import datetime
from fpdf import FPDF

from analyzer import (
    analyze_sentiment,
    calculate_priority_score,
    classify_theme,
    generate_insights,
)

st.set_page_config(
    page_title="Insight Search",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, #E75480, #FF8FB1) !important;
    color: white !important;
    border-radius: 18px !important;
    border: none !important;
    height: 48px;
    font-weight: 600 !important;
}
.stTextInput input {
    border-radius: 24px !important;
    padding: 14px 22px !important;
    border: 1px solid #FFD3E0 !important;
}
div[data-testid="stMetric"] {
    border: 1px solid #FFD9E5;
    border-radius: 18px;
    padding: 18px;
}
.insight-box {
    border-left: 5px solid #E75480;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}
[data-testid="stSidebar"] {
    border-right: 2px solid #FFD3E0;
}
[data-testid="stSidebar"] h2 {
    color: #C94F7C;
    font-size: 22px !important;
}
.upload-info {
    border: 1px dashed #E75480;
    border-radius: 12px;
    padding: 12px;
    font-size: 13px;
    margin-top: 8px;
}
.file-loaded {
    border-left: 4px solid #2ecc71;
    border-radius: 8px;
    padding: 10px;
    margin-top: 8px;
    font-size: 13px;
}
@media (prefers-color-scheme: dark) {
    div[data-testid="stMetric"] { border: 1px solid #FF8FB1; }
    .insight-box { border-left: 5px solid #FF8FB1; }
    .upload-info { border: 1px dashed #FF8FB1; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_db():
    try:
        conn = sqlite3.connect("instagram_data.db")
        df = pd.read_sql("SELECT * FROM comments", conn)
        conn.close()
        df["comment_date"] = pd.to_datetime(df["comment_date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки стандартной БД: {e}")
        return pd.DataFrame()


def parse_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith(".json"):
        data = json.load(uploaded_file)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "comments" in data:
            df = pd.DataFrame(data["comments"])
        else:
            st.error("Неподдерживаемая структура JSON")
            return None
    else:
        st.error("Неподдерживаемый формат. Используйте CSV, Excel или JSON.")
        return None

    text_col = None
    for candidate in ["comment_text", "commentText", "text", "Текст", "Комментарий"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        for col in df.columns:
            if df[col].dtype == object:
                text_col = col
                break

    if text_col is None:
        st.error("Не найдена колонка с текстом. Назовите 'comment_text' или 'text'.")
        return None

    rename_map = {text_col: "comment_text"}
    if "username" in df.columns:
        rename_map["username"] = "comment_author"
    elif "comment_author" not in df.columns:
        df["comment_author"] = "unknown"

    if "publishedAt" in df.columns:
        rename_map["publishedAt"] = "comment_date"

    df = df.rename(columns=rename_map)

    if "comment_author" not in df.columns:
        df["comment_author"] = "unknown"
    if "post_url" not in df.columns:
        df["post_url"] = ""

    df = df.dropna(subset=["comment_text"])
    df = df[df["comment_text"].astype(str).str.strip().str.len() > 0]

    if "comment_date" in df.columns:
        df["comment_date"] = pd.to_datetime(df["comment_date"], errors="coerce")
        if df["comment_date"].isna().all():
            df["comment_date"] = pd.to_datetime(df["comment_date"], errors="coerce", dayfirst=True)
    else:
        df["comment_date"] = pd.NaT

    if "sentiment" not in df.columns:
        progress = st.progress(0, text="Размечаю тональность через RuBERT...")
        total = len(df)
        sentiments = []
        for i, text in enumerate(df["comment_text"].astype(str).tolist()):
            sentiments.append(analyze_sentiment([text])[0])
            if (i + 1) % max(1, total // 20) == 0:
                progress.progress((i + 1) / total, text=f"Размечаю: {i+1}/{total}")
        df["sentiment"] = sentiments
        progress.progress(0.7, text="Классифицирую темы...")
        df["theme"] = df["comment_text"].apply(classify_theme)
        progress.progress(0.9, text="Считаю Priority Score...")
        df["priority"] = df.apply(calculate_priority_score, axis=1)
        progress.progress(1.0, text="Готово!")
        time.sleep(0.5)
        progress.empty()
    else:
        if "theme" not in df.columns:
            df["theme"] = df["comment_text"].apply(classify_theme)
        if "priority" not in df.columns:
            df["priority"] = df.apply(calculate_priority_score, axis=1)

    return df


def generate_pdf_report(filtered_df, query):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    font_bold_path = "C:\\Windows\\Fonts\\arialbd.ttf"

    if os.path.exists(font_path):
        pdf.add_font("CustomFont", "", font_path, uni=True)
        if os.path.exists(font_bold_path):
            pdf.add_font("CustomFont", "B", font_bold_path, uni=True)
        font_name = "CustomFont"
    else:
        font_name = "helvetica"

    pdf.set_font(font_name, "B", 20)
    pdf.cell(0, 15, "Insight Search", ln=True, align="C")
    pdf.set_font(font_name, "", 12)
    pdf.cell(0, 8, "Отчёт по анализу комментариев", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font(font_name, "", 10)
    pdf.cell(0, 6, f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}", ln=True, align="C")
    pdf.cell(0, 6, f"Запрос: {query}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font(font_name, "B", 14)
    pdf.cell(0, 10, "1. Общая статистика", ln=True)
    pdf.set_font(font_name, "", 11)

    total = len(filtered_df)
    neg = len(filtered_df[filtered_df["sentiment"] == "Негатив"])
    neu = len(filtered_df[filtered_df["sentiment"] == "Нейтрально"])
    pos = len(filtered_df[filtered_df["sentiment"] == "Позитив"])
    top_theme = filtered_df["theme"].mode()[0] if not filtered_df.empty else "-"
    avg_priority = round(filtered_df["priority"].mean(), 2) if not filtered_df.empty else 0

    pdf.cell(0, 8, f"Всего комментариев: {total}", ln=True)
    if total > 0:
        pdf.cell(0, 8, f"Негатив: {neg} ({neg/total*100:.1f}%)", ln=True)
        pdf.cell(0, 8, f"Нейтрально: {neu} ({neu/total*100:.1f}%)", ln=True)
        pdf.cell(0, 8, f"Позитив: {pos} ({pos/total*100:.1f}%)", ln=True)
    pdf.cell(0, 8, f"Главная тема: {top_theme}", ln=True)
    pdf.cell(0, 8, f"Средний приоритет: {avg_priority}", ln=True)
    pdf.ln(5)

    pdf.set_font(font_name, "B", 14)
    pdf.cell(0, 10, "2. Графики", ln=True)
    pdf.ln(2)

    def save_plotly_fig(fig, width=500, height=350):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.write_image(tmp.name, width=width, height=height, scale=2)
        pdf.image(tmp.name, x=15, w=180)
        os.unlink(tmp.name)

    try:
        fig_pie = px.pie(
            filtered_df, names="sentiment", hole=0.4, color="sentiment",
            color_discrete_map={"Позитив": "#FF8FB1", "Негатив": "#E75480", "Нейтрально": "#FFB7CE"},
            title="Распределение тональности",
        )
        save_plotly_fig(fig_pie)
        pdf.ln(2)
    except Exception:
        pass

    try:
        theme_stats = filtered_df["theme"].value_counts().reset_index()
        theme_stats.columns = ["theme", "count"]
        theme_stats = theme_stats[theme_stats["theme"] != "Общее"]
        if not theme_stats.empty:
            fig_bar = px.bar(
                theme_stats, x="theme", y="count", color="theme",
                color_discrete_sequence=["#E75480", "#FF8FB1", "#FFB7CE", "#FFC2D1"],
                title="Распределение по темам (без «Общее»)",
            )
            save_plotly_fig(fig_bar)
            pdf.ln(2)
    except Exception:
        pass

    pdf.add_page()
    pdf.set_font(font_name, "B", 14)
    pdf.cell(0, 10, "3. Список комментариев", ln=True)
    pdf.ln(2)

    col_w = {"num": 8, "text": 90, "author": 30, "sent": 30, "theme": 32}
    row_h = 6

    def draw_header():
        pdf.set_font(font_name, "B", 8)
        pdf.cell(col_w["num"], 7, "#", border=1)
        pdf.cell(col_w["text"], 7, "Комментарий", border=1)
        pdf.cell(col_w["author"], 7, "Автор", border=1)
        pdf.cell(col_w["sent"], 7, "Тональность", border=1)
        pdf.cell(col_w["theme"], 7, "Тема", border=1, ln=True)

    draw_header()
    pdf.set_font(font_name, "", 7)

    for idx, (_, row) in enumerate(filtered_df.iterrows(), 1):
        text = str(row["comment_text"]).replace("\n", " ").strip()
        if len(text) > 150:
            text = text[:147] + "..."
        author = str(row["comment_author"])[:20]
        sent = str(row["sentiment"])[:15]
        theme = str(row["theme"])[:15]

        if pdf.will_page_break(row_h * 3):
            pdf.add_page()
            draw_header()
            pdf.set_font(font_name, "", 7)

        y_start = pdf.get_y()
        x_start = pdf.l_margin

        pdf.set_xy(x_start, y_start)
        pdf.cell(col_w["num"], row_h, str(idx), border=1)

        x_text = x_start + col_w["num"]
        pdf.set_xy(x_text, y_start)
        pdf.multi_cell(col_w["text"], row_h, text, border=1, new_x="RIGHT", new_y="TOP")
        text_h = pdf.get_y() - y_start

        row_total_h = max(row_h, text_h)
        y_after = y_start + row_total_h

        pdf.set_xy(x_text + col_w["text"], y_start)
        pdf.cell(col_w["author"], row_total_h, author, border=1)

        pdf.set_xy(x_text + col_w["text"] + col_w["author"], y_start)
        pdf.cell(col_w["sent"], row_total_h, sent, border=1)

        pdf.set_xy(x_text + col_w["text"] + col_w["author"] + col_w["sent"], y_start)
        pdf.cell(col_w["theme"], row_total_h, theme, border=1)

        pdf.set_y(y_after)

    return bytes(pdf.output())


with st.sidebar:
    st.markdown("## 📊 Источник данных")

    st.markdown("### 📁 Загрузить свой файл")
    uploaded_file = st.file_uploader(
        "Выберите файл",
        type=["csv", "xlsx", "xls", "json"],
        help="CSV, Excel или JSON. Колонка с текстом: comment_text или text.",
    )

    st.markdown("""
    <div class="upload-info">
    <b>Форматы:</b><br>
    • CSV — comment_text, comment_author, publishedAt<br>
    • Excel (.xlsx, .xls)<br>
    • JSON — список или {"comments": [...]}<br>
    <b>Обязательно:</b> колонка с текстом.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📦 Или стандартный")
    use_default = st.checkbox("Использовать стандартный датасет (Instagram-МВД, 7000)",
                               value=(uploaded_file is None))

    st.divider()
    st.markdown("### О приложении")
    st.markdown("""
    **Insight Search** — анализ комментариев Instagram:
    • Тональность (RuBERT-base)
    • Темы (Организация, Безопасность, Логистика, Эмоции)
    • Priority Score (1–5)
    """)


if uploaded_file is not None:
    comments_all = parse_uploaded_file(uploaded_file)
    if comments_all is None or len(comments_all) == 0:
        st.error("Не удалось загрузить файл или он пуст.")
        st.stop()
    st.sidebar.markdown(f"""
    <div class="file-loaded">
    ✅ Загружено: <b>{len(comments_all)}</b> строк<br>
    📄 {uploaded_file.name}
    </div>
    """, unsafe_allow_html=True)
elif use_default:
    comments_all = load_default_db()
    st.sidebar.success(f"✅ Загружено: {len(comments_all)} комментариев")
else:
    st.info("👈 Загрузите файл или включите стандартный датасет в левом меню.")
    st.stop()


st.markdown("<h1 style='text-align:center; font-size:52px; font-weight:800;'>Insight Search</h1>",
            unsafe_allow_html=True)
st.markdown("<div style='text-align:center; opacity:0.7; font-size:17px; margin-bottom:24px;'>"
            "Instagram Content Analytics Platform</div>",
            unsafe_allow_html=True)

if "search_results" not in st.session_state:
    st.session_state["search_results"] = pd.DataFrame()

col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    search_query = st.text_input(
        "Search",
        placeholder="Введите ключевые слова, URL или никнейм...",
        label_visibility="collapsed",
    )
    mode = st.radio(
        "Mode",
        ["🔑 По словам", "🔗 По URL", "👤 По автору", "✍️ Один текст"],
        horizontal=True,
        label_visibility="collapsed",
    )
    analyze_clicked = st.button("Анализировать", use_container_width=True)

st.divider()

if analyze_clicked:
    if not search_query.strip():
        st.warning("Введите запрос")
    else:
        progress = st.progress(0, text="Ищу комментарии...")
        time.sleep(0.3)

        term = search_query.strip().lower()
        result = pd.DataFrame()

        if mode == "🔑 По словам":
            keys = [k.strip().lower() for k in search_query.split(",")]
            result = comments_all[
                comments_all["comment_text"].apply(
                    lambda x: any(k in str(x).lower() for k in keys)
                )
            ].copy()
        elif mode == "🔗 По URL":
            result = comments_all[
                comments_all["post_url"].astype(str).str.contains(term, na=False, case=False)
            ].copy()
        elif mode == "👤 По автору":
            result = comments_all[
                comments_all["comment_author"].astype(str).str.lower().str.contains(term, na=False)
            ].copy()
        elif mode == "✍️ Один текст":
            result = pd.DataFrame([{
                "comment_text": search_query,
                "comment_author": "Вы",
                "comment_date": pd.Timestamp.now(),
                "post_url": "",
            }])

        progress.progress(0.5, text="Анализирую тональность...")

        if result.empty:
            progress.empty()
            st.error("Ничего не найдено")
            st.session_state["search_results"] = pd.DataFrame()
        else:
            if "sentiment" not in result.columns:
                result["sentiment"] = analyze_sentiment(result["comment_text"].tolist())
            if "theme" not in result.columns:
                result["theme"] = result["comment_text"].apply(classify_theme)
            if "priority" not in result.columns:
                result["priority"] = result.apply(calculate_priority_score, axis=1)

            progress.progress(1.0, text="Готово!")
            time.sleep(0.4)
            progress.empty()

            st.session_state["search_results"] = result


if not st.session_state["search_results"].empty:
    df_result = st.session_state["search_results"]

    f1, f2, f3 = st.columns(3)
    with f1:
        sentiment_filter = st.multiselect(
            "Тональность",
            df_result["sentiment"].unique(),
            default=df_result["sentiment"].unique(),
        )
    with f2:
        theme_filter = st.multiselect(
            "Темы",
            df_result["theme"].unique(),
            default=df_result["theme"].unique(),
        )
    with f3:
        priority_filter = st.slider("Минимальный приоритет", 1.0, 5.0, 1.0)

    filtered = df_result[
        (df_result["sentiment"].isin(sentiment_filter))
        & (df_result["theme"].isin(theme_filter))
        & (df_result["priority"] >= priority_filter)
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Комментариев", len(filtered))
    c2.metric("Негатив", len(filtered[filtered["sentiment"] == "Негатив"]))
    c3.metric("Главная тема",
              filtered["theme"].mode()[0] if not filtered.empty else "-")
    c4.metric("Средний приоритет",
              round(filtered["priority"].mean(), 1) if not filtered.empty else 0)

    tab1, tab2, tab3 = st.tabs(["📊 Аналитика", "💬 Комментарии", "🧠 Инсайты"])

    with tab1:
        g1, g2 = st.columns(2)
        with g1:
            fig = px.pie(
                filtered,
                names="sentiment",
                hole=0.45,
                color="sentiment",
                color_discrete_map={
                    "Позитив": "#FF8FB1",
                    "Негатив": "#E75480",
                    "Нейтрально": "#FFB7CE",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            if len(filtered) > 0:
                text = " ".join(filtered["comment_text"].astype(str))
                wc = WordCloud(width=800, height=400, background_color=None,
                               mode="RGBA", colormap="RdPu").generate(text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig_wc)

        theme_stats = filtered["theme"].value_counts().reset_index()
        theme_stats.columns = ["theme", "count"]
        fig2 = px.bar(theme_stats, x="theme", y="count", color="theme",
                      color_discrete_sequence=["#E75480", "#FF8FB1", "#FFB7CE", "#FFC2D1"])
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Упоминания ключевых слов и доля негатива")
        if search_query.strip():
            keywords = [k.strip() for k in search_query.split(",") if k.strip()]
            kw_rows = []
            for kw in keywords:
                mask = filtered["comment_text"].str.lower().str.contains(kw.lower(), na=False)
                sub = filtered[mask]
                if len(sub) == 0:
                    continue
                kw_rows.append({
                    "Ключевое слово": kw,
                    "Упоминаний": len(sub),
                    "Негатив %": round((sub["sentiment"] == "Негатив").mean() * 100, 1),
                    "Нейтрально %": round((sub["sentiment"] == "Нейтрально").mean() * 100, 1),
                    "Позитив %": round((sub["sentiment"] == "Позитив").mean() * 100, 1),
                })
            kw_df = pd.DataFrame(kw_rows)
            if not kw_df.empty:
                st.dataframe(kw_df, use_container_width=True, hide_index=True)
                fig_kw = px.bar(kw_df.sort_values("Негатив %"),
                                x="Негатив %", y="Ключевое слово",
                                orientation="h",
                                color="Негатив %",
                                color_continuous_scale="RdPu",
                                title="Доля негативных комментариев по ключевым словам")
                st.plotly_chart(fig_kw, use_container_width=True)

        st.markdown("### Динамика упоминаний по дням")
        daily = filtered.copy()
        daily["comment_date"] = pd.to_datetime(daily["comment_date"], errors="coerce")
        daily = daily.dropna(subset=["comment_date"])
        if not daily.empty:
            daily["date"] = daily["comment_date"].dt.date
            daily_agg = daily.groupby(["date", "sentiment"]).size().reset_index(name="count")
            fig_dyn = px.line(daily_agg, x="date", y="count", color="sentiment",
                              markers=True,
                              color_discrete_map={"Позитив": "#FF8FB1",
                                                  "Негатив": "#E75480",
                                                  "Нейтрально": "#FFB7CE"},
                              title="Количество упоминаний по дням")
            st.plotly_chart(fig_dyn, use_container_width=True)

        st.markdown("### Тренд-анализ тональности")

        daily_trend, slope, direction = eu.compute_trend(filtered)
        if daily_trend is not None and len(daily_trend) >= 2:
            if direction == "up":
                icon, text = "📈", "негатив растёт"
                color = "#e74c3c"
            elif direction == "down":
                icon, text = "📉", "негатив снижается"
                color = "#2ecc71"
            else:
                icon, text = "➡️", "уровень негатива стабилен"
                color = "#95a5a6"

            st.markdown(f"""
            <div class="insight-box" style="border-left: 5px solid {color};">
                <b>{icon} Тренд:</b> {text}
                (наклон: {slope:+.2f}% в день).
                {f"За {len(daily_trend)} дней наблюдалось изменение." if len(daily_trend) > 1 else ""}
            </div>
            """, unsafe_allow_html=True)

            fig_trend = px.line(
                daily_trend, x="date", y="neg_share",
                markers=True,
                title="Доля негативных комментариев по дням, %",
            )
            fig_trend.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Недостаточно данных для тренд-анализа (нужно минимум 3 даты).")

    with tab2:
        table_df = filtered.copy()
        table_df["comment_date"] = pd.to_datetime(table_df["comment_date"], errors="coerce")
        table_df["comment_length"] = table_df["comment_text"].astype(str).apply(len)

        st.markdown("### Фильтрация комментариев")
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            sort_option = st.selectbox("Сортировка",
                                       ["По приоритету", "По новизне", "По длине"])
        with fc2:
            min_length = st.slider("Мин. длина", 0, 500, 0)
        with fc3:
            max_length = st.slider("Макс. длина", 0, 1000, 1000)
        with fc4:
            only_questions = st.checkbox("Только вопросы")

        table_df = table_df[(table_df["comment_length"] >= min_length)
                            & (table_df["comment_length"] <= max_length)]
        if only_questions:
            table_df = table_df[table_df["comment_text"].astype(str).str.contains(r"\?")]

        if sort_option == "По приоритету":
            table_df = table_df.sort_values(by="priority", ascending=False)
        elif sort_option == "По новизне":
            table_df = table_df.sort_values(by="comment_date", ascending=False)
        elif sort_option == "По длине":
            table_df = table_df.sort_values(by="comment_length", ascending=False)

        available_cols = ["comment_text", "comment_author", "comment_date",
                          "sentiment", "theme", "priority", "comment_length", "post_url"]
        show_cols = [c for c in available_cols if c in table_df.columns]
        st.dataframe(table_df[show_cols], use_container_width=True, hide_index=True, height=600)
        st.markdown(f"Найдено комментариев: **{len(table_df)}**")

    with tab3:
        st.markdown("## 🧠 Аналитический отчёт")

        insights_data = generate_insights(filtered, full_df=comments_all)

        st.markdown(f"""
        <div class="insight-box" style="border-left: 5px solid #3498db;">
            <b>📋 Общая сводка</b><br>
            {insights_data['summary']}
        </div>
        """, unsafe_allow_html=True)

        for block in insights_data["blocks"]:
            with st.expander(f"**{block['title']}**", expanded=True):
                for item in block["items"]:
                    st.markdown(f"- {item}")

                if block.get("conclusion"):
                    st.markdown(f"""
                    <div class="insight-box" style="border-left: 4px solid #e67e22; margin-top: 10px;">
                        <b>💡 Вывод:</b> {block['conclusion']}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div class="insight-box" style="border-left: 5px solid #2ecc71; background: linear-gradient(145deg, white, #f0fff4);">
            <b>🎯 Итоговый вывод</b><br><br>
            {insights_data['conclusion']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 🔍 Кластеры критики")

        st.markdown("""
        <div class="insight-box" style="border-left: 5px solid #9b59b6;">
            <b>Как это работает:</b> мы используем эмбеддинги RuBERT-tiny2
            для семантической группировки негативных комментариев.
            Каждый кластер — это отдельная проблема или тема недовольства.
        </div>
        """, unsafe_allow_html=True)

        neg_only = filtered[filtered["sentiment"] == "Негатив"]
        if len(neg_only) >= 6:
            with st.spinner("Кластеризация негативных комментариев..."):
                work, descriptions = eu.cluster_comments(filtered, n_clusters=4, only_negative=True)

            if descriptions:
                for c, desc in descriptions.items():
                    st.markdown(f"""
                    <div class="insight-box" style="border-left: 5px solid #9b59b6;">
                        <b>Кластер #{c+1}</b> — {desc['size']} комментариев<br><br>
                        <b>Топ-слова:</b> {', '.join(desc['top_words'][:6])}<br><br>
                        <b>Примеры:</b><br>
                        """ + "<br>".join([f"• {ex[:120]}" for ex in desc["examples"]]) + """
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Для кластеризации нужно минимум 6 негативных комментариев.")

    col_csv, col_pdf = st.columns(2)

    with col_csv:
        csv = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Скачать отчёт CSV",
            csv,
            "report.csv",
            "text/csv",
            use_container_width=True,
        )

    with col_pdf:
        try:
            pdf_bytes = generate_pdf_report(filtered, search_query)
            st.download_button(
                "📄 Скачать полный отчёт (PDF)",
                pdf_bytes,
                "insight_report.pdf",
                "application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Ошибка генерации PDF: {e}")