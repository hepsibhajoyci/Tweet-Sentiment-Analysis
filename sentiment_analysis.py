"""
Advanced Twitter Sentiment Analysis
====================================
Features:
- VADER + TextBlob dual-engine sentiment scoring
- Confidence scoring & score distribution
- Entity-level breakdown
- Confusion matrix vs. ground truth labels
- Word frequency & wordcloud per sentiment
- Interactive-style multi-panel dashboard saved as PNG
- CSV export of enriched results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# ── NLP ──────────────────────────────────────────────────────────────────────
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠  TextBlob not installed – running VADER-only mode.")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("⚠  wordcloud not installed – skipping word-cloud panels.")

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ── NLTK downloads ────────────────────────────────────────────────────────────
for resource in ["vader_lexicon", "stopwords", "punkt", "punkt_tab"]:
    nltk.download(resource, quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n📂  Loading dataset …")
df = pd.read_csv("twitter_training.csv", header=None,
                 names=["ID", "Entity", "Sentiment", "Text"])

# Drop rows with missing text or sentiment
df.dropna(subset=["Text", "Sentiment"], inplace=True)
df["Text"] = df["Text"].astype(str).str.strip()
df = df[df["Text"].str.len() > 2].reset_index(drop=True)

# Normalise ground-truth labels
label_map = {
    "positive": "Positive", "negative": "Negative",
    "neutral":  "Neutral",  "irrelevant": "Irrelevant"
}
df["Sentiment"] = df["Sentiment"].str.lower().map(label_map).fillna("Unknown")
print(f"   Rows after cleaning: {len(df):,}")
print(f"   Ground-truth distribution:\n{df['Sentiment'].value_counts()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # mentions / hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)             # punctuation / numbers
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

df["Clean_Text"] = df["Text"].apply(clean_text)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SENTIMENT ENGINES
# ─────────────────────────────────────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()

def vader_scores(text: str) -> dict:
    return sia.polarity_scores(str(text))

def label_from_compound(score: float) -> str:
    if score >= 0.05:  return "Positive"
    if score <= -0.05: return "Negative"
    return "Neutral"

# VADER
print("⚙️   Running VADER …")
vader_results = df["Text"].apply(vader_scores)
df["VADER_Compound"] = vader_results.apply(lambda x: x["compound"])
df["VADER_Pos"]      = vader_results.apply(lambda x: x["pos"])
df["VADER_Neg"]      = vader_results.apply(lambda x: x["neg"])
df["VADER_Neu"]      = vader_results.apply(lambda x: x["neu"])
df["VADER_Sentiment"] = df["VADER_Compound"].apply(label_from_compound)

# TextBlob (optional)
if TEXTBLOB_AVAILABLE:
    print("⚙️   Running TextBlob …")
    df["TB_Polarity"]     = df["Clean_Text"].apply(lambda t: TextBlob(t).sentiment.polarity)
    df["TB_Subjectivity"] = df["Clean_Text"].apply(lambda t: TextBlob(t).sentiment.subjectivity)
    df["TB_Sentiment"]    = df["TB_Polarity"].apply(label_from_compound)

    # Ensemble: average compound scores
    df["Ensemble_Score"] = (df["VADER_Compound"] + df["TB_Polarity"]) / 2
    df["Predicted_Sentiment"] = df["Ensemble_Score"].apply(label_from_compound)
else:
    df["Predicted_Sentiment"] = df["VADER_Sentiment"]

# Confidence = absolute value of the primary compound score
df["Confidence"] = df["VADER_Compound"].abs()

print("\n✅  Sentiment prediction distribution:")
print(df["Predicted_Sentiment"].value_counts())


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EVALUATION vs. GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────
# Keep only labels that appear in both columns
eval_labels = ["Positive", "Negative", "Neutral"]
eval_df = df[df["Sentiment"].isin(eval_labels) &
             df["Predicted_Sentiment"].isin(eval_labels)]

print("\n📊  Classification Report (VADER vs. ground truth):")
print(classification_report(eval_df["Sentiment"],
                            eval_df["Predicted_Sentiment"],
                            labels=eval_labels))
acc = accuracy_score(eval_df["Sentiment"], eval_df["Predicted_Sentiment"])
print(f"   Overall accuracy: {acc:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HELPER – top words per sentiment
# ─────────────────────────────────────────────────────────────────────────────
def top_words(texts, n=15):
    tokens = []
    for t in texts:
        tokens.extend([w for w in word_tokenize(str(t))
                       if w.isalpha() and w not in STOP_WORDS and len(w) > 2])
    return Counter(tokens).most_common(n)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("\n🎨  Building dashboard …")

PALETTE = {
    "Positive":   "#2ecc71",
    "Negative":   "#e74c3c",
    "Neutral":    "#3498db",
    "Irrelevant": "#95a5a6",
}
BG   = "#0f0f1a"
CARD = "#1a1a2e"
TEXT = "#e8e8f0"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": "#333355", "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": "#222244",
    "font.family": "monospace",
})

fig = plt.figure(figsize=(22, 26), facecolor=BG)
fig.suptitle("🐦  Twitter Sentiment Analysis Dashboard",
             fontsize=22, fontweight="bold", color=TEXT, y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

colors = [PALETTE.get(s, "#888") for s in
          df["Predicted_Sentiment"].value_counts().index]

# ── A: Bar – predicted distribution ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts = df["Predicted_Sentiment"].value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[PALETTE.get(s, "#888") for s in counts.index],
               edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f"{val:,}", ha="center", va="bottom", fontsize=9)
ax1.set_title("Predicted Sentiment Distribution", fontweight="bold")
ax1.set_xlabel("Sentiment"); ax1.set_ylabel("Count")
ax1.yaxis.grid(True, linestyle="--", alpha=0.4)

# ── B: Pie chart ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
wedge_colors = [PALETTE.get(s, "#888") for s in counts.index]
ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=wedge_colors, startangle=140,
        wedgeprops={"edgecolor": BG, "linewidth": 1.5},
        textprops={"color": TEXT})
ax2.set_title("Sentiment Share", fontweight="bold")

# ── C: Ground-truth bar ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
gt_counts = df["Sentiment"].value_counts()
ax3.bar(gt_counts.index, gt_counts.values,
        color=[PALETTE.get(s, "#888") for s in gt_counts.index],
        edgecolor="white", linewidth=0.5)
ax3.set_title("Ground Truth Distribution", fontweight="bold")
ax3.set_xlabel("Sentiment"); ax3.set_ylabel("Count")
ax3.yaxis.grid(True, linestyle="--", alpha=0.4)

# ── D: Confusion matrix ──────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
cm = confusion_matrix(eval_df["Sentiment"],
                      eval_df["Predicted_Sentiment"],
                      labels=eval_labels)
cmap = LinearSegmentedColormap.from_list("cm", ["#0f0f1a", "#3498db", "#2ecc71"])
sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
            xticklabels=eval_labels, yticklabels=eval_labels,
            linewidths=0.5, linecolor="#333355", ax=ax4,
            annot_kws={"size": 12})
ax4.set_title("Confusion Matrix (Predicted vs. Ground Truth)", fontweight="bold")
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")

# ── E: VADER compound score distribution ─────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
for sent, grp in df.groupby("Predicted_Sentiment"):
    ax5.hist(grp["VADER_Compound"], bins=40, alpha=0.6,
             color=PALETTE.get(sent, "#888"), label=sent, density=True)
ax5.axvline(0.05, color="white", linestyle="--", linewidth=0.8)
ax5.axvline(-0.05, color="white", linestyle="--", linewidth=0.8)
ax5.set_title("VADER Compound Score Distribution", fontweight="bold")
ax5.set_xlabel("Compound Score"); ax5.set_ylabel("Density")
ax5.legend(fontsize=8)
ax5.yaxis.grid(True, linestyle="--", alpha=0.4)

# ── F: Confidence box-plot ───────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
sent_order = ["Positive", "Neutral", "Negative"]
box_data   = [df[df["Predicted_Sentiment"] == s]["Confidence"].values
              for s in sent_order]
bp = ax6.boxplot(box_data, labels=sent_order, patch_artist=True,
                 medianprops={"color": "white", "linewidth": 2})
for patch, s in zip(bp["boxes"], sent_order):
    patch.set_facecolor(PALETTE.get(s, "#888"))
ax6.set_title("Prediction Confidence", fontweight="bold")
ax6.set_ylabel("|Compound Score|")
ax6.yaxis.grid(True, linestyle="--", alpha=0.4)

# ── G: Top entities by tweet count ──────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
top_entities = df["Entity"].value_counts().head(10)
ax7.barh(top_entities.index[::-1], top_entities.values[::-1],
         color="#9b59b6", edgecolor="white", linewidth=0.5)
ax7.set_title("Top 10 Entities by Tweet Count", fontweight="bold")
ax7.set_xlabel("Tweets")
ax7.xaxis.grid(True, linestyle="--", alpha=0.4)

# ── H: Entity sentiment stacked bar (top 8) ──────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
top8 = df["Entity"].value_counts().head(8).index
entity_sent = (df[df["Entity"].isin(top8)]
               .groupby(["Entity", "Predicted_Sentiment"])
               .size().unstack(fill_value=0))
entity_sent = entity_sent.reindex(columns=["Positive","Neutral","Negative"],
                                  fill_value=0)
entity_sent_pct = entity_sent.div(entity_sent.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(entity_sent_pct))
for sent in ["Positive", "Neutral", "Negative"]:
    ax8.barh(entity_sent_pct.index, entity_sent_pct[sent],
             left=bottom, color=PALETTE[sent], label=sent,
             edgecolor="white", linewidth=0.3)
    bottom += entity_sent_pct[sent].values
ax8.set_title("Entity Sentiment Breakdown (%)", fontweight="bold")
ax8.set_xlabel("Percentage")
ax8.legend(loc="lower right", fontsize=7)

# ── I: Top words per sentiment ───────────────────────────────────────────────
sent_list = ["Positive", "Neutral", "Negative"]
for idx, sent in enumerate(sent_list):
    ax = fig.add_subplot(gs[3, idx])
    subset = df[df["Predicted_Sentiment"] == sent]["Clean_Text"]
    words  = top_words(subset, n=12)
    if words:
        wdf = pd.DataFrame(words, columns=["word", "count"])
        ax.barh(wdf["word"][::-1], wdf["count"][::-1],
                color=PALETTE[sent], edgecolor="white", linewidth=0.4)
    ax.set_title(f"Top Words – {sent}", fontweight="bold",
                 color=PALETTE[sent])
    ax.set_xlabel("Frequency")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

plt.savefig("sentiment_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("✅  Dashboard saved → sentiment_dashboard.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EXPORT ENRICHED DATASET
# ─────────────────────────────────────────────────────────────────────────────
export_cols = ["ID", "Entity", "Text", "Sentiment",
               "Predicted_Sentiment", "VADER_Compound",
               "VADER_Pos", "VADER_Neg", "VADER_Neu", "Confidence"]
if TEXTBLOB_AVAILABLE:
    export_cols += ["TB_Polarity", "TB_Subjectivity", "Ensemble_Score"]

df[export_cols].to_csv("sentiment_results_enriched.csv", index=False)
print("✅  Enriched CSV saved → sentiment_results_enriched.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  WORD CLOUDS (optional)
# ─────────────────────────────────────────────────────────────────────────────
if WORDCLOUD_AVAILABLE:
    fig_wc, axes_wc = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
    fig_wc.suptitle("Word Clouds by Predicted Sentiment",
                    fontsize=16, color=TEXT, fontweight="bold")
    for ax_wc, sent in zip(axes_wc, sent_list):
        text_blob = " ".join(df[df["Predicted_Sentiment"] == sent]["Clean_Text"])
        wc = WordCloud(width=500, height=300,
                       background_color="#1a1a2e",
                       colormap="RdYlGn" if sent == "Positive" else
                                "RdGy"   if sent == "Negative" else "Blues",
                       stopwords=STOP_WORDS,
                       max_words=80).generate(text_blob or "empty")
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title(sent, color=PALETTE[sent], fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("sentiment_wordclouds.png", dpi=130, bbox_inches="tight",
                facecolor=BG)
    print("✅  Word clouds saved → sentiment_wordclouds.png")
    plt.show()

print("\n🎉  Analysis complete!")