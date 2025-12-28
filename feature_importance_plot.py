import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

content_pipeline = joblib.load("content_type_model.pkl")
rf = content_pipeline.named_steps["clf"]

df = pd.read_csv("dataset.csv")

features = df.drop(columns=["label", "genre"]).columns
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importance (Music vs Podcast)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()
plt.show()

TOP_N = 5
top_features = features[indices][:TOP_N]

print("Content type counts (Music vs Podcast):")
print(df["label"].value_counts())

music_df = df[df["label"] == "music"]

print("\nMusic genre counts:")
print(music_df["genre"].value_counts())

for label in ["music", "podcast"]:
    vals = np.sort(df[df["label"] == label]["pkt_size_bin_2"])
    y = np.arange(1, len(vals) + 1) / len(vals)
    plt.plot(vals, y, label=label)

plt.xlabel("pkt_size_bin_2")
plt.ylabel("Cumulative Probability")
plt.title("ECDF of pkt_size_bin_2")
plt.legend()
plt.grid(True)
plt.show()


print("Top features used for plots:")
for f in top_features:
    print("-", f)

for feature in top_features:
    plt.figure(figsize=(6,4))
    
    sns.violinplot(
        x="label",
        y=feature,
        data=df,
        inner="box",   # boxplot inside violin
        cut=0
    )

    plt.title(f"{feature} distribution (Music vs Podcast)")
    plt.xlabel("Content Type")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

music_df = df[df["label"] == "music"]

for feature in top_features:
    plt.figure(figsize=(10,5))
    
    sns.violinplot(
        x="genre",
        y=feature,
        data=music_df,
        inner="box",
        cut=0
    )

    plt.title(f"{feature} distribution across Music Genres")
    plt.xlabel("Genre")
    plt.ylabel(feature)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

