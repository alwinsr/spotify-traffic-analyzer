import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("dataset.csv")

# -----------------------------
# Train content-type model
# -----------------------------
X_content = df.drop(columns=["label", "genre"])
y_content = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_content, y_content, test_size=0.2, random_state=42, stratify=y_content
)

content_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

content_pipeline.fit(X_train, y_train)
y_pred = content_pipeline.predict(X_test)
print("Content Type Classification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(content_pipeline, "content_type_model.pkl")

# -----------------------------
# Train genre model (only music)
# -----------------------------
music_df = df[df["label"] == "music"]
X_genre = music_df.drop(columns=["label", "genre"])
y_genre = music_df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X_genre, y_genre, test_size=0.2, random_state=42, stratify=y_genre
)

genre_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
])

genre_pipeline.fit(X_train, y_train)
y_pred = genre_pipeline.predict(X_test)
print("Music Genre Classification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(genre_pipeline, "genre_model.pkl")
