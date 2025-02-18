import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import re

def download_nltk_resources():
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

download_nltk_resources()

def normalize_transcript(transcript: str) -> str:
    transcript = re.sub(r"[^\w\s]|(?<!\d)\d(?!\d)", "", transcript)
    transcript = re.sub(r"\s+", " ", transcript).lower()
    transcript = " ".join(
        lemmatizer.lemmatize(word)
        for word in transcript.split()
        if word not in stop_words
    )
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

transcript_dir = Path("/work/pi_vcpartridge_umass_edu/ytb_csv/")
try:
    df = pd.read_csv(transcript_dir / "transcripts.csv", header=0)
    df.dropna(subset=["transcript"], inplace=True)
except FileNotFoundError:
    print("Error: transcripts.csv file not found.")
    df = pd.DataFrame()

df["transcript"] = df["transcript"].apply(normalize_transcript)

df.to_csv(transcript_dir / "transcripts_normalized.csv", index=False)
