import librosa
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

AUDIO_DIR = "../../../work/pi_vcpartridge_umass_edu/ytb_wavs/"


# Function to extract multiple audio features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Compute features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_flux = np.mean(np.diff(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rms_energy = np.mean(librosa.feature.rms(y=y))

        return [zcr, spectral_centroid, spectral_flux, rms_energy]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Process all WAV files
data = []
file_names = []
for file_name in tqdm(os.listdir(AUDIO_DIR)):
    if file_name.endswith(".wav"):
        file_path = os.path.join(AUDIO_DIR, file_name)
        features = extract_features(file_path)

        if features:
            data.append(features)
            file_names.append(file_name.replace(".wav", ""))


# Convert to DataFrame
df = pd.DataFrame(
    data, columns=["ZCR", "SpectralCentroid", "SpectralFlux", "RMSEnergy"]
)
df["File"] = file_names

# Normalize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop(columns=["File"]))

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_scaled)

# Assign labels: The cluster with a higher mean ZCR is likely speech
cluster_means = df.groupby("Cluster")["ZCR"].mean()
speech_cluster = cluster_means.idxmax()

df["Classification"] = df["Cluster"].apply(
    lambda x: "Speech" if x == speech_cluster else "Music"
)

# Save results
df[
    ["File", "ZCR", "SpectralCentroid", "SpectralFlux", "RMSEnergy", "Classification"]
].to_csv("clustered_results.csv", index=False)

print("Clustering complete! Results saved to clustered_results.csv.")
