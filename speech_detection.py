from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
from pyannote.core.annotation import Annotation
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import os

load_dotenv()

WAV_PATH = "../../../work/pi_vcpartridge_umass_edu/ytb_wavs/"

wav_files = [f for f in os.listdir(WAV_PATH) if f.endswith(".wav")]

model = Model.from_pretrained(
    "pyannote/segmentation-3.0", use_auth_token=os.getenv("PYANNOTE_TOKEN")
)
pipeline = VoiceActivityDetection(segmentation=model)

HYPER_PARAMETERS = {
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 1.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0,
}

pipeline.instantiate(HYPER_PARAMETERS)

results = []

for wav_file in tqdm(wav_files):
    vad = pipeline(WAV_PATH + wav_file)
    total_speech_duration = vad.get_timeline().duration()
    results.append((wav_file.replace(".wav", ""), total_speech_duration))

df = pd.DataFrame(results, columns=["video_id", "speech_duration"])
df.to_csv("speech_detection_results.csv", index=False)
