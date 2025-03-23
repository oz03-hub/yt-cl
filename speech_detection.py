from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
from pyannote.core.annotation import Annotation
from dotenv import load_dotenv
import os

load_dotenv()

model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=os.getenv("PYANNOTE_TOKEN"))
pipeline = VoiceActivityDetection(segmentation=model)

examples = [
    "apQoQ-MR1fA.wav"
]

HYPER_PARAMETERS = {
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 1.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}

pipeline.instantiate(HYPER_PARAMETERS)

results = []

for example in examples:
    vad = pipeline(example)
    results.append(vad)

print(results[0])
print(len(results[0]))
print(results[0].get_timeline().duration())
