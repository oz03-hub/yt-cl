# import asyncio
# from shazamio import Shazam

# async def main():
#     shazam = Shazam()
#     out = await shazam.recognize("../../../work/pi_vcpartridge_umass_edu/ytb_wavs/TJmVu-CXvHg.wav")
#     print(out)

# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())

import asyncio
from aiohttp_retry import ExponentialRetry
from shazamio import Shazam, HTTPClient
import pandas as pd
import time

WAV_DATA_PATH = "../../../work/pi_vcpartridge_umass_edu/ytb_wavs/"

async def main(input_path: str, output_path: str):
    shazam = Shazam(http_client=HTTPClient(retry_options=ExponentialRetry(attempts=3)))

    df = pd.read_csv(input_path)
    video_ids = df["video_id"].tolist()
    is_music = [-1] * len(video_ids)

    start_idx = 0
    if "is_music" in df.columns:
        processed_mask = df["is_music"] != -1
        start_idx = processed_mask.sum()
        is_music[:start_idx] = df["is_music"][:start_idx].tolist()
        print(f"Resuming from {start_idx}...")

    for i, video_id in enumerate(video_ids[start_idx:], start=start_idx):
        if i % 20 == 0:
            time.sleep(20)
        else:
            time.sleep(1)

        wav_file_path = f"{WAV_DATA_PATH}/{video_id}.wav"
        try:
            out = await shazam.recognize(wav_file_path)
            if len(out["matches"]) > 0:
                is_music[i] = 1
            else:
                is_music[i] = 0
        except Exception as e:
            is_music[i] = 0
            print(f"Error recognizing {video_id}: {e}")
        
        if (i+1) % 100 == 0:
            df["is_music"] = is_music
            df.to_csv(output_path, index=False)
            print(f"Checkpoint {i+1}/{len(video_ids)} saved")

    df["is_music"] = is_music
    df.to_csv(output_path, index=False)
    print(f"Finished processing {len(video_ids)} videos")

if __name__ == "__main__":
    input_path = "data/clean/transcripts_with_features_normalized.csv"
    output_path = "data/clean/transcripts_with_features_normalized_shazam.csv"
    asyncio.run(main(input_path, output_path))
