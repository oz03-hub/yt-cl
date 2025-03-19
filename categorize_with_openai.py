import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


df = pd.read_csv("data/merged.csv", index_col=0)

system_prompt = """You are a highly precise assistant designed to categorize YouTube videos based on their metadata into one of the following categories:  

1. **Bad**:  
   - The transcript, title, and description provide little to no meaningful context for topic modeling.  
   - Content consists of generic phrases, spam, gibberish, excessive self-promotion, or non-informative text.  
   - The transcript is too short, silent, or filled with unstructured noise (e.g., music, laughter, or random words).  

2. **Good**:  
   - At least one metadata source (title, description, or transcript) contains meaningful context relevant for topic modeling.  
   - The transcript is coherent and provides a discussion, narrative, or structured information that can be categorized.  
   - The title and description meaningfully describe the content of the video.  

Your task is to classify each video as either **"Good"** or **"Bad"** based on these definitions.  

You will be given a video title, description, transcript, duration in seconds, number of views, number of likes, number of channel followers, and assigned categories by Youtube. You can use these to help you make a better decision.

You must return **only one word**: "Good" or "Bad", without any explanations or additional text.
"""

base_prompt = """Please categorize the following YouTube video into one of the two categories:

- **Bad**: The metadata (title, description, and transcript) lacks meaningful context, contains only generic words, noise, or is otherwise not useful for topic modeling.
- **Good**: At least one metadata field contains valuable context, allowing the video to be classified by topic.

Return only "Good" or "Bad" with no additional text or explanation.
"""

def prompt_openai(row: pd.Series):
    prompt = f"""
    Video Title: {row["title"]}
    Video Description: {row["description"]}
    Video Transcript: {row["transcript"]}
    Video Duration: {row["duration"]} seconds
    Number of Views: {row["view_count"]}
    Number of Likes: {row["like_count"]}
    Number of Channel Followers: {row["channel_follower_count"]}
    Assigned Categories by Youtube: {row["categories"]}

    Return only one word: "Good" or "Bad". Try to be generous with the "Good" category.
    """

    return base_prompt + prompt

def get_openai_response(prompt: str):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature=0.1)

    return response.choices[0].message.content.strip()

# multithreaded
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(get_openai_response, prompt_openai(row)) for index, row in df.iterrows()]
    results = [future.result() for future in tqdm(futures)]

df["openai_response"] = results

df.to_csv("data/merged_categorized.csv", index=False)
