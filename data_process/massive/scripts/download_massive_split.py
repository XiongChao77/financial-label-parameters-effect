import os
import time
import requests
import pandas as pd
import os,time
current_work_dir = os.path.dirname(__file__) 

from pathlib import Path
import json
def load_api_key(key_file: str = "massive_key.json") -> str:
    """
    从当前目录的 massive_key.json 读取 API key。

    支持以下字段名：
    - api_key
    - API_KEY
    - massive_api_key
    - MASSIVE_API_KEY
    """
    key_path = Path(key_file)

    if not key_path.exists():
        raise FileNotFoundError(f"Cannot find key file: {key_path.resolve()}")

    with open(key_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    possible_keys = [
        "api_key",
        "API_KEY",
        "massive_api_key",
        "MASSIVE_API_KEY",
    ]

    for k in possible_keys:
        if k in data and data[k]:
            return str(data[k]).strip()

    raise ValueError(
        f"No API key found in {key_file}. "
        f"Expected one of: {possible_keys}"
    )

api_key_path = os.path.join(current_work_dir, "massive_key.json")


API_KEY = load_api_key(api_key_path)

BASE_URL = "https://api.massive.com/stocks/v1/splits"

params = {
    "limit": 5000,
    "sort": "execution_date.asc",
    "apiKey": API_KEY,
}

all_rows = []
url = BASE_URL
page_no = 0

while url:
    page_no += 1
    print(f"Fetching page {page_no}: {url}")

    if url == BASE_URL:
        r = requests.get(url, params=params, timeout=60)
    else:
        r = requests.get(url, timeout=60)

    r.raise_for_status()
    data = r.json()

    rows = data.get("results", [])
    all_rows.extend(rows)

    print(f"  rows this page: {len(rows)}, total: {len(all_rows)}")

    next_url = data.get("next_url")

    if next_url:
        # next_url 有时不带 apiKey，保险起见补上
        if "apiKey=" not in next_url:
            sep = "&" if "?" in next_url else "?"
            next_url = f"{next_url}{sep}apiKey={API_KEY}"

    url = next_url

    # 防止触发限速
    time.sleep(0.2)

df = pd.DataFrame(all_rows)

print(df.head())
print(df.shape)

massive_splits_all = os.path.join(current_work_dir,'massive_splits_all.csv')
df.to_csv(massive_splits_all, index=False)

print(f"Saved to: {massive_splits_all}")