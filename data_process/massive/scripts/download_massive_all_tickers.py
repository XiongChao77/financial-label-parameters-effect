import json, time,random,requests,os
import pandas as pd

from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

current_work_dir = os.path.dirname(__file__) 
BASE_URL = "https://api.massive.com/v3/reference/tickers"


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


def add_api_key_to_url(url: str, api_key: str) -> str:
    """
    next_url 有时不带 apiKey，这里统一补上。
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["apiKey"] = [api_key]
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def create_retry_session(
    total_retries: int = 8,
    backoff_factor: float = 2.0,
) -> requests.Session:
    """
    创建带 urllib3 自动重试的 requests session。
    """
    session = requests.Session()

    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def get_with_manual_retry(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    max_attempts: int = 10,
    timeout: tuple[int, int] = (10, 120),
) -> requests.Response:
    """
    在 urllib3 Retry 外再包一层手动重试。

    用于处理：
    - ConnectionResetError
    - ConnectionError
    - ReadTimeout
    - RemoteDisconnected
    - 429 / 5xx 临时错误
    """
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)

            if resp.status_code in [429, 500, 502, 503, 504]:
                wait = min(120, 2 ** attempt) + random.uniform(0, 1)
                print(
                    f"HTTP {resp.status_code}. "
                    f"Retrying in {wait:.1f}s... "
                    f"attempt {attempt}/{max_attempts}"
                )
                time.sleep(wait)
                continue

            return resp

        except requests.exceptions.RequestException as e:
            last_error = e
            wait = min(120, 2 ** attempt) + random.uniform(0, 1)
            print(
                f"Request error: {repr(e)}. "
                f"Retrying in {wait:.1f}s... "
                f"attempt {attempt}/{max_attempts}"
            )
            time.sleep(wait)

    raise RuntimeError(f"Request failed after {max_attempts} attempts: {last_error}")


def append_rows_to_csv(rows: list[dict], output_csv: str) -> int:
    """
    直接追加写入最终 CSV，不使用 part 文件。
    """
    if not rows:
        return 0

    output_path = Path(output_csv)
    df = pd.DataFrame(rows)

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    df.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )

    return len(df)


def download_all_tickers(
    api_key: str,
    output_csv: str = "all_tickers.csv",
    market: str = "stocks",
    locale: str = "us",
    active: bool | None = None,
    limit: int = 1000,
    sleep_sec: float = 0.5,
    overwrite: bool = True,
):
    """
    下载 Massive All Tickers reference data。

    参数说明：
    - market="stocks": 只下载股票市场
    - locale="us": 只下载美国市场
    - active=None: 下载 active=true 和 active=false
    - active=True: 只下载当前活跃 ticker
    - active=False: 只下载非活跃 ticker
    - overwrite=True: 每次重新下载并覆盖旧 CSV
    """

    if not api_key:
        raise ValueError("API key is empty.")

    output_path = Path(output_csv)

    if overwrite and output_path.exists():
        output_path.unlink()
        print(f"Removed existing file: {output_csv}")

    params = {
        "apiKey": api_key,
        "market": market,
        "locale": locale,
        "limit": min(limit, 1000),
        "sort": "ticker",
        "order": "asc",
    }

    if active is not None:
        params["active"] = str(active).lower()

    print("=" * 80)
    print("Starting All Tickers download")
    print(f"Output CSV: {output_path.resolve()}")
    print(f"market: {market}")
    print(f"locale: {locale}")
    print(f"active: {active}")
    print(f"limit per page: {params['limit']}")
    print("=" * 80)

    session = create_retry_session()

    url = BASE_URL
    page = 0
    total_rows = 0

    while url:
        page += 1
        print(f"Downloading page {page}...")

        if page == 1:
            resp = get_with_manual_retry(session, url, params=params)
        else:
            url = add_api_key_to_url(url, api_key)
            resp = get_with_manual_retry(session, url)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Request failed at page {page}: "
                f"HTTP {resp.status_code}\n{resp.text[:1000]}"
            )

        data = resp.json()
        rows = data.get("results", [])

        written_rows = append_rows_to_csv(rows, output_csv)
        total_rows += written_rows

        print(
            f"  rows this page: {len(rows)}, "
            f"written rows: {written_rows}, "
            f"total rows: {total_rows}"
        )

        url = data.get("next_url")

        time.sleep(sleep_sec)

    print("=" * 80)
    print("All Tickers download finished")
    print(f"Total rows: {total_rows}")
    print(f"Saved to: {output_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    api_key_path = os.path.join(current_work_dir, "massive_key.json")
    API_KEY = load_api_key(api_key_path)

    download_all_tickers(
        api_key=API_KEY,
        output_csv=os.path.join(current_work_dir, "all_tickers.csv"),
        market="stocks",
        locale="us",
        active=None,       # 重要：None 表示 active 和 inactive 都下载
        limit=1000,
        sleep_sec=0.5,
        overwrite=True,
    )