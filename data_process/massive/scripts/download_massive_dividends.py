import os
import time
import random
import requests
import pandas as pd

from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://api.massive.com/stocks/v1/dividends"
DATE_COL = "ex_dividend_date"

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
    创建带自动重试的 requests session。

    会自动重试：
    - 连接失败
    - 读取失败
    - 429 Too Many Requests
    - 5xx 服务端错误
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

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
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
    在 urllib3 Retry 外面再包一层手动重试。

    这样可以处理：
    - ConnectionResetError
    - ConnectionError
    - ReadTimeout
    - RemoteDisconnected
    - 服务端临时断开连接
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


def normalize_date_series(s: pd.Series) -> pd.Series:
    """
    统一日期格式为 YYYY-MM-DD 字符串。
    """
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def prepare_existing_csv_for_resume(output_csv: str) -> tuple[str | None, int]:
    """
    启动时扫描已有 CSV。

    如果文件存在：
    1. 读取已有数据
    2. 找到最新 ex_dividend_date
    3. 删除 ex_dividend_date >= 最新日期 的数据
    4. 把保留的数据直接写回原 CSV
    5. 返回最新日期，后续从这一天重新下载

    为什么删除 >= 最新日期？
    因为程序可能在下载某一天的数据时中断。
    从最新日期重新下载，可以保证这一天的数据完整。
    """
    output_path = Path(output_csv)

    if not output_path.exists():
        print(f"No existing file found: {output_csv}")
        return None, 0

    if output_path.stat().st_size == 0:
        print(f"Existing file is empty: {output_csv}")
        return None, 0

    print(f"Scanning existing file: {output_csv}")

    df = pd.read_csv(output_path)

    if df.empty:
        print("Existing file has no rows.")
        return None, 0

    if DATE_COL not in df.columns:
        raise ValueError(
            f"Existing CSV does not contain required column: {DATE_COL}"
        )

    df[DATE_COL] = normalize_date_series(df[DATE_COL])
    df = df.dropna(subset=[DATE_COL])

    if df.empty:
        print("Existing file has no valid ex_dividend_date.")
        output_path.unlink()
        return None, 0

    latest_date = df[DATE_COL].max()

    keep_df = df[df[DATE_COL] < latest_date].copy()
    removed_rows = len(df) - len(keep_df)

    if keep_df.empty:
        output_path.unlink()
        kept_rows = 0
        print(
            f"Latest existing date: {latest_date}. "
            f"Removed {removed_rows} rows. "
            f"No older rows kept, deleted existing file."
        )
    else:
        keep_df = keep_df.sort_values(DATE_COL)
        keep_df.to_csv(output_path, index=False)
        kept_rows = len(keep_df)

        print(
            f"Latest existing date: {latest_date}. "
            f"Removed {removed_rows} rows with {DATE_COL} >= {latest_date}. "
            f"Kept {kept_rows} rows."
        )

    return latest_date, kept_rows


def append_rows_to_csv(rows: list[dict], output_csv: str) -> int:
    """
    直接追加写入最终 CSV，不使用 part 文件。
    """
    if not rows:
        return 0

    output_path = Path(output_csv)
    df = pd.DataFrame(rows)

    if DATE_COL in df.columns:
        df[DATE_COL] = normalize_date_series(df[DATE_COL])
        df = df.dropna(subset=[DATE_COL])
        df = df.sort_values(DATE_COL)

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    df.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
    )

    return len(df)


def download_all_dividends(
    api_key: str,
    output_csv: str = "all_stock_dividends.csv",
    ex_dividend_date_gte: str | None = None,
    ex_dividend_date_lte: str | None = None,
    limit: int = 5000,
    sleep_sec: float = 0.5,
    checkpoint_every_pages: int = 20,
):
    """
    下载 Massive 全市场股票分红数据。

    特点：
    - 不使用 .part 文件
    - 数据直接落盘到 output_csv
    - 每次启动自动扫描已有 CSV
    - 从已有数据的最新 ex_dividend_date 重新下载
    - 自动实现断点恢复和增量更新

    注意：
    - 为了保证最新日期的数据完整，会删除已有文件中 latest_date 当天及之后的数据。
    - 然后重新从 latest_date 开始下载。
    """

    if not api_key:
        raise ValueError(
            "API key is empty. 请设置环境变量 MASSIVE_API_KEY。"
        )

    output_path = Path(output_csv)

    resume_date, existing_rows = prepare_existing_csv_for_resume(output_csv)

    effective_gte = ex_dividend_date_gte

    if resume_date:
        if effective_gte:
            effective_gte = max(effective_gte, resume_date)
        else:
            effective_gte = resume_date

    params = {
        "apiKey": api_key,
        "limit": min(limit, 5000),
        "sort": "ex_dividend_date.asc",
    }

    if effective_gte:
        params["ex_dividend_date.gte"] = effective_gte

    if ex_dividend_date_lte:
        params["ex_dividend_date.lte"] = ex_dividend_date_lte

    print("=" * 80)
    print("Starting download")
    print(f"Output CSV: {output_path.resolve()}")
    print(f"Existing rows kept: {existing_rows}")
    print(f"Download from {DATE_COL}.gte: {params.get('ex_dividend_date.gte')}")
    print(f"Download to   {DATE_COL}.lte: {params.get('ex_dividend_date.lte')}")
    print(f"Limit per page: {params['limit']}")
    print("=" * 80)

    session = create_retry_session()

    url = BASE_URL
    page = 0
    new_rows_total = 0

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
        new_rows_total += written_rows

        current_total = existing_rows + new_rows_total

        print(
            f"  rows this page: {len(rows)}, "
            f"written rows: {written_rows}, "
            f"new rows this run: {new_rows_total}, "
            f"estimated total rows: {current_total}"
        )

        url = data.get("next_url")

        if page % checkpoint_every_pages == 0:
            print(
                f"Checkpoint: page={page}, "
                f"new rows this run={new_rows_total}, "
                f"file={output_csv}"
            )

        time.sleep(sleep_sec)

    print("=" * 80)
    print("Download finished")
    print(f"New rows written this run: {new_rows_total}")
    print(f"Estimated total rows: {existing_rows + new_rows_total}")
    print(f"Saved to: {output_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":

    api_key_path = os.path.join(current_work_dir, "massive_key.json")
    API_KEY = load_api_key(api_key_path)

    download_all_dividends(
        api_key=API_KEY,
        output_csv=os.path.join(current_work_dir,'all_stock_dividends.csv'),
        ex_dividend_date_gte=None,
        ex_dividend_date_lte=None,
        limit=5000,
        sleep_sec=0.5,
        checkpoint_every_pages=20,
    )