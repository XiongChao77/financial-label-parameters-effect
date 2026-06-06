import boto3
from botocore.config import Config
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
    try:
        API_KEY = data['api_key']
        ACCESS_KEY_ID = data['access_key_id']
        return API_KEY,ACCESS_KEY_ID
    except:
        raise ValueError(
            f"No API key found in {key_file}. "
            f"Expected one of: {possible_keys}"
        )

api_key_path = os.path.join(current_work_dir, "massive_key.json")
API_KEY,ACCESS_KEY_ID = load_api_key(api_key_path)

# Initialize a session using your credentials
session = boto3.Session(
  aws_access_key_id='e42051b5-5d66-406f-8f26-b14f5fd5ee7d',
  aws_secret_access_key=API_KEY,
)

# Create a client with your session and specify the endpoint
s3 = session.client(
  's3',
  endpoint_url='https://files.massive.com',
  config=Config(signature_version='s3v4'),
)

# List Example
# Initialize a paginator for listing objects
paginator = s3.get_paginator('list_objects_v2')

# Choose the appropriate prefix depending on the data you need:
# - 'global_crypto' for global cryptocurrency data
# - 'global_forex' for global forex data
# - 'us_indices' for US indices data
# - 'us_options_opra' for US options (OPRA) data
# - 'us_stocks_sip' for US stocks (SIP) data
prefix = 'us_stocks_sip'  # Example: Change this prefix to match your data need
period = 'minute' # day/minute /trades/quotes
directory = os.path.join(current_work_dir,'massive',period)
os.makedirs(directory,exist_ok=True)
# List objects using the selected prefix
for page in paginator.paginate(Bucket='flatfiles', Prefix=prefix):
  for obj in page['Contents']:
    if period in obj['Key']:# and '2026':
        print(str(time.time())+" "+ obj['Key'])
        
        object_key = obj['Key']
        local_file_name = object_key.split('/')[-1]
        year = local_file_name.split('-')[0]
        os.makedirs(os.path.join(directory,year),exist_ok=True)

        # This constructs the full local file path

        local_file_path = os.path.join(directory,year,local_file_name )

        # Download the file
        # Specify the bucket name
        bucket_name = 'flatfiles'
        if not os.path.exists(local_file_path ):
          s3.download_file(bucket_name, object_key, local_file_path)
