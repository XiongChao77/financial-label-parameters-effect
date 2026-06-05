import boto3
from botocore.config import Config

# Initialize a session using your credentials
session = boto3.Session(
  aws_access_key_id='e42051b5-5d66-406f-8f26-b14f5fd5ee7d',
  aws_secret_access_key='Px7wGNLdw8bOmDIXm0jDDA0fznnrtcM6',
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

# List objects using the selected prefix
for page in paginator.paginate(Bucket='flatfiles', Prefix=prefix):
  for obj in page['Contents']:
    if 'day' in obj['Key']:
        print(obj['Key'])
        
        object_key = obj['Key']
        local_file_name = object_key.split('/')[-1]

        # This constructs the full local file path
        local_file_path = './massive/day/' + local_file_name

        # Download the file
        # Specify the bucket name
        bucket_name = 'flatfiles'
        s3.download_file(bucket_name, object_key, local_file_path)
