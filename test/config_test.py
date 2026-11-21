import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.config import settings

config = settings()
print(f"AWS region: {config.aws_region}")
print(
    "AWS credentials configured: "
    f"{bool(config.aws_access_key_id and config.aws_secret_access_key)}"
)
print(f"S3 bucket configured: {bool(config.s3_bucket)}")
print(f"Database URL configured: {bool(config.database_url)}")
print(f"Bedrock embedding model: {config.bedrock_embed_model}")
print(f"Bedrock chat model: {config.bedrock_chat_model}")