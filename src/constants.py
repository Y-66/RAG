import os
from dotenv import load_dotenv

load_dotenv()  # 让 .env 内容生效

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

os.environ["LANGSMITH_TRACING"] = "true"
