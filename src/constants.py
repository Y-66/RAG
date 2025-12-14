import os
from dotenv import load_dotenv

load_dotenv()  # 让 .env 内容生效

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

# 路径和向量库配置集中放在这里，方便主流程调用
DEFAULT_PDF_PATH = "src/public/python-tutorial.pdf"
CHROMA_PERSIST_DIR = "src/public/chroma_langchain_db"
CHROMA_COLLECTION = "example_collection"

os.environ["LANGSMITH_TRACING"] = "true"
