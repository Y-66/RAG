
from langchain import rate_limiters
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai import ChatOpenAI


from constants import OPENAI_API_KEY, ZHIPUAI_API_KEY

llm_openai = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)


my_chat_model = init_chat_model( # v1.0 新写法
    model="gpt-4o",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    temperature=0.5,
    timeout=10,
    max_tokens=1000,
    # rate_limiters=...
)

callback = UsageMetadataCallbackHandler()

# ------------------------------------ Test ------------------------------------------------
if __name__ == "__main__":
    """
    Log 概率是 LLM 的“内心评分卡”，它提供了一种量化的方法来衡量模型对其输出中每个词的把握程度，
    是构建高可靠性 AI 应用的关键元数据。
    """
    response = llm_openai.invoke("Why do parrots talk?", config={"callbacks": [callback]})
    print(response.response_metadata["logprobs"])
    print(callback.usage_metadata)