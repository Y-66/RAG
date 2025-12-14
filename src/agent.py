from langchain.agents import create_agent
from langchain.messages import HumanMessage

from model_factory import get_chat_model
from tools.retriever_tool import retrieve_context


def build_agent():
    llm = get_chat_model()
    return create_agent(
        llm,
        tools=[retrieve_context],
        system_prompt="你是一个智能的问答助手，你的知识库有关于Python 简介及数据分析概述的内容。"
        "如果用户有问相关的问题，你需要先调用工具获取相关内容，然后再给出答案。",
    )


if __name__ == "__main__":
    agent = build_agent()
    query = (
        "Python 语言的应用领域有哪些？"
    )

    for event in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()