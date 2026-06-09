import os
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ibm import ChatWatsonx, WatsonxEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


load_dotenv()


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=_required_env("WATSONX_URL"),
    api_key=_required_env("WATSONX_API_KEY"),
    project_id=_required_env("WATSONX_PROJECT_ID"),
    max_tokens=2000,
    temperature=0,
)

watson_embedding = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url=_required_env("WATSONX_URL"),
    api_key=_required_env("WATSONX_API_KEY"),
    project_id=_required_env("WATSONX_PROJECT_ID"),
    params={"temperature": 0},
)


class SelfRAGState(BaseModel):
    query: str = ""
    retrieved_docs: List[Document] = Field(default_factory=list)
    answer: str = ""
    evaluation: str = ""
    retry_count: int = 0


def retrieve(state: SelfRAGState):
    vectorstore = Chroma(
        collection_name="research",
        embedding_function=watson_embedding,
        persist_directory="./db/chroma_db/",
    )
    docs = vectorstore.similarity_search(state.query, k=3)

    return {"retrieved_docs": docs}


def generate(state: SelfRAGState):
    context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)

    prompt = """\
다음 컨텍스트를 참고하여 질문에 답하세요.
컨텍스트에 없는 내용은 모른다고 답하세요.
컨텍스트에 정보가 부족하면 그 사실을 명시하세요.

컨텍스트:
{context}

질문:
{query}
"""

    response = watson_llm.invoke(prompt.format(context=context, query=state.query))

    return {"answer": response.content}


def evaluate(state: SelfRAGState):
    """생성한 답변의 충실도와 관련성을 평가한다."""

    context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)

    prompt = ChatPromptTemplate.from_template(
        """
질문:
{query}

컨텍스트:
{context}

답변:
{answer}

반드시 아래 둘 중 하나로만 출력하세요
'sufficient', 'insufficient'
"""
    )

    response = watson_llm.invoke(
        prompt.format(query=state.query, answer=state.answer, context=context)
    )

    content = response.content.lower().strip()
    evaluation = "insufficient" if content.startswith("insufficient") else "sufficient"

    return {"evaluation": evaluation}


def rewrite_query(state: SelfRAGState):
    """평가 결과를 반영하여 검색 질문을 개선한다."""

    prompt = """\
원래 질문 : {query}
검색 결과가 충분하지 않았습니다.
더 구체적이고 검색하기 좋은 질문으로 재작성하세요.
질문만 출력하세요.
"""

    response = watson_llm.invoke(prompt.format(query=state.query))

    return {"query": response.content, "retry_count": state.retry_count + 1}


def route_after_eval(state: SelfRAGState):
    """재검색 여부를 결정한다."""

    if state.evaluation == "sufficient" or state.retry_count >= 2:
        return "done"
    return "retry"


builder = StateGraph(SelfRAGState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("evaluate", evaluate)
builder.add_node("rewrite", rewrite_query)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges(
    "evaluate", route_after_eval, {"done": END, "retry": "rewrite"}
)
builder.add_edge("rewrite", "retrieve")

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"query": "where can i use chatGPT?", "retry_count": 0})
    print(result["answer"])
    print(graph.get_graph().draw_ascii())
