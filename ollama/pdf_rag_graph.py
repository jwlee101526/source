from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import gradio as gr
from langchain_chroma import Chroma
from langchain_ibm import ChatWatsonx, WatsonxEmbeddings
import os

from langgraph.graph import StateGraph, START, END
from typing import List

from pydantic import BaseModel, Field


load_dotenv()

CHROMA_DIR = "./db/chroma_db"
COLLECTION_NAME = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Models, Embeddings
watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=os.getenv("WATSONX_URL"),
    api_key=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    max_tokens=2000,
    params = {
        "temperature": 0
    }
)

watson_embedding = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url=os.getenv("WATSONX_URL"),
    api_key=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
)

# 1. State 정의
class RagState(BaseModel):
    query: str = ""
    retrieve_docs: List[Document] = Field(default_factory=list)
    answer: str = ""

def retrieve(state: RagState):
    vectorstore = get_chroma_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    docs = retriever.invoke(state.query)
    return {"retrieve_docs": docs}

def generate(state: RagState):
    if not state.retrieve_docs:
        return {"answer": "검색된 문서가 없습니다. 먼저 PDF를 분석해서 Chroma DB를 생성하세요."}

    context = "\n\n".join(doc.page_content for doc in state.retrieve_docs)

    message = """\
        당신은 PDF기반 RAG AI입니다.
        다음 문서를 참고해서 질문에 답변하세요.
        문서에 없는 내용은 모른다고 답하세요.

        문서:
        {context}

        질문:
        {question}

    """
    rag_prompt = ChatPromptTemplate.from_template(message)
    chain = rag_prompt | watson_llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": state.query})
    return {"answer": answer}


builder = StateGraph(RagState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


def get_file_path(file):
    return getattr(file, "name", file)


def load_and_split_pdf(file):
    loader = PyMuPDFLoader(get_file_path(file))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    return pages, chunks


def create_chroma_vectorstore(chunks):
    existing = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=watson_embedding,
        persist_directory=CHROMA_DIR,
    )
    existing.delete_collection()

    return Chroma.from_documents(
        documents=chunks,
        embedding=watson_embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )


def get_chroma_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=watson_embedding,
        persist_directory=CHROMA_DIR,
    )


def process_pdf(file):
    if file is None:
        return "파일을 업로드해주세요."

    pages, chunks = load_and_split_pdf(file)

    # 총 페이지 수
    total_pages = len(pages)

    # 첫 페이지 내용
    first_page_content = pages[0].page_content if pages else ""

    total_chunks = len(chunks)
    first_chunk_content = chunks[0].page_content if chunks else ""
    first_chunk_metadata = str(chunks[0].metadata) if chunks else ""

    vectorstore = create_chroma_vectorstore(chunks)

    return f"""총 페이지 수: {total_pages}

총 청크 수: {total_chunks}

Chroma 저장 문서 수: {vectorstore._collection.count()}

첫 페이지 내용:
{first_page_content}

첫 번째 청크 내용:
{first_chunk_content}

첫 번째 청크 metadata:
{first_chunk_metadata}
"""

def rag_qna(file, question):
    '''
    invoke() => result['answer'] 리턴
    '''


    if file is None:
        return ("PDF 파일을 업로드 해주세요.", "")

    if not question:
        return ("질문을 입력해주세요.", "")

    _, split_docs = load_and_split_pdf(file)
    create_chroma_vectorstore(split_docs)
    result = graph.invoke({"query": question})
    retriever_docs = result["retrieve_docs"]
    answer = result["answer"]

    retrieved_text = ""
    for i, doc in enumerate(retriever_docs):
        retrieved_text += f"""
        [검색 문서 {i+1}]

        내용:
        {doc.page_content}

        메타데이터:
        {doc.metadata}
        {'='*50}
        """

    return retrieved_text, answer


# ============
# Gradio
# ============
with gr.Blocks() as app:
    gr.Markdown("## PDF RAG 학습 앱")

    with gr.Tabs():
        with gr.Tab("LCEL RAG -> LangGraph RAG 변환"):
            input_file = gr.File(label="PDF 파일 업로드", file_types=[".pdf"])
            btn_start = gr.Button("분석 시작")

            output = gr.Textbox(label="처리 결과", lines=15)

        with gr.Tab("2. RAG QA"):
            input_file_2 = gr.File(label="PDF 파일 업로드", file_types=[".pdf"])
            input_question_2 = gr.Textbox(label="질문 입력", lines=1)
            btn_start_2 = gr.Button("질문하기")

            output_retrieved = gr.Textbox(label="검색된 Chunk")
            output_answer = gr.Textbox(label="최종 답변")

    btn_start.click(
        fn=process_pdf,
        inputs=[input_file],
        outputs=[output],
    )

    btn_start_2.click(
        fn=rag_qna,
        inputs=[input_file_2, input_question_2],
        outputs=[output_retrieved, output_answer],
    )

if __name__ == "__main__":
    app.launch()
