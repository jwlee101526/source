from pyexpat import model

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_ibm import ChatWatsonx
import os

load_dotenv()

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


def process_pdf(file):
    if file is None:
        return "파일을 업로드해주세요.", "", "", ""

    loader = PyPDFLoader(file)
    pages = loader.load()

    # 총 페이지 수
    total_pages = len(pages)

    # 첫 페이지 내용
    first_page_content = pages[0].page_content if pages else ""

    # 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(pages)

    # 총 chunk 수, 첫 청크 내용, 첫 청크 Metadata
    total_chunks = len(chunks)
    first_chunk_content = chunks[0].page_content if chunks else ""
    first_chunk_metadata = str(chunks[0].metadata) if chunks else ""

    return (
        total_pages,
        first_page_content,
        total_chunks,
        first_chunk_content,
        first_chunk_metadata,
    )

def rag_qna(file, question):
    if file is None:
        return ("PDF 파일을 업로드 해주세요.", "")

    loader = PyPDFLoader(file)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    faiss_store = FAISS.from_documents(
        documents=split_docs, embedding=OllamaEmbeddings(model="nomic-embed-text-v2-moe")
    )

    retriever = faiss_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retriever_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in retriever_docs])

    message = """\
        당신은 PDF기반 RAG AI입니다.
        다음 문서를 참고해서 질문에 답변하세요.

        문서:
        {context}

        질문:
        {question}

    """
    rag_prompt = ChatPromptTemplate.from_template(message)

    chain = rag_prompt | watson_llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": question})

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
        with gr.Tab("1. PDF & Chunk 확인"):
            input_file = gr.File(label="PDF 파일 업로드", file_types=[".pdf"])
            btn_start = gr.Button("분석 시작")

            output_pages = gr.Textbox(label="총 페이지 수", lines=1)
            output_first_page = gr.Textbox(label="첫 페이지 내용", lines=5)
            output_chunk_count = gr.Textbox(label="총 청크 수")
            output_chunk_content = gr.Textbox(label="첫 번째 청크 내용", lines=5)
            output_chunk_meta = gr.Textbox(label="첫 번째 청크 metadata")

        with gr.Tab("2. RAG QA"):
            input_file_2 = gr.File(label="PDF 파일 업로드", file_types=[".pdf"])
            input_question_2 = gr.Textbox(label="질문 입력", lines=1)
            btn_start_2 = gr.Button("질문하기")

            output_retrieved = gr.Textbox(label="검색된 Chunk")
            output_answer = gr.Textbox(label="최종 답변")

    btn_start.click(
        fn=process_pdf,
        inputs=[input_file],
        outputs=[output_pages, output_first_page, output_chunk_count, output_chunk_content, output_chunk_meta],
    )

    btn_start_2.click(
        fn=rag_qna,
        inputs=[input_file_2, input_question_2],
        outputs=[output_retrieved, output_answer],
    )

if __name__ == "__main__":
    app.launch()
