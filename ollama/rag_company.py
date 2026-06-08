import os
from pathlib import Path
import pickle
from rich import print

import gradio as gr
from langchain_classic.document_loaders import TextLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.retrievers.self_query.chroma import ChromaTranslator
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

TITLE = "사내 문서 RAG"
PROJECT_NAME = "rag_company"

LOADERS = {
    ".pdf" : PyMuPDFLoader,
    ".docx" : UnstructuredWordDocumentLoader,
    ".xlsx" : UnstructuredExcelLoader,
    ".txt" : TextLoader
}

CHROMA_DIR = f"./db/{PROJECT_NAME}"
COLLECTION_NAME = "rag_company"
CHUNK_PATH = "./db/chunks.pk1"

DOCUMENTS = []
CHUNKS = []
VECTORSTORE = None

def initialize():
    global VECTORSTORE

    VECTORSTORE = get_vectorstore()

8
def get_watson_embedding():
    import os
    from dotenv import load_dotenv
    from langchain_ibm import WatsonxEmbeddings

    load_dotenv()

    apikey = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    watson_ai_url = os.getenv("WATSONX_URL")

    watson_embedding = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url=f"{watson_ai_url}",
        api_key=f"{apikey}",
        project_id=f"{project_id}",
        params={"temperature": 0},
    )

    return watson_embedding

def get_watson_llm():
    import os
    from dotenv import load_dotenv
    from langchain_ibm import ChatWatsonx

    load_dotenv()

    apikey = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    watson_ai_url = os.getenv("WATSONX_URL")

    watson_llm = ChatWatsonx(
        model_id="ibm/granite-4-h-small",
        url=f"{watson_ai_url}",
        api_key=f"{apikey}",
        project_id=f"{project_id}",
        max_tokens=2000,
        params={"temperature": 0},
    )

    return watson_llm

# ======
# Tab 1 - 기능 구현
# ======

def extract_metadata(file_path):
    """파일명에서 메타데이터를 추출한다. (예: 2026 상 삼성전자 ...)"""
    path = Path(file_path)
    parts = path.stem.split()

    return {
        "year": parts[0] if len(parts) > 0 else "",
        "recruitment_period": parts[1] if len(parts) > 1 else "",
        "company": parts[2] if len(parts) > 2 else "",
        "file_name": path.name,
    }

def upload_files(files):
    DOCUMENTS.clear()

    if not files:
        return "업로드된 파일이 없습니다."

    skipped = []
    for file in files:
        ext = Path(file).suffix.lower()
        loader_cls = LOADERS.get(ext)
        if loader_cls is None:
            skipped.append(Path(file).name)
            continue

        docs = loader_cls(file).load()
        meta_info = extract_metadata(file)

        for doc in docs:
            doc.metadata.update(meta_info)
            DOCUMENTS.append(doc)

    msg = f"문서 수 : {len(DOCUMENTS)}  로드 완료 (파일 {len(files)}개)"
    if skipped:
        msg += f"\n지원하지 않는 파일: {', '.join(skipped)}"

    return msg


def preview_chunks():
    global CHUNKS

    if not DOCUMENTS:
        return "먼저 문서를 업로드하세요."

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    CHUNKS = splitter.split_documents(DOCUMENTS)

    preview = []
    for i, chunk in enumerate(CHUNKS[:10]):
        preview.append(f"[CHUNK {i+1}]\n {chunk.page_content[:100]}")

    return "\n\n".join(preview)


def create_db():
    global VECTORSTORE
    global CHUNKS

    if not CHUNKS:
        return "먼저 Chunk를 생성하세요."

    embedding = get_watson_embedding()

    existing = VECTORSTORE or Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
    )
    existing.delete_collection()

    VECTORSTORE = Chroma.from_documents(
        documents=CHUNKS,
        embedding=embedding,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    # BM25 검색용으로 청크를 디스크에 저장한다.
    save_chunks(CHUNKS)

    return f"""
    Vector DB 생성 완료 (청크 {len(CHUNKS)}개)

    CHUNK:
    {len(CHUNKS)}

    Vector:
    {VECTORSTORE._collection.count()}
    """

# ======
# Tab 2 - 검색 테스트
# ======
def save_chunks(chunks):
    """BM25 인덱스 재생성을 위해 청크를 CHUNK_PATH 에 저장한다."""
    os.makedirs(os.path.dirname(CHUNK_PATH), exist_ok=True)
    with open(CHUNK_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks():
    """저장된 청크를 불러온다. 없으면 빈 리스트."""
    if not Path(CHUNK_PATH).exists():
        return []
    with open(CHUNK_PATH, "rb") as f:
        return pickle.load(f)


def format_docs(docs):
    """검색된 Document 리스트를 보기 좋은 문자열로 변환한다."""
    if not docs:
        return "검색 결과가 없습니다."

    result = []
    for i, doc in enumerate(docs[:3], 1):
        meta = doc.metadata
        result.append(
            f"[문서 {i}] {meta.get('file_name', '-')}\n"
            f"회사 : {meta.get('company', '-')} | "
            f"연도 : {meta.get('year', '-')} | "
            f"시기 : {meta.get('recruitment_period', '-')}\n"
            f"{doc.page_content.strip()[:300]}"
        )

    return "\n\n".join(result)


def get_vectorstore():
    """메모리에 있으면 그대로, 없으면 영속화된 Chroma 를 불러온다."""
    global VECTORSTORE

    if VECTORSTORE is None:
        VECTORSTORE = Chroma(
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=get_watson_embedding(),
        )

    return VECTORSTORE


def get_rerank_retriever(vectorstore):
    """BM25 + Dense 앙상블 결과를 Cohere 로 재정렬하는 검색기를 만든다."""
    chunks = load_chunks()

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    if chunks:
        bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, base_retriever],
            weights=[0.35, 0.65],
        )
    else:
        ensemble = base_retriever

    reranker = CohereRerank(model="rerank-v4.0-pro", top_n=5)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )


def search(query):
    if not query or not query.strip():
        empty = "검색어를 입력하세요."
        return empty, empty, empty, empty

    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        msg = "먼저 Vector DB 를 생성하세요."
        return msg, msg, msg, msg

    # 1) Dense (유사도 검색)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    dense_result = format_docs(dense_retriever.invoke(query))

    # 2) BM25 (키워드 검색)
    chunks = load_chunks()
    if chunks:
        bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
        bm25_result = format_docs(bm25_retriever.invoke(query))
    else:
        bm25_result = "저장된 청크가 없습니다. 먼저 Vector DB 를 생성하세요."

    # 3) SelfQuery
    metadata_field_info = [
        AttributeInfo(name="year", description="채용 연도 (예: 2026)", type="string"),
        AttributeInfo(name="recruitment_period", description="채용 시기 (예: 상, 하)", type="string"),
        AttributeInfo(name="company", description="회사명 (예: 삼성전자)", type="string"),
    ]
    self_query_retriever = SelfQueryRetriever.from_llm(
        llm=get_watson_llm(),
        vectorstore=vectorstore,
        document_contents="계열사 직무기술서 문서",
        metadata_field_info=metadata_field_info,
        structured_query_translator=ChromaTranslator(),
        search_kwargs={"k": 5}
    )
    self_query_result = format_docs(self_query_retriever.invoke(query))

    # 4) Ensemble(BM25 + Dense) -> Rerank
    rerank_retriever = get_rerank_retriever(vectorstore)
    rerank_result = format_docs(rerank_retriever.invoke(query))

    return bm25_result, dense_result, rerank_result, self_query_result

# ======
# Tab 3 - RAG 채팅
# ======
RAG_SYSTEM_PROMPT = """당신은 사내 문서를 기반으로 질문에 답변하는 어시스턴트입니다.
아래 [참고 문서] 내용과 이전 대화 맥락을 근거로 한국어로 명확하고 구체적으로 답변하세요.
참고 문서에 없는 내용은 추측하지 말고 "제공된 문서에서 찾을 수 없습니다." 라고 답하세요.
답변 마지막에 참고한 내부 백터 DB의 문서명을 명시하세요.

[참고 문서]
{context}"""

def chat(message, history):
    """이전 대화(history)를 맥락으로 rerank 검색 후 LLM 응답을 토큰 단위로 스트리밍한다."""
    history = history or []

    if not message or not message.strip():
        yield "", history
        return

    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "먼저 Vector DB 를 생성하세요."},
        ]
        yield "", history
        return

    # 1) rerank 검색기로 관련 문서를 가져온다.
    retriever = get_rerank_retriever(vectorstore)
    docs = retriever.invoke(message)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 2) 시스템 프롬프트 + 이전 대화 + 이번 질문으로 프롬프트를 구성한다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ])
    chain = prompt | get_watson_llm()

    # 3) LLM 응답을 토큰 단위로 스트리밍한다.
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield "", history

    answer = ""
    # 방금 추가한 user/assistant(빈 답변) 2건을 제외한 과거 대화만 맥락으로 넘긴다.
    for chunk in chain.stream({"context": context, "query": message, "history": history[:-2]}):
        answer += chunk.content
        history[-1]["content"] = answer
        yield "", history

    # 답변 근거가 된 DB 문서명을 코드로 덧붙인다. (LLM 지시 미준수 대비)
    sources = sorted({doc.metadata.get("file_name", "-") for doc in docs})
    if sources:
        history[-1]["content"] = answer + "\n\n---\n 참고 문서: " + ", ".join(sources)
        yield "", history

# ======
# Gradio UI
# ======
with gr.Blocks(title=TITLE) as app:
    gr.Markdown(f"# {TITLE}")

    with gr.Tabs(selected="tab_manage"):
        with gr.Tab("문서 관리", id="tab_manage"):
            input_tab1_upload_files = gr.File(label="문서 업로드", file_count="multiple")
            btn_tab1_upload = gr.Button("문서 업로드", variant="primary")
            output_tab1_upload = gr.Textbox(label="업로드 결과", interactive=False)

            btn_tab1_chunk = gr.Button("Chunk 확인")
            output_tab1_chunk = gr.Textbox(label="Chunk 미리보기", lines=5, interactive=False)

            btn_tab1_create_db = gr.Button("Vector DB 생성", variant="primary")
            output_tab1_create_db = gr.Textbox(label="DB 생성 결과", interactive=False)

        with gr.Tab("검색 테스트"):
            input_tab2_search = gr.Textbox(label="검색어", placeholder="검색할 내용을 입력하세요")
            btn_tab2_search = gr.Button("검색", variant="primary")
            with gr.Blocks():
                output_tab2_bm25 = gr.Textbox(label="BM25", interactive=False)
                output_tab2_dense = gr.Textbox(label="Dense", interactive=False)
                output_tab2_rerank = gr.Textbox(label="Rerank", interactive=False)
                output_tab2_selfquery = gr.Textbox(label="SelfQuery", interactive=False)

        with gr.Tab("RAG 채팅"):
            chatbot_tab3 = gr.Chatbot(label="대화")
            input_tab3_chat = gr.Textbox(label="질문", placeholder="질문을 입력하세요")
            btn_tab3_chat = gr.Button("전송", variant="primary")
    
    btn_tab1_upload.click(fn=upload_files, inputs=input_tab1_upload_files, outputs=output_tab1_upload)
    btn_tab1_chunk.click(fn=preview_chunks, inputs=None, outputs=output_tab1_chunk)
    btn_tab1_create_db.click(fn=create_db, inputs=None, outputs=output_tab1_create_db)
    btn_tab2_search.click(fn=search, inputs=input_tab2_search, outputs=[output_tab2_bm25, output_tab2_dense, output_tab2_rerank, output_tab2_selfquery])
    btn_tab3_chat.click(fn=chat, inputs=[input_tab3_chat, chatbot_tab3], outputs=[input_tab3_chat, chatbot_tab3])
    input_tab3_chat.submit(fn=chat, inputs=[input_tab3_chat, chatbot_tab3], outputs=[input_tab3_chat, chatbot_tab3])

if __name__ == "__main__":
    initialize()
    app.launch()
