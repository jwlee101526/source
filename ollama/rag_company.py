import os
from pathlib import Path
import pickle
from rich import print

import gradio as gr
from langchain_classic.document_loaders import TextLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    pass

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

    return f"""
    Vector DB 생성 완료 (청크 {len(CHUNKS)}개)

    CHUNK:
    {len(CHUNKS)}

    Vector:
    {VECTORSTORE._collection.count()}
    """


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
            output_tab2_search = gr.Textbox(label="검색 결과", lines=10, interactive=False)

        with gr.Tab("RAG 채팅"):
            chatbot_tab3 = gr.Chatbot(label="대화")
            input_tab3_chat = gr.Textbox(label="질문", placeholder="질문을 입력하세요")
            btn_tab3_chat = gr.Button("전송", variant="primary")
    
    btn_tab1_upload.click(fn=upload_files, inputs=input_tab1_upload_files, outputs=output_tab1_upload)
    btn_tab1_chunk.click(fn=preview_chunks, inputs=None, outputs=output_tab1_chunk)
    btn_tab1_create_db.click(fn=create_db, inputs=None, outputs=output_tab1_create_db)
    # btn_tab2_search.click(fn=search, inputs=input_tab2_search, outputs=output_tab2_search)
    # btn_tab3_chat.click(fn=chat, inputs=[input_tab3_chat, chatbot_tab3], outputs=[input_tab3_chat, chatbot_tab3])

if __name__ == "__main__":
    app.launch()
