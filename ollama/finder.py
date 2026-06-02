import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader

# loader : csv
# vectorstore : memory

TITLE = "Educate Kids"
DESCRIPTION = "입력한 단어와 비슷한 단어 또는 문장을 찾아드립니다."
DB_NAME = "finder_db"
CSV_PATH = "./data/myData.csv"

global vectorstore

def initialize():
    global vectorstore

    csv_loader = CSVLoader(
        CSV_PATH,
        encoding="utf-8",
        csv_args={"delimiter": ",", "fieldnames": ["WORDS"]},
    )
    documents = csv_loader.load()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_watson_embedding(),
        persist_directory=f"./db/{DB_NAME}",
        collection_name=DB_NAME,
    )

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


def find_similar(query: str):
    if not query or not query.strip():
        return "", ""

    results = vectorstore.similarity_search_with_score(query.strip(), k=2)

    outputs = []
    for doc, score in results:
        outputs.append(f"{doc.page_content} (거리: {score:.4f})")

    while len(outputs) < 2:
        outputs.append("")

    return outputs[0], outputs[1]


with gr.Blocks(theme=gr.themes.Soft(), title="Educate Kids") as app:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            input_query = gr.Textbox(
                label="질문",
                placeholder="단어 또는 문장을 입력하세요",
                lines=2,
            )
            btn_search = gr.Button("유사한 단어 찾기", variant="primary")

        with gr.Column(scale=1):
            result1 = gr.Textbox(label="Word 1", interactive=False)
            result2 = gr.Textbox(label="Word 2", interactive=False)

    btn_search.click(fn=find_similar, inputs=input_query, outputs=[result1, result2])
    input_query.submit(fn=find_similar, inputs=input_query, outputs=[result1, result2])


if __name__ == "__main__":
    initialize()
    app.launch()
