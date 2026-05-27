from langchain_ollama import ChatOllama
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import json
import gradio as gr

load_dotenv()


MODELS = ["qwen", "exaone", "watsonx"]


class NewsInfo(BaseModel):
    title: str = Field(description="뉴스 기사의 제목")
    date: str = Field(description="뉴스 기사의 작성일 (YYYY-MM-DD 형식)")
    keywords: list = Field(description="뉴스 기사의 핵심 키워드 리스트")
    category: str = Field(description="뉴스 기사의 카테고리")


def config_llm(model_name, temperature):
    if model_name == "qwen":
        return ChatOllama(model="qwen3.5:4b", temperature=temperature)
    if model_name == "exaone":
        return ChatOllama(model="exaone3.5:2.4b", temperature=temperature)
    if model_name == "watsonx":
        return ChatWatsonx(
            model_id="ibm/granite-4-h-small",
            url=os.environ["WATSONX_URL"],
            apikey=os.environ["WATSONX_API_KEY"],
            project_id=os.environ["WATSONX_PROJECT_ID"],
            params={"temperature": temperature, "max_tokens": 2000},
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_chain(model_name, temperature):
    parser = JsonOutputParser(pydantic_object=NewsInfo)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 뉴스 기사에서 핵심 정보를 정확하게 추출하는 전문 분석가입니다. "
                "주어진 뉴스 텍스트에서 title, date, keywords, category를 JSON 형식으로 추출하세요.\n"
                "{format_instructions}",
            ),
            ("human", "다음 뉴스 기사를 분석해 주세요:\n\n{news}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | config_llm(model_name, temperature) | parser


def extract_news(news_text, model_name, temperature, progress=gr.Progress()):
    print(f"[extract_news] called: model={model_name}, temp={temperature}, len(text)={len(news_text)}")
    articles = [a.strip() for a in news_text.split("---") if a.strip()]
    if not articles:
        return "뉴스 텍스트를 입력해 주세요."
    if not model_name:
        return "모델을 선택해 주세요."

    progress(0, desc=f"{len(articles)}개 기사 batch 처리 중...")
    print(f"[extract_news] {len(articles)}개 기사 batch 시작")
    chain = build_chain(model_name, temperature)

    try:
        results = chain.batch([{"news": a} for a in articles])
        print(f"[extract_news] batch 완료: {len(results)}건")
    except Exception as e:
        print(f"[extract_news] 예외: {type(e).__name__}: {e}")
        return f"**처리 중 오류 발생:**\n```\n{type(e).__name__}: {e}\n```"

    output = []
    for i, (article, result) in enumerate(zip(articles, results), start=1):
        output.append(f"### 기사 {i}")
        output.append(f"**원문 일부:** {article[:60]}...")
        output.append("```json")
        output.append(json.dumps(result, ensure_ascii=False, indent=2))
        output.append("```")
    return "\n".join(output)


with gr.Blocks(title="news extractor") as app:
    gr.Markdown("## News Information Extractor")
    gr.Markdown("뉴스 기사에서 title, date, keywords, category를 JSON으로 추출합니다. "
                "기사는 `---` 로 구분하여 3개 이상 입력하세요.")

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### 모델 선택")
            model_select = gr.Radio(choices=MODELS, label="LLM", value="qwen")

            gr.Markdown("### 파라미터")
            temperature = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                label="Temperature",
            )

            run_btn = gr.Button("추출 실행", variant="primary")

        with gr.Column(scale=3):
            news_input = gr.Textbox(
                label="뉴스 텍스트 (--- 로 기사 구분)",
                lines=15,
                placeholder="뉴스 기사 1\n---\n뉴스 기사 2\n---\n뉴스 기사 3",
            )
            result_output = gr.Markdown(label="추출 결과")

    run_btn.click(
        extract_news,
        inputs=[news_input, model_select, temperature],
        outputs=[result_output],
    )


if __name__ == "__main__":
    app.queue().launch(debug=True)