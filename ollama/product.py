from langchain_ollama import ChatOllama
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
import time
import gradio as gr

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "product-review-analyzer"


MODELS = ["qwen", "exaone", "watsonx", "gemma"]
MAX_REVIEWS = 10


class ReviewAnalysis(BaseModel):
    sentiment: Literal["긍정", "부정", "중립"] = Field(description="리뷰 감성")
    score: float = Field(description="감성 점수. 반드시 0.0(매우 부정) ~ 1.0(매우 긍정) 사이의 소수값만 허용")
    pros: list[str] = Field(description="장점 목록")
    cons: list[str] = Field(description="단점 목록")
    recommend: bool = Field(description="추천 여부")
    reply: str = Field(description="고객에게 보낼 답변")


def config_llm(model_name, temperature):
    if model_name == "qwen":
        return ChatOllama(model="qwen3.5:4b", temperature=temperature)
    if model_name == "exaone":
        return ChatOllama(model="exaone3.5:2.4b", temperature=temperature)
    if model_name == "gemma":
        return ChatOllama(model="gemma4:e2b", temperature=temperature)
    if model_name == "watsonx":
        return ChatWatsonx(
            model_id="ibm/granite-4-h-small",
            url=os.environ["WATSONX_URL"],
            apikey=os.environ["WATSONX_API_KEY"],
            project_id=os.environ["WATSONX_PROJECT_ID"],
            params={"temperature": temperature, "max_tokens": 2000},
        )
    raise ValueError(f"Unknown model: {model_name}")


def analyze_reviews(*args):
    *reviews, model_name, temperature = args
    valid = [(i, r) for i, r in enumerate(reviews) if r and r.strip()]
    if not valid:
        return "리뷰를 입력하세요.", None

    parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 전문 상품 리뷰 분석가입니다. 주어진 리뷰를 분석하여 "
                "감성, 점수, 장점, 단점, 추천 여부와 고객 답변을 작성하세요.\n"
                "{format_instructions}",
            ),
            ("human", "리뷰: {review}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | config_llm(model_name, temperature) | parser

    inputs = [{"review": r} for _, r in valid]

    start = time.time()
    results = chain.batch(inputs)
    elapsed = time.time() - start

    output = {f"리뷰 {idx + 1}": result.model_dump() for (idx, _), result in zip(valid, results)}
    return f"{elapsed:.2f}초 ({model_name}) — {len(valid)}개 리뷰 처리", output

# ====================
# Gradio
# ====================

def add_review_box(count):
    new_count = min(count + 1, MAX_REVIEWS)
    updates = [gr.update(visible=(i < new_count)) for i in range(MAX_REVIEWS)]
    return [new_count] + updates


def remove_review_box(count):
    new_count = max(count - 1, 1)
    updates = [gr.update(visible=(i < new_count)) for i in range(MAX_REVIEWS)]
    return [new_count] + updates


with gr.Blocks(title="Product Review Analyzer") as app:
    gr.Markdown("## Product Review Analyzer")
    review_count = gr.State(1)

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### 모델 선택")
            model_select = gr.Radio(choices=MODELS, label="LLM", value="qwen")

            gr.Markdown("### 파라미터")
            temperature = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                label="Temperature",
            )

        with gr.Column(scale=3):
            gr.Markdown("### 리뷰 입력")
            review_boxes = []
            for i in range(MAX_REVIEWS):
                box = gr.Textbox(
                    placeholder=f"리뷰 {i + 1} 입력 (예: 배송이 빠르고 품질도 좋았어요.)",
                    label=f"리뷰 {i + 1}",
                    lines=3,
                    visible=(i == 0),
                )
                review_boxes.append(box)

            with gr.Row():
                add_btn = gr.Button("+ 항목 추가", variant="secondary", scale=1)
                remove_btn = gr.Button("- 항목 제거", variant="secondary", scale=1)
                analyze_btn = gr.Button("일괄 분석", variant="primary", scale=2)

            speed_output = gr.Markdown()
            result_output = gr.JSON(label="분석 결과")

    add_btn.click(
        add_review_box,
        inputs=[review_count],
        outputs=[review_count] + review_boxes,
    )

    remove_btn.click(
        remove_review_box,
        inputs=[review_count],
        outputs=[review_count] + review_boxes,
    )

    analyze_btn.click(
        analyze_reviews,
        inputs=review_boxes + [model_select, temperature],
        outputs=[speed_output, result_output],
    )


if __name__ == "__main__":
    app.launch(debug=True)
