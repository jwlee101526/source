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


MODELS = ["qwen", "exaone", "watsonx"]


class ReviewAnalysis(BaseModel):
    sentiment: Literal["긍정", "부정", "중립"] = Field(description="리뷰 감성")
    score: float = Field(description="감성 점수 (0.0 ~ 1.0)", ge=0.0, le=1.0)
    pros: list[str] = Field(description="장점 목록")
    cons: list[str] = Field(description="단점 목록")
    recommend: bool = Field(description="추천 여부")
    reply: str = Field(description="고객에게 보낼 답변")


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


def analyze_review(review, model_name, temperature):
    if not review.strip():
        return {}, "리뷰를 입력하세요."

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

    start = time.time()
    result = chain.invoke({"review": review})
    elapsed = time.time() - start

    return result.model_dump(), f"⏱️ {elapsed:.2f}초 ({model_name})"


with gr.Blocks(title="product review analyzer") as app:
    gr.Markdown("## Product Review Analyzer")

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### 모델 선택")
            model_select = gr.Radio(
                choices=MODELS,
                label="LLM",
                value="qwen",
            )

            gr.Markdown("### 파라미터")
            temperature = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                label="Temperature",
            )

        with gr.Column(scale=3):
            review_input = gr.Textbox(
                placeholder="(예: 배송이 빠르고 품질도 좋았어요. 다만 색상이 사진과 조금 달라요.)",
                label="상품 리뷰",
                lines=5,
            )
            analyze_btn = gr.Button("분석", variant="primary")
            result_output = gr.JSON(label="분석 결과")
            speed_output = gr.Markdown()

    analyze_btn.click(
        analyze_review,
        inputs=[review_input, model_select, temperature],
        outputs=[result_output, speed_output],
    )


if __name__ == "__main__":
    app.launch(debug=True)