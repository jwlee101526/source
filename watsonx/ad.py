from dotenv import load_dotenv
import os
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import gradio as gr

load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

credentials = Credentials(
    url=f"{watsonx_ai_url}",
    api_key=f"{apikey}",
)
client = APIClient(credentials)

model = ModelInference(
    model_id="ibm/granite-4-h-small",
    api_client=client,
    project_id=f"{project_id}",
    params={"max_tokens": 1000},
)


def ad_text(name, brand_name, strengthen, tone, keyword, value):

    system_prompt = """
    당신은 최고의 카피라이터 입니다.
    당신의 임무는 주어진 조건을 이용해 창의적인 광고 문구를 작성하는 것입니다.
    """

    user_prompt = f"""
    아래 내용을 참고하여 1~2줄 짜리 광고 문구 5개를 작성해
    - 제품명 : {name}
    - 브랜드명 : {brand_name}
    - 제품특징 : {strengthen}
    - 브랜드 핵심 가치 : {value}
    - 톤앤매너 : {tone}
    - 필수 포함 키워드 : {keyword}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":  user_prompt},
    ]

    generated_response = model.chat(messages=messages)
    return generated_response["choices"][0]["message"]["content"]


demo = gr.Interface(
    fn=ad_text,
    inputs=[
        gr.Textbox(lines=1, placeholder="제품명 입력", label="제품명"),
        gr.Textbox(lines=1, placeholder="브랜드명 입력", label="브랜드명"),
        gr.Textbox(lines=1, placeholder="제품특징 입력", label="제품특징"),
        gr.Textbox(lines=1, placeholder="톤앤매너 입력", label="톤앤매너"),
        gr.Textbox(lines=1, placeholder="필수 포함 키워드 입력", label="필수 포함 키워드"),
        gr.Textbox(lines=1, placeholder="브랜드 핵심 가치 입력", label="브랜드 핵심 가치"),
    ],
    outputs=[gr.Markdown()],
    title="광고 문구 프로그램",
    description="광고 문구 생성",
)

demo.launch()
