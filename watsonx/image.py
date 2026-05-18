from dotenv import load_dotenv
import os
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import gradio as gr
import io
import base64

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
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    api_client=client,
    project_id=f"{project_id}",
    params={"max_tokens": 1000},
)

def image_to_base64(image):
    """
    pillow 형식의 이미지를 가져와 원하는 포맷으로 저장
    base64 인코딩
    """

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return image_base64


def summarize_image(image):

    
    if image is None:
        return "이미지를 업로드하여 주십시오."

    base64_img = image_to_base64(image)

    system_prompt = """
    당신은 이미지 분석 전문가 AI입니다. 주어진 이미지를 사용자의 요청에 따라 
    - 이미지 설명
    - 분위기 분석
    - 감정 분석
    - 객체 설명
    - 캡션 생성
    - 스타일 분석

    등을 수행하세요
    """

    user_prompt = "주어진 이미지를 분석해라"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    },
                },
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ],
        },
    ]

    generated_response = model.chat(messages=messages)
    return generated_response["choices"][0]["message"]["content"]


demo = gr.Interface(
    fn=summarize_image,
    inputs=[
        gr.Image(type="pil"),
    ],
    outputs=[gr.Markdown()],
    title="이미지 분석 프로그램",
    description="이미지 업로드 시 ai 요약",
)

demo.launch()
