import base64
import io

from dotenv import load_dotenv
import os
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import gradio as gr
from PIL import Image

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
    params={"max_tokens": 3000},
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

def travel_recommand(message, history):

    system_prompt = f"""
    당신은 여행 설계 전문가 입니다.

    사용자가 업로드 한 이미지의

    - 분위기
    - 감성
    - 색감
    - 스타일

    을 분석해서 여행지를 추천할 것.

    반드시 

    1. 일정표
    2. 추천장소
    3. 맛집
    4. 예상 비용

    을 포함할 것
    """

    messages_to_send = [
        {"role": "system", "content": system_prompt}
    ]

    for item in history:
        role = item["role"]
        content = item["content"]
        
        texts = []

        if isinstance(content, list):
            for c in content:
                
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
        elif isinstance(content,str):
            texts.append(content)

        messages_to_send.append({"role": role, "content": "".join(texts)})

    
    text = message.get("text", "")
    files = message.get("files", "")

    if files:
        image = Image.open(files[0])
        base64_img = image_to_base64(image)
        messages_to_send.append({
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
                    "text": text,
                },
            ],
        })
    else:
        messages_to_send.append({"role": "user", "content": text})

    response = model.chat(messages=messages_to_send)

    full_response = response["choices"][0]["message"]["content"]

    return full_response


demo = gr.ChatInterface(
    fn=travel_recommand,
    multimodal=True,
    title="AI 여행 플래너 V3",
    description="가고 싶은 여행지의 사진과 여행 스타일을 입력하면 AI가 맞춤형 여행 일정을 추천해 드립니다.",
)

demo.launch()