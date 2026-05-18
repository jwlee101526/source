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
    params={"max_tokens": 3000},
)


def travel_recommand(message, history):

    print ("history: " + str(history))

    system_prompt = f"""
    당신은 여행 설계 전문가 입니다.

    사용자의 

    - 여행 지역
    - 여행 스타일
    - 여행 기간
    - 예산

    해당 정보를 참고하여 여행 플랜을 설계하세요.

    반드시 

    1. 일정표
    2. 추천장소
    3. 맛집
    4. 예상 비용

    을 포함할 것
    """

    messages = [
        {"role": "system", "content": system_prompt}
    ]


    for item in history:
        content = item["content"][0]["text"]
        messages.append({"role": item["role"], "content": content})

    messages.append({"role": "user", "content": message})

    generated_response = model.chat_stream(messages=messages)

    full_response = ""

    for chunk in generated_response:
        if chunk["choices"]:
            full_response += chunk["choices"][0]["delta"].get("content", "")
            yield full_response

    return full_response


demo = gr.ChatInterface(
    fn=travel_recommand,
    title="AI 여행 플래너 V2",
    description="정보를 입력하면 AI가 맞춤형 여행 일정을 추천해 드립니다.",
)

demo.launch()
