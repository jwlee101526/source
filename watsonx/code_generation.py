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
    model_id="ibm/granite-8b-code-instruct",
    api_client=client,
    project_id=f"{project_id}",
    params={"max_tokens": 3000},
)


def code_text(text):

    system_prompt = """
    당신은 모든 프로그래밍 언어에 대한 코드 작성 전문가입니다.
    사용자의 요구 사항을 분석하여
    - 정확한 코드
    - 실행 가능한 코드
    - 가독성이 좋은 코드

    [규칙]
    1. 반드시 코드 블록(````) 형식으로 작성
    2. 코드에 적절한 주석 포함
    3. 필요한 라이브러리가 있다면 함께 설명
    4. 코드 동작 원리를 간단히 설명
    5. 오류 가능성이 있는 부분은 주의사항 추가
    6. 사용자의 요청 언어에 대한 적절한 코드 작성
    7. 불필요하게 긴 설명은 피하고 핵심 위주로 작성

    [응답 형식]
    1. 기능 설명
    2. 코드
    3. 코드 설명
    4. 실행 결과 또는 사용 예시
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    generated_response = model.chat(messages=messages)
    return generated_response["choices"][0]["message"]["content"]


demo = gr.Interface(
    fn=code_text,
    inputs=[
        gr.Textbox(lines=10, placeholder="여기에 코드를 입력하세요", label="요구사항 입력")
    ],
    outputs=[gr.Markdown()],
    title="코드 작성 프로그램",
    description="요구사항 작성 시 AI가 코드를 생성해 드립니다",
)

demo.launch()
