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


def interview_text(genre, question_number):

    system_prompt = """
    당신은 전문적인 인터뷰어 입니다.
    당신의 임무는 주어진 조건을 이용해 인터뷰 질문을 작성하는 것입니다.
    """

    user_prompt = f"""
    아래 내용을 참고하여 구체적 인터뷰 질문을 작성해라
    - 장르 : {genre}
    - 질문 수 : {question_number}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    generated_response = model.chat(messages=messages)
    return generated_response["choices"][0]["message"]["content"]


demo = gr.Interface(
    fn=interview_text,
    inputs=[
        gr.Textbox(lines=1, placeholder="장르 입력", label="genre"),
        gr.Textbox(lines=1, placeholder="질문 수 입력", label="question_number"),
    ],
    outputs=[gr.Markdown()],
    title="인터뷰 질문 생성 프로그램",
    description="장르 작성 시 AI가 장르의 특징 및 인터뷰 질문을 생성합니다."
)

demo.launch()
