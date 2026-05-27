from langchain_ollama import ChatOllama
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv()


MODELS = ["qwen", "exaone", "watsonx"]


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


def call_langchain(message, history, model_name, temperature):
    if not message.strip():
        yield "", history
        return

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 전문적인 베테랑 요리 전문가입니다. 질문자의 질문에 대한 궁금점을 해결하고, 음식의 재료, 조리법, 팁을 명확하고 간결하게 한국어로 설명하세요.",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | config_llm(model_name, temperature) | StrOutputParser()

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    for chunk in chain.stream({"question": message}):
        history[-1]["content"] += chunk
        yield "", history


with gr.Blocks(title="food chatbot") as app:
    gr.Markdown("## Food Chatbot")

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### 모델 선택")
            model_select = gr.Radio(
                choices=MODELS,
                label="LLM",
            )

            gr.Markdown("### 파라미터")
            temperature = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                label="Temperature",
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Chat")
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="(예: 김치찌개 끓이는 방법)",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                send_btn = gr.Button("전송", variant="primary", scale=1)

    inputs = [msg, chatbot, model_select, temperature]
    outputs = [msg, chatbot]

    msg.submit(call_langchain, inputs, outputs)
    send_btn.click(call_langchain, inputs, outputs)


if __name__ == "__main__":
    app.launch(debug=True)
