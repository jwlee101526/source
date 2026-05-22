import gradio as gr
from transformers import pipeline

whisper = pipeline(task="automatic-speech-recognition", model="openai/whisper-base")
generator = pipeline(task="text-generation")

def change_txt(file):
    result = whisper(file, return_timestamps=True)
    return result["text"]

def question_answer(question):
    """
    question 답변 생성 후 리턴
    """

    pass


# ========================================================================
# Gradio UI
# ========================================================================

with gr.Blocks(title="AI 음성 비서") as app:
    gr.Markdown("""
        <h1 style="text-align: center;">AI 음성 비서</h1>
        <p style="text-align: center;">음성으로 명령을 내리고, AI의 답변을 들을 수 있는 통합 인터페이스입니다.</p>
        """)

    with gr.Tabs():
        with gr.TabItem("음성 처리"):
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("### 1. 음성 파일 업로드")
                    audio_input = gr.Audio(label="마이크 입력", type="filepath")
                    btn_transcribe = gr.Button("텍스트로 변환", variant="primary")

                with gr.Column(variant="panel"):
                    gr.Markdown("### 2. 변환된 텍스트")
                    text_transcribed = gr.Textbox(label="인식 결과", lines=5)

        with gr.TabItem("질문 및 답변"):
            with gr.Column():

                question_input = gr.Textbox(
                    label="질문하기", placeholder="질문을 입력하세요"
                )
                btn_ask = gr.Button("텍스트 변환", variant="primary")

            with gr.Column():
                answer_output = gr.Textbox(label="AI 답변", lines=3)
                btn_tts = gr.Button("답변 음성으로 듣기")

            with gr.Column():
                audio_output = gr.Audio(label="AI 음성 응답", autoplay=True)


    btn_transcribe.click(
        fn=change_txt, inputs=audio_input, outputs=text_transcribed
    )
    btn_ask.click(
        fn=question_answer, inputs=question_input, outputs=[answer_output, audio_output]
    )

if __name__ == "__main__":
    app.launch(theme="NeoPy/shadowthedgehog", debug=True)
