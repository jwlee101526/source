import gradio as gr

with gr.Blocks(title="AI 음성 비서") as app:
    gr.Markdown("# AI 음성 비서")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="오디오", type="filepath")
            btn_transcribe = gr.Button("텍스트 변환")

        with gr.Column(scale=1):
            text_transcribed = gr.Textbox(label="텍스트 변환")

    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(label="질문하기")
            btn_ask = gr.Button("질문하기")
        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="answer", placeholder="답변")
            btn_tts = gr.Button("답변 음성 변환")

    with gr.Row():
        audio_outut = gr.Audio(label="AI 음성 답변", autoplay=True)


if __name__ == "__main__":
    app.launch(debug=True)
