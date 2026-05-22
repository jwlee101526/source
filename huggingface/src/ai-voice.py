import edge_tts
import gradio as gr
import asyncio
from transformers import pipeline

whisper = pipeline(task="automatic-speech-recognition", model="openai/whisper-base")
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
tts = pipeline(task="text-to-speech")

voice_txt = ""
current_answer = ""


async def text_to_voice(text):
    voice = "ko-KR-InJoonNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("answer.mp3")


def make_voice():
    global current_answer
    if not current_answer:
        return None

    asyncio.run(text_to_voice(current_answer))

    return "answer.mp3"


def change_txt(file):
    global voice_txt
    result = whisper(file, return_timestamps=True)
    voice_txt = result["text"]
    return voice_txt


def question_answer(question):
    """
    question 답변 생성 후 리턴
    """
    global voice_txt, current_answer

    if not voice_txt:
        return "음성을 텍스트로 변환 후 질문하세요", None

    prompt = f"""
    다음 음성 내용을 참고하여 질문에 답변하세요.
    음성 내용:
    {voice_txt}

    질문:
        {question}

    답변 : 
    """

    # 위치 인자로 prompt 전달
    result = generator(
        prompt,
        max_new_tokens=50,
        return_full_text=False,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    current_answer = result[0]["generated_text"].strip()
    
    # 답변 생성 후 바로 음성 파일 생성
    asyncio.run(text_to_voice(current_answer))

    return current_answer, "answer.mp3"


# ========================================================================
# Gradio UI
# ========================================================================

with gr.Blocks(title="AI 음성 비서") as app:
    gr.Markdown("""
        <h1 style="text-align: center;">AI 음성 비서</h1>
        <p style="text-align: center;">음성으로 명령을 내리고, AI의 답변을 들을 수 있는 통합 인터페이스입니다.</p>
        """)

    with gr.Row():
        with gr.Column(variant="panel"):
            gr.Markdown("### 1. 음성 파일 업로드")
            audio_input = gr.Audio(label="마이크 입력", type="filepath")
            btn_transcribe = gr.Button("텍스트로 변환", variant="primary")

        with gr.Column(variant="panel"):
            gr.Markdown("### 2. 변환된 텍스트")
            text_transcribed = gr.Textbox(label="인식 결과", lines=5)

    with gr.Column():
        question_input = gr.Textbox(label="질문하기", placeholder="질문을 입력하세요")
        btn_ask = gr.Button("텍스트 변환", variant="primary")

    with gr.Column():
        answer_output = gr.Textbox(label="AI 답변", lines=3)
        btn_tts = gr.Button("답변 음성으로 듣기")

    with gr.Column():
        audio_output = gr.Audio(label="AI 음성 응답", autoplay=True)

    btn_transcribe.click(fn=change_txt, inputs=audio_input, outputs=text_transcribed)
    btn_ask.click(
        fn=question_answer, inputs=question_input, outputs=[answer_output, audio_output]
    )
    btn_tts.click(fn=make_voice, outputs=audio_output)

if __name__ == "__main__":
    app.launch(theme="NeoPy/shadowthedgehog", debug=True)
