import gradio as gr
from transformers import pipeline


captioner = pipeline("image-to-text")
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

# 최근 업로드된 이미지를 기억하기 위한 전역 변수
current_caption = None


def chat(message, history):
    global current_caption

    print("message: ", message)
    print("history: ", history)

    try:
        text = message.get("text", "")

        if message.get("files"):
            # 파일 경로 추출
            file_info = message["files"][0]
            image_path = file_info["path"] if isinstance(file_info, dict) else file_info
            
            # 캡션 생성
            caption = captioner(image_path)[0]["generated_text"]
            current_caption = caption
            return caption
            
        elif text:
            if not current_caption:
                return f"""
                    사용자 질문
                    {text}
                """

            # Qwen 모델은 Chat Template 형식으로 사용
            messages = [
                {"role": "system", "content": "당신은 이미지 분석 AI입니다. 주어진 이미지 설명을 바탕으로 사용자의 질문에 한국어로 간결하게 한 문장으로 답변하세요."},
                {"role": "user", "content": f"이미지 설명: {current_caption}\n\n질문: {text}"}
            ]
            
            result = generator(
                messages,
                max_new_tokens=100,
                pad_token_id=generator.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )[0]["generated_text"]
            
            answer = result[-1]["content"] if isinstance(result, list) else result
            
            return answer

        else:
            return "메시지를 입력하거나 이미지를 업로드해주세요."

    except Exception as e:
        return f"이미지 처리 중 오류가 발생했습니다: {str(e)}"


demo = gr.ChatInterface(
    fn=chat,
    multimodal=True,
    title="멀티모달 AI 챗봇",
    description="이미지를 업로드하면 AI가 이미지에 대한 설명을 답변합니다.",
)

if __name__ == "__main__":
    demo.launch(debug=True)
