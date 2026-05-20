import re
import gradio as gr
from transformers import pipeline

classifier = pipeline(
    task="sentiment-analysis",
    model="tabularisai/multilingual-sentiment-analysis"
)

LABEL_MAPPINGS = {
    "Very Positive": "긍정",
    "Positive": "긍정",
    "Negative": "부정"
}

def predict_sentiment(text: str) -> str:
    if not text or not text.strip():
        return "텍스트를 입력해주세요."

    try:
        result = classifier(text)[0]
        raw_label = result["label"]
        score = result["score"]

        final_label = LABEL_MAPPINGS.get(raw_label, raw_label)

        return f"감정 : {final_label}, 점수 : {score:.4f}"

    except Exception as e:
        return f"분석 중 오류가 발생했습니다: {str(e)}"

app = gr.Interface(
    fn=predict_sentiment,
    title="AI 감정분석 웹앱",
    description="HuggingFace Transformers를 사용한 모델 기반 감정분석",
    inputs=[gr.Text(
        label="Text", 
        lines=3,
        placeholder="문장을 입력하세요",
    )],
    outputs=[gr.Text(
        label="Result",
        lines=3
    )],
)

if __name__ == "__main__":
    app.launch(debug=True)
