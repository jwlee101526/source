import gradio as gr
import pandas as pd
from transformers import pipeline

classifier = pipeline(
    task="sentiment-analysis",
    model="tabularisai/multilingual-sentiment-analysis",
    top_k=None,
)


def predict_sentiment(text: str):
    if not text or not text.strip():
        return {}, None

    try:
        results = classifier(text)[0]

        label_map = {
            "Very Positive": "매우 긍정 😁",
            "Positive": "긍정 😊",
            "Neutral": "중립 😐",
            "Negative": "부정 🤨",
            "Very Negative": "매우 부정 😞",
        }

        sentiment_data = [
            {"감정": label_map.get(res["label"], res["label"]), "점수": res["score"]}
            for res in results
        ]

        label_scores = {d["감정"]: d["점수"] for d in sentiment_data}
        df = pd.DataFrame(sentiment_data)

        return label_scores, df

    except Exception as e:
        print(e)
        return {}, None


with gr.Blocks() as app:
    gr.Markdown("# AI 감정분석 웹앱")
    gr.Markdown("HuggingFace Transformers를 사용한 모델 기반 감정분석")

    input_text = gr.Textbox(
        label="Text", 
        lines=3, 
        placeholder="문장을 입력하세요"
    )

    btn = gr.Button("분석하기")

    with gr.Row():
        text_output = gr.Label(label="Result")

        plot_output = gr.BarPlot(
            value=None, 
            x="감정", 
            y="점수", 
            label="chart", 
            tooltip=["감정", "점수"]
        )

    btn.click(predict_sentiment, inputs=input_text, outputs=[text_output, plot_output])


if __name__ == "__main__":
    app.launch(debug=True)
