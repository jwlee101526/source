import re
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

    # 마침표, 물음표, 느낌표 또는 줄바꿈 기준으로 문장 분리
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]

    try:
        all_results = classifier(sentences)

        label_map = {
            "Very Positive": "매우 긍정 😁",
            "Positive": "긍정 😊",
            "Neutral": "중립 😐",
            "Negative": "부정 🤨",
            "Very Negative": "매우 부정 😞",
        }

        detailed_data = []
        for sentence, results in zip(sentences, all_results):
            # 가장 높은 점수의 감정만 추출
            top_res = max(results, key=lambda x: x['score'])
            label = label_map.get(top_res["label"], top_res["label"])
            detailed_data.append({
                "문장": sentence, 
                "감정": label, 
                "점수": round(top_res["score"], 4)
            })

        df = pd.DataFrame(detailed_data)
        label_scores = {d["감정"]: d["점수"] for d in detailed_data}

        return label_scores, df

    except Exception as e:
        print(e)
        return {}, None


with gr.Blocks() as app:
    gr.Markdown("# AI 감정분석 웹앱")
    gr.Markdown("HuggingFace Transformers기반 감정분석 프로그램")

    input_text = gr.Textbox(label="Text", lines=5, placeholder="여러 문장을 입력하세요")
    btn = gr.Button("분석하기")

    with gr.Row():
        text_output = gr.Label(label="분석 결과 요약")
        table_output = gr.Dataframe(
            label="문장별 상세 결과",
            headers=["문장", "감정", "점수"],
            datatype=["str", "str", "number"],
            col_count=(3, "fixed")
        )

    btn.click(predict_sentiment, inputs=input_text, outputs=[text_output, table_output])


if __name__ == "__main__":
    app.launch(debug=True)
