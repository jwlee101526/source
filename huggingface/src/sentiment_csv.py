import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from transformers import pipeline
from tqdm import tqdm

classifier = pipeline(
    task="sentiment-analysis",
    model="tabularisai/multilingual-sentiment-analysis",
    top_k=None,
)


def predict_sentiment(file, progress=gr.Progress(track_tqdm=True)):

    df_input = pd.read_csv(file.name)
    reviews = df_input['review'].to_list()

    results_text = []
    
    label_map = {
        "Very Positive": "매우 긍정 😁",
        "Positive": "긍정 😊",
        "Neutral": "중립 😐",
        "Negative": "부정 🤨",
        "Very Negative": "매우 부정 😞",
    }
    
    pos_labels = {"Very Positive", "Positive"}
    neg_labels = {"Very Negative", "Negative"}
    
    positive_count = 0
    negative_count = 0
    positive_scores = []
    negative_scores = []
    
    best_pos = {"sentence": "", "label": "", "score": -1}
    best_neg = {"sentence": "", "label": "", "score": -1}
    
    for sentence in tqdm(reviews):
        result = classifier(sentence)[0]

        best = max(result, key=lambda x: x['score'])
        label = best["label"]
        display_label = label_map.get(label, label)
        score = best["score"]

        if label in pos_labels:
            positive_count += 1
            positive_scores.append(score)
            
            if label == "Very Positive" and best_pos["label"] != "Very Positive":
                best_pos = {"sentence": sentence, "label": label, "score": score}
            elif (label == "Very Positive" and score > best_pos["score"]) or \
                 (label == "Positive" and best_pos["label"] != "Very Positive" and score > best_pos["score"]):
                best_pos = {"sentence": sentence, "label": label, "score": score}

        elif label in neg_labels:
            negative_count += 1
            negative_scores.append(score)
            
            if label == "Very Negative" and best_neg["label"] != "Very Negative":
                best_neg = {"sentence": sentence, "label": label, "score": score}
            elif (label == "Very Negative" and score > best_neg["score"]) or \
                 (label == "Negative" and best_neg["label"] != "Very Negative" and score > best_neg["score"]):
                best_neg = {"sentence": sentence, "label": label, "score": score}

        results_text.append([sentence, display_label, score])
    
    total = len(reviews)
    positive_ratio = (positive_count / total) * 100 if total > 0 else 0
    negative_ratio = (negative_count / total) * 100 if total > 0 else 0
    avg_pos = sum(positive_scores) / len(positive_scores) if positive_scores else 0
    avg_neg = sum(negative_scores) / len(negative_scores) if negative_scores else 0

    stats = (
        f"### 분석 결과 요약\n\n"
        f"- **총 리뷰 수**: {total} 개\n"
        f"- **긍정 리뷰**: {positive_count} 개 (평균 점수: {avg_pos:.4f})\n"
        f"- **부정 리뷰**: {negative_count} 개 (평균 점수: {avg_neg:.4f})\n\n"
        f"---\n"
        f"- **긍정 비율**: **{positive_ratio:.2f}%**\n"
        f"- **부정 비율**: **{negative_ratio:.2f}%**\n\n"
        f"---\n"
        f"- **가장 긍정적인 리뷰**: '{best_pos['sentence']}' ({best_pos['score']:.4f})\n"
        f"- **가장 부정적인 리뷰**: '{best_neg['sentence']}' ({best_neg['score']:.4f})"
    )

    # Chart
    fig, ax = plt.subplots()
    ax.pie(
        [positive_count, negative_count],
        labels=["긍정", "부정"],
        autopct="%.1f%%",
        startangle=90,
        counterclock=False
    )

    return stats, results_text, fig


with gr.Blocks() as app:
    gr.Markdown("# AI 감정분석 웹앱")
    gr.Markdown("HuggingFace Transformers기반 감정분석 프로그램")

    input_file = gr.File()
    btn = gr.Button("분석하기")

    with gr.Row():
        with gr.Column(scale=1):
            text_output = gr.Markdown(
                label="분석 결과 요약",
                visible=False
            )
        with gr.Column(scale=1):
            chart_box = gr.Plot(
                label="분석 결과 그래프", 
                visible=False
            )
            
    table_output = gr.Dataframe(
        headers=["문장", "감정", "점수"],
        label="문장별 상세 결과",
        visible=False
    )

    btn.click(predict_sentiment, inputs=input_file, outputs=[text_output, table_output, chart_box]).then(
        fn=lambda: [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)], 
        outputs=[text_output, table_output, chart_box]
    )



if __name__ == "__main__":
    app.launch(debug=True)
