import gradio as gr
from transformers import pipeline

summarizer = pipeline(task="summarization")
classifier = pipeline(task="sentiment-analysis")
ner = pipeline(task="ner", grouped_entities=True) # 개체명 인식
translator = pipeline(task="translation", model="facebook/nllb-200-distilled-600M")

def analyze_news(article):
    if not article:
        return "", "", "", ""

    # 1. 기사 요약
    summary_result = summarizer(article, truncation=True)
    summary = summary_result[0]["summary_text"]

    # 2. 감성 분석
    sentiment_result = classifier(article, truncation=True)
    sentiment = (
        f"감성 : {sentiment_result[0]['label']} "
        f"점수 : {sentiment_result[0]['score']:.4f}"
    )

    # 3. 키워드 추출
    ner_result = ner(article)
    keywords = []

    for item in ner_result:
        word = item['word']

        if word not in keywords:
            keywords.append(word)

    keyword_text = ", ".join(keywords)

    # 4. 번역
    translated_result = translator(summary, src_lang="eng_Latn", tgt_lang="kor_Hang")
    translated_summary = translated_result[0]['translation_text']

    return summary, sentiment, keyword_text, translated_summary


# ========================================================================
# Gradio UI
# ========================================================================

with gr.Blocks(title="AI 뉴스 분석기") as app:
    gr.Markdown("""
        <h1 style="text-align: center;">AI 뉴스 분석기</h1>
        """)

    with gr.Row():
        with gr.Column(scale=2, variant="panel"):
            gr.Markdown("### 1. 기사 입력")
            article_input = gr.Textbox(
                label="영문 뉴스 기사 입력",
                lines=15,
                placeholder="영문 뉴스 기사를 입력하세요."
            )
            btn_analyze = gr.Button("뉴스 분석 시작", variant="primary")

        with gr.Column(scale=2, variant="panel"):
            gr.Markdown("### 2. 분석 결과")
            summary_output = gr.Textbox(
                label="기사 분석 결과",
                lines=5
            )
            sentiment_output = gr.Textbox(
                label="감정 분석 결과",
                lines=5
            )
            ner_output = gr.Textbox(
                label="키워드 추출",
                lines=5
            )
            translate_output = gr.Textbox(
                label="번역 결과",
                lines=5
            )


    btn_analyze.click(
        fn=analyze_news,
        inputs=article_input,
        outputs=[summary_output, sentiment_output, ner_output, translate_output]
    )

if __name__ == "__main__":
    app.launch(theme="NeoPy/shadowthedgehog", debug=True)
