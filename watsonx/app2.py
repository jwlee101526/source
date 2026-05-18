import gradio as gr

def review(name, grade):
    return  name + "⭐" * int(grade)

demo = gr.Interface(
    fn=review,
    inputs=[gr.Text(label="음식명"), gr.Slider(1,5,label="만족도")],
    outputs=[gr.Text(label="만족도 출력")],
    api_name="별점 리뷰 생성"
)

demo.launch()