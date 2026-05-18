import gradio as gr

def cheer(name, level):
    return name + "님 화이팅 " + "🎉" * int(level)

def review(name, grade):
    return  name + "⭐" * int(grade)

def bmi_calculator(height, weight):

    bmi = weight / ((float(height) / 100) ** 2)

    if bmi < 18.5:
        result = "저체중"
    elif bmi < 22.9: 
        result = "정상체중"
    elif bmi < 24.9:
        result = "과체중"
    else:
        result = "비만"

    # 당신의 BMI 지수는 키 : 158, 몸무게 : 60, 판정 : 저체중
    return  f"당신의 BMI 지수는 키 : {height}, 몸무게 : {weight}, 판정 : {result}"

with gr.Blocks() as demo:
    with gr.Tab("응원"):
        name = gr.Text(label="이름")
        chreer_strength = gr.Slider(1,5,label="응원강도")
        msg = gr.Textbox(label="응원 메시지")
        cheer_btn = gr.Button("응원")
        cheer_btn.click(fn=cheer, inputs=[name, chreer_strength], outputs=[msg])
    with gr.Tab("별점"):
        name = gr.Text(label="음식명")
        level = gr.Slider(1,5,label="별점")
        msg = gr.Textbox(label="만족도 확인")
        review_btn = gr.Button("별점등록")
        review_btn.click(fn=review, inputs=[name,level], outputs=[msg])
    with gr.Tab("BMI"):
        height = gr.Number(label="키")
        weight = gr.Number(label="몸무게")
        result = gr.Text(label="BMI 판정")
        gr.Button("BMI 판정").click(fn=bmi_calculator, inputs=[height,weight],outputs=[result])


demo.launch()
