import gradio as gr

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

demo = gr.Interface(
    fn=bmi_calculator,
    inputs=[gr.Number(label="키"), gr.Number(label="몸무게")],
    outputs=[gr.Text(label="BMI 판정")],
    api_name="BMI 판정기"
)

demo.launch()