import gradio as gr

def predict(image):
    # 이미지 width,height 정보 리턴
    width, height = image.size
    return f"""
    이미지 분석 결과
    - 가로 : {width}px
    - 세로 : {height}px
    - 이미지 모드 : {image.mode}
    """

interface = gr.Interface(predict, gr.Image(type="pil"), gr.Textbox())
interface.launch()