import base64

from langchain_ibm import ChatWatsonx, WatsonxEmbeddings
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
import os
from langchain_ollama import ChatOllama
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# 이미지 다운로드
url = "https://images.unsplash.com/photo-1770009079291-82b8a594ee23?q=80&w=709&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
img = requests.get(url).content

# 인코딩
img_b64 = base64.b64encode(img).decode()

# 모델
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
                {"type": "text", "text": "이 사진에 대해 자세히 설명과 분석을 해라"},
            ],
        )
    ]
)
