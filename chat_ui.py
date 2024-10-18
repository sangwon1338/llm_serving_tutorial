import streamlit as st
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from vllm import LLM, SamplingParams





# FastAPI 서버 URL 설정
API_URL = "http://localhost:8000/chat"

# Streamlit 페이지 설정
st.title("Llama 3 Chatbot")
st.write("FastAPI와 연결된 Llama 3 모델을 사용한 채팅 시스템입니다.")

# 사용자 입력 섹션
user_input = st.text_input("질문을 입력하세요:")

# '답변 받기' 버튼 생성
if st.button("답변 받기"):
    if user_input:
        # FastAPI 서버에 POST 요청을 보낼 데이터
        payload = {
            "query": user_input,
            "max_tokens": 50  # 최대 생성 토큰 수 설정
        }
        
        # FastAPI 서버에 요청 보내기
        with st.spinner("답변을 생성 중입니다..."):
            response = requests.post(API_URL, json=payload)
        
        # 응답이 성공적으로 돌아온 경우
        if response.status_code == 200:
            result = response.json()
            st.write(f"챗봇: {result['response']}")
        else:
            st.error("FastAPI 서버에서 오류가 발생했습니다.")
    else:
        st.warning("질문을 입력하세요.")