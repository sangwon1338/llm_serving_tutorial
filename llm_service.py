from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import bitsandbytes as bnb
import requests
# FastAPI 인스턴스 생성
app = FastAPI()

# Llama 모델 및 토크나이저 로드 (4-bit 양자화)
model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"  # Hugging Face에서 지원하는 모델 이름
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델을 4-bit로 양자화하여 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,  # 여러 GPU에 자동으로 분산
    quantization_config=bnb_config
)

# 모델을 CUDA로 이동 (가능한 경우)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# 요청 데이터 형식 정의
class ChatRequest(BaseModel):
    query: str  # ChatRequest의 속성으로 query 정의
    max_tokens: int = 50

# 응답 생성 함수
def generate_response(context, query, max_tokens=50):
    full_prompt = f"다음 문서를 바탕으로 질문에 답하세요:\n\n{context}\n\n질문: {query}\n답변:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("답변:")[-1]

# 채팅 API 엔드포인트
@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query

    # 벡터 DB 서비스에서 관련 문서 검색
    response = requests.post("http://localhost:8001/search", json={"query": query})
    relevant_docs = response.json()["documents"]

    # 검색된 문서의 텍스트를 결합하여 Llama 모델에 전달
    context = "\n".join(relevant_docs)
    generated_response = generate_response(context, query, request.max_tokens)

    return {"response": generated_response}

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)