from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import os

# FastAPI 인스턴스 생성
app = FastAPI()

# Chroma DB 생성 및 임베딩 모델 로드
persist_directory = "./chroma_db"
embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct',    model_kwargs={"device": "cuda:0"},  # cuda, cpu
    encode_kwargs={"normalize_embeddings": True},)

# 예시 문서 데이터
documents = [
    Document(page_content="L1708년(강희 47년)에 강희제는 내몽골의 여러 칸들과 신료들이 모인 자리에서 황태자 윤잉이 주색잡기에만 빠져 있고 정신이 이상해져 자신의 기대를 저버렸다는 것을 알리고 황태자를 폐위시켰다. 폐위 직후 윤잉이 군사를 이끌고 강희제가 묵던 천막을 힐끔거리며 강희제를 시해하려 하였다는데, 사실 이것은 윤잉의 이복형인 윤시가 자신이 슬쩍 군사를 움직이고 그것을 윤잉의 잘못으로 모함하였던 것이다. 강희제는 진실을 알고 나서 크게 후회하며 이듬해인 1709년(강희 48년)에 자신의 죽은 황후인 효성인황후에 대한 마음과 사실 윤잉이 대역죄를 저지르지 않았기 때문에 다시 황태자로 복위시켜 주었다. 그러나 윤잉은 반성하지 않고 심지어 부황의 비빈들까지도 노렸고 더군다나 일부 비빈을 범하기도 하였다. 이에 분한 강희제는 천인공노할 패륜을 저질렀다 하며 윤잉을 크게 비난하였다. 또한 강희제가 남쪽으로 순행을 갈 당시 강희제를 몰아낼 정변을 주도했다 하여 윤잉을 다시 황태자에서 폐위시키고 영원히 서인으로 삼아 함안궁에 유폐하니 그 때가 1712년(강희 51년)이었다.."),
    Document(page_content='빅서에서 발견된 약 50 마리를 1911년 보호한 이래 남방해달의 서식지는 계속적으로 확장되는 추세에 있으나, 2007년과 2010년 사이 해달 개체수와 서식지는 다소 줄어들었다. 2010년 봄 현재, 남방해달의 서식지 북방 한계선은 투니타스 개울에서 남동쪽으로 2 킬로미터 정도 떨어진 피죤포인트로 옮겨갔고, 남방 한계선은 COP 유전에서 가비오타 주립공원으로 이동했다. 최근에는 특정 시아노박테리아(마이크로시스티스Microcystis)가 생산하는 마이크로시스틴이라는 독소가 해달이 먹이로 삼는 조개에 집중적으로 감염되어 해달을 중독시키고 있다. 시아노박테리아는 질소와 인이 풍부한 고인 물에서 번성한다. 이 질소와 인은 대개 오수 정화조나 농업용 비료가 유출된 것으로, 우기에 유속이 빠를 때 바다로 유입될 수 있다. 2010년, 캘리포니아 해안선을 따라 많은 수의 해달 시체가 발견되었으며, 상어 습격률이 증가한 것 역시 해달의 죽음을 재촉하는 요소로 작용하고 있다. 백상아리(Carcharodon carcharias)는 상대적으로 지방이 빈약한 편인 해달을 잡아먹지는 않지만, 상어에 물려서 죽은 해달 시체는 1980년에 8%, 1990년에 15%, 2010년과 2011년 사이에 30%로 증가 추세에 있다.'),
    Document(page_content='전투는 한국 전쟁의 가장 혹독한 겨울 날씨 상황에서 벌어졌다. 도로는 한국의 산악 지형을 뚫고 만들어졌으며 가파른 경사와 골짜기로 이루어졌다. 황초령과 덕동고개와 같은 주요 고지가 도로 전체를 감제하고 있었다. 도로의 사정은 열악했고, 몇몇 구간에서는 도로가 일차선이었다. 1950년 11월 14일 시베리아에서 내려온 한랭전선이 장진호 전체를 뒤덮었고, 이에 따라 기온이 영하 37도까지 내려갔다. 추운 날씨는 땅을 얼게 만들었고, 이로 인해 미끄러운 도로와 동상자 발생, 무기 오작동의 위험도 수반하게 되었다. 모르핀 역시 부상자들에게 투여하려면 얼지 않도록 해야 했다. 냉동 액체는 전장에서 아무런 쓸모가 없었으며 부상을 치료하기 위해 천을 찢는 것은 괴저와 동상의 위험이 있었다. 지프와 라디오를 이용하는 진지들은 온도가 낮아져 제대로 기능하지 못했다. 총기의 윤활유는 젤리처럼 변했고 전투에서 총을 쓰는 것도 어려웠다. 격발 핀의 용수철도 총탄을 원활하게 발사하지 못하거나 걸리적거리는 경우도 있었다.'),
]


embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct',    model_kwargs={"device": "cuda:1"},  # cuda, cpu
    encode_kwargs={"normalize_embeddings": True},)
# 3. Chroma 벡터 데이터베이스 생성 및 문서 삽입
persist_directory = "./chroma_db"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# 요청 데이터 형식 정의
class QueryRequest(BaseModel):
    query: str
    k: int = 3  # 검색할 문서 개수

# 문서 검색 API
@app.post("/search")
async def search_documents(request: QueryRequest):
    query = request.query
    k = request.k
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)
    
    # 검색된 문서 반환
    return {
        "documents": [doc.page_content for doc in relevant_docs[:k]]
    }

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)