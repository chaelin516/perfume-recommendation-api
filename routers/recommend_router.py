from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import os

# ─── 데이터 로드 ────────────────────────────────────────────────────────────────
# recommend_router.py 파일 기준으로 한 단계 상위 폴더의 data/perfume_final_dataset.csv
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
df = pd.read_csv(DATA_PATH)
df.fillna("", inplace=True)

# ─── 모델 및 인코더 경로 설정 ───────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_model_perfume.keras")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../models/encoder.pkl")

# ─── lazy loading 용 전역 변수 ───────────────────────────────────────────────────────
_model = None
_encoder = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model

def get_encoder():
    global _encoder
    if _encoder is None:
        with open(ENCODER_PATH, "rb") as f:
            _encoder = pickle.load(f)
    return _encoder

# ─── 요청(Request) 스키마 ───────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    gender: Literal['women', 'men', 'unisex']
    season: Literal['spring', 'summer', 'fall', 'winter']
    time: Literal['day', 'night']
    impression: Literal['confident', 'elegant', 'pure', 'friendly', 'mysterious', 'fresh']
    activity: Literal['casual', 'work', 'date']
    weather: Literal['hot', 'cold', 'rainy', 'any']

# ─── 응답(Response) 스키마 ──────────────────────────────────────────────────────────
class PerfumeRecommendItem(BaseModel):
    name: str
    brand: str
    image_url: str
    notes: str
    emotions: str
    reason: str

router = APIRouter(prefix="/perfumes", tags=["Perfume"])

@router.post(
    "/recommend",
    response_model=List[PerfumeRecommendItem],
    summary="모델 기반 향수 추천",
    description="gender, season, time, impression, activity, weather 정보를 입력받아 Keras 모델 예측으로 향수 10개를 추천합니다."
)
def recommend_perfumes(request: RecommendRequest):
    # 1) raw_features: 요청 필드 6개를 순서대로 리스트에 담기
    raw_features = [
        request.gender,
        request.season,
        request.time,
        request.impression,
        request.activity,
        request.weather
    ]

    # 2) encoder 로드 후 전처리(transform)
    encoder = get_encoder()
    try:
        # encoder.transform 은 2D 배열([n_samples, n_features])을 기대
        x_input = encoder.transform([raw_features])  # 결과: (1, D) 형태
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"입력값 전처리 중 오류: {str(e)}")

    # 3) 모델 로드 후 예측(predict)
    model = get_model()
    try:
        preds = model.predict(x_input)  # preds.shape == (1, num_perfumes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 예측 중 오류: {str(e)}")

    # 4) 1차원 배열로 변환
    scores = preds.flatten()  # (num_perfumes,) 형태

    # 5) 모델 출력 크기와 DataFrame 행 개수 일치 확인
    if len(scores) != len(df):
        raise HTTPException(
            status_code=500,
            detail="모델 출력 크기가 데이터프레임 행 개수와 맞지 않습니다."
        )

    # 6) DataFrame 복사본에 점수(score) 컬럼 추가
    df_with_scores = df.copy()
    df_with_scores["score"] = scores

    # 7) score 기준 내림차순 정렬 후 상위 10개 추출
    top_10 = df_with_scores.sort_values(by="score", ascending=False).head(10)

    # 8) 최종 응답 리스트 생성
    response_list: List[PerfumeRecommendItem] = []
    for _, row in top_10.iterrows():
        response_list.append(
            PerfumeRecommendItem(
                name=row["name"],
                brand=row["brand"],
                image_url=row["image_url"],
                notes=row["notes"],
                emotions=row["emotion_tags"],
                reason=f"추천 이유: 모델 예측 점수 {row['score']:.3f}"
            )
        )

    return response_list
