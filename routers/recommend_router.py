# project_root/routers/recommend_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import os

# ─── 1. 데이터(향수 메타) 로드 ───────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
try:
    df = pd.read_csv(DATA_PATH)
    df.fillna("", inplace=True)
except Exception as e:
    raise RuntimeError(f"perfume_final_dataset.csv 로드 중 오류: {e}")

# ─── 2. Keras 모델 및 Encoder 경로 설정 ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_model_perfume.keras")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../models/encoder.pkl")

# ─── 3. 전역 변수(lazy loading) ─────────────────────────────────────────────────
_model = None
_encoder = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = load_model(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Keras 모델 로드 중 오류: {e}")
    return _model

def get_encoder():
    global _encoder
    if _encoder is None:
        try:
            with open(ENCODER_PATH, "rb") as f:
                _encoder = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"encoder.pkl 로드 중 오류: {e}")
    return _encoder

# ─── 4. 요청(Request) 스키마 정의 ────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    gender: Literal["women", "men", "unisex"]
    season: Literal["spring", "summer", "fall", "winter"]
    time: Literal["day", "night"]
    impression: Literal["confident", "elegant", "pure", "friendly", "mysterious", "fresh"]
    activity: Literal["casual", "work", "date"]
    weather: Literal["hot", "cold", "rainy", "any"]

# ─── 5. 응답(Response) 스키마 정의 ────────────────────────────────────────────────
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
    description=(
        "API로 받은 gender, season, time, impression, activity, weather 정보를 "
        "encoder.pkl에 정의된 전처리기 → Keras 모델에 그대로 전달하여, "
        "예측 점수가 높은 상위 10개 향수를 반환합니다."
    )
)
def recommend_perfumes(request: RecommendRequest):
    # ─── 1) 요청에서 받은 6개 값을 순서대로 리스트에 담기 ──────────────────────────────
    raw_features = [
        request.gender,
        request.season,
        request.time,
        request.impression,
        request.activity,
        request.weather
    ]

    # ─── 2) Encoder 로 전처리(transform) ─────────────────────────────────────────────
    try:
        encoder = get_encoder()
        # Encoder가 이미 fit된 상태여야 transform() 호출 가능
        x_input = encoder.transform([raw_features])  # 결과: (1, D) 형태의 NumPy 배열
    except Exception as e:
        # “This OneHotEncoder instance is not fitted yet...” 등의 오류 방지
        raise HTTPException(status_code=400, detail=f"입력값 전처리 중 오류: {e}")

    # ─── 3) Keras 모델 예측(predict) ───────────────────────────────────────────────
    try:
        model = get_model()
        preds = model.predict(x_input)  # preds.shape == (1, num_perfumes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 예측 중 오류: {e}")

    # ─── 4) 1차원 배열로 변환 ───────────────────────────────────────────────────────
    scores = preds.flatten()  # (num_perfumes,) 형태

    # ─── 5) 모델 출력 개수와 DataFrame 행 개수 일치 확인 ─────────────────────────────
    if len(scores) != len(df):
        raise HTTPException(
            status_code=500,
            detail="모델 출력 크기가 perfume_final_dataset.csv의 행 개수와 일치하지 않습니다."
        )

    # ─── 6) DataFrame 복사본에 점수(score) 컬럼 추가 ─────────────────────────────────
    df_with_scores = df.copy()
    df_with_scores["score"] = scores

    # ─── 7) score 내림차순 정렬 후 상위 10개 추출 ────────────────────────────────────
    top_10 = df_with_scores.sort_values(by="score", ascending=False).head(10)

    # ─── 8) 최종 응답 리스트 생성 ────────────────────────────────────────────────────
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
