import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from perfume_backend.utils.model_predictor import predict_emotion_cluster

router = APIRouter(prefix="/perfumes", tags=["Perfume"])

# CSV 로드 (도커 기준 상대 경로)
df = pd.read_csv("./data/perfume_final_dataset.csv")
df.fillna("", inplace=True)

# 벡터화 준비
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "))
emotion_matrix = vectorizer.fit_transform(df["emotion_tags"])

# 요청 스키마
class RecommendRequest(BaseModel):
    gender: Literal['women', 'men', 'unisex']
    season: Literal['spring', 'summer', 'fall', 'winter']
    time: Literal['day', 'night']
    impression: Literal['confident', 'elegant', 'pure', 'friendly', 'mysterious', 'fresh']
    activity: Literal['casual', 'work', 'date']
    weather: Literal['hot', 'cold', 'rainy', 'any']

# 응답 스키마
class PerfumeRecommendItem(BaseModel):
    name: str
    brand: str
    image_url: str
    notes: str
    emotions: str
    reason: str

# ✅ 향수 추천 API (1차)
@router.post(
    "/recommend",
    response_model=List[PerfumeRecommendItem],
    summary="1차 향수 추천",
    description="gender, season, time, impression, activity, weather 기반 향수 10개 추천"
)
def recommend_perfumes(request: RecommendRequest):
    filtered_df = df[
        (df["gender"] == request.gender) &
        (df["season_tags"] == request.season) &
        (df["time_tags"] == request.time) &
        (df["impression_tags"].str.contains(request.impression)) &
        (df["activity_tags"].str.contains(request.activity)) &
        ((request.weather == "any") | df["weather_tags"].str.contains(request.weather))
    ]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="조건에 맞는 향수가 없습니다.")

    filtered_vectors = vectorizer.transform(filtered_df["emotion_tags"])
    user_emotions = ["clean", "sensual"]
    user_vec = vectorizer.transform([", ".join(user_emotions)])

    similarities = cosine_similarity(user_vec, filtered_vectors).flatten()
    filtered_df = filtered_df.copy()
    filtered_df["similarity"] = similarities

    top_10 = filtered_df.sort_values(by="similarity", ascending=False).head(10)

    return [
        PerfumeRecommendItem(
            name=row["name"],
            brand=row["brand"],
            image_url=row["image_url"],
            notes=row["notes"],
            emotions=row["emotion_tags"],
            reason=f"추천 이유: 당신의 감정과 {round(row['similarity'] * 100, 1)}% 유사합니다."
        )
        for _, row in top_10.iterrows()
    ]

# ✅ 클러스터 기반 향수 추천
class ClusterRequest(BaseModel):
    cluster_id: int  # 0 ~ 5

@router.post(
    "/recommend-by-cluster",
    response_model=List[PerfumeRecommendItem],
    summary="클러스터 기반 향수 추천",
    description="감정 클러스터 ID(0~5)를 기반으로 향수 10개 추천"
)
def recommend_by_cluster(request: ClusterRequest):
    cluster_emotion_map = {
        0: ["clean", "fresh"],
        1: ["romantic", "elegant"],
        2: ["mysterious", "sensual"],
        3: ["friendly", "pure"],
        4: ["cozy", "calm"],
        5: ["confident", "energetic"]
    }

    target_tags = cluster_emotion_map.get(request.cluster_id)
    if not target_tags:
        raise HTTPException(status_code=400, detail="유효하지 않은 클러스터 ID입니다.")

    user_vec = vectorizer.transform([", ".join(target_tags)])
    similarities = cosine_similarity(user_vec, emotion_matrix).flatten()

    df_copy = df.copy()
    df_copy["similarity"] = similarities
    top_10 = df_copy.sort_values(by="similarity", ascending=False).head(10)

    return [
        PerfumeRecommendItem(
            name=row["name"],
            brand=row["brand"],
            image_url=row["image_url"],
            notes=row["notes"],
            emotions=row["emotion_tags"],
            reason=f"추천 이유: 감정 클러스터 #{request.cluster_id}와 유사한 향수예요."
        )
        for _, row in top_10.iterrows()
    ]

# ✅ 감정 클러스터 예측 API
class EmotionClusterRequest(BaseModel):
    gender: Literal['women', 'men', 'unisex']
    season: Literal['spring', 'summer', 'fall', 'winter']
    time: Literal['day', 'night']
    impression: Literal['confident', 'elegant', 'pure', 'friendly', 'mysterious', 'fresh']
    activity: Literal['casual', 'work', 'date']
    weather: Literal['hot', 'cold', 'rainy', 'any']

class EmotionClusterResponse(BaseModel):
    cluster_id: int

@router.post(
    "/predict-emotion",
    response_model=EmotionClusterResponse,
    summary="감정 클러스터 예측",
    description="입력 정보로부터 감정 클러스터 ID 예측"
)
def predict_emotion_cluster_id(request: EmotionClusterRequest):
    model_input = [
        request.gender,
        request.season,
        request.time,
        request.impression,
        request.activity,
        request.weather
    ]
    cluster_id = predict_emotion_cluster(model_input)
    return EmotionClusterResponse(cluster_id=cluster_id)
