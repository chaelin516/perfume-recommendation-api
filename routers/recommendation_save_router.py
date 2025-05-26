from fastapi import APIRouter, Depends
from sqlmodel import Session
from perfume_backend.db.session import get_session
from perfume_backend.models.recommendation import RecommendedPerfume
from perfume_backend.schemas.recommendation import SaveRecommendationsRequest
from fastapi.responses import JSONResponse
from datetime import datetime

router = APIRouter(prefix="/recommendations", tags=["Recommendation"])

@router.post("/save", summary="추천 향수 저장", description="1차 또는 2차 추천 향수 목록을 저장합니다.")
def save_recommendations(request: SaveRecommendationsRequest, session: Session = Depends(get_session)):
    for item in request.recommendations:
        record = RecommendedPerfume(
            user_id=request.user_id,
            recommend_round=request.recommend_round,
            perfume_name=item.perfume_name,
            perfume_brand=item.perfume_brand,
            score=item.score,
            created_at=datetime.utcnow()
        )
        session.add(record)
    session.commit()
    return JSONResponse(content={"message": "추천 결과가 성공적으로 저장되었습니다."})
