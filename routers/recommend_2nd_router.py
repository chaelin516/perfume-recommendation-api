# routers/recommend_2nd_router.py
# 🆕 완전히 수정된 2차 향수 추천 API - AI 모델 연동 완성

import os
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from collections import Counter
import re

# 🔗 1차 추천 모듈에서 필요한 함수들 import
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ─── 1차 추천 시스템 함수들 import ─────────────────────────────────────
try:
    from routers.recommend_router import (
        get_model,
        safe_transform_input,
        EMOTION_CLUSTER_MAP,
        _model_available,
        check_model_availability,
        df as perfume_df  # 향수 데이터셋
    )

    logger = logging.getLogger("recommend_2nd_router")
    logger.info("✅ 1차 추천 모듈 import 성공")
except ImportError as e:
    logger = logging.getLogger("recommend_2nd_router")
    logger.error(f"❌ 1차 추천 모듈 import 실패: {e}")
    # 폴백용 변수들
    _model_available = False
    EMOTION_CLUSTER_MAP = {
        0: "차분한, 편안한",
        1: "자신감, 신선함",
        2: "우아함, 친근함",
        3: "순수함, 친근함",
        4: "신비로운, 매력적",
        5: "활기찬, 에너지"
    }

# ✅ 업데이트된 스키마 import
from schemas.recommend import SecondRecommendItem

# ─── 로거 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ─── 데이터 로딩 ───────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
try:
    # 1차 추천에서 이미 로딩된 df 사용하거나 직접 로딩
    try:
        df = perfume_df  # 1차 추천에서 import
        logger.info(f"✅ 1차 추천 데이터셋 재사용: {df.shape[0]} rows")
    except:
        df = pd.read_csv(DATA_PATH)
        df.fillna("", inplace=True)
        logger.info(f"✅ 새로 로딩한 데이터셋: {df.shape[0]} rows")

    # emotion_cluster 컬럼 정수형으로 변환
    if 'emotion_cluster' in df.columns:
        df['emotion_cluster'] = pd.to_numeric(df['emotion_cluster'], errors='coerce').fillna(0).astype(int)
        logger.info(f"📊 Emotion clusters: {sorted(df['emotion_cluster'].unique())}")

    logger.info(f"📋 Available columns: {list(df.columns)}")

except Exception as e:
    logger.error(f"❌ perfume_final_dataset.csv 로드 중 오류: {e}")
    raise RuntimeError(f"perfume_final_dataset.csv 로드 중 오류: {e}")


# ─── 2차 추천 요청 스키마 정의 ─────────────────────────────────────────────────────────────
class UserPreferences(BaseModel):
    """1차 추천을 위한 사용자 선호도 (AI 모델 입력)"""

    gender: str = Field(..., description="성별", example="women")
    season_tags: str = Field(..., description="계절", example="spring")
    time_tags: str = Field(..., description="시간", example="day")
    desired_impression: str = Field(..., description="원하는 인상", example="confident, fresh")
    activity: str = Field(..., description="활동", example="casual")
    weather: str = Field(..., description="날씨", example="hot")


class SecondRecommendRequest(BaseModel):
    """2차 추천 요청 스키마 - AI 모델 호출 포함"""

    user_preferences: UserPreferences = Field(
        ...,
        description="1차 추천을 위한 사용자 선호도 (AI 모델 입력)"
    )

    user_note_scores: Dict[str, int] = Field(
        ...,
        description="사용자의 노트별 선호도 점수 (0-5)",
        example={
            "jasmine": 5,
            "rose": 4,
            "amber": 3,
            "musk": 0,
            "citrus": 2,
            "vanilla": 1
        }
    )

    # Optional fields (1차 추천 결과가 있으면 제공)
    emotion_proba: Optional[List[float]] = Field(
        None,
        description="6개 감정 클러스터별 확률 배열 (제공되지 않으면 AI 모델로 계산)",
        min_items=6,
        max_items=6,
        example=[0.01, 0.03, 0.85, 0.02, 0.05, 0.04]
    )

    selected_idx: Optional[List[int]] = Field(
        None,
        description="1차 추천에서 선택된 향수 인덱스 목록 (제공되지 않으면 AI 모델로 계산)",
        min_items=1,
        max_items=20,
        example=[23, 45, 102, 200, 233, 305, 399, 410, 487, 512]
    )

    @validator('user_note_scores')
    def validate_note_scores(cls, v):
        for note, score in v.items():
            if not isinstance(score, int) or score < 0 or score > 5:
                raise ValueError(f"노트 '{note}'의 점수는 0-5 사이의 정수여야 합니다.")
        return v

    @validator('emotion_proba')
    def validate_emotion_proba(cls, v):
        if v is None:
            return v
        if len(v) != 6:
            raise ValueError("emotion_proba는 정확히 6개의 확률값을 가져야 합니다.")
        total = sum(v)
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"emotion_proba의 합은 1.0에 가까워야 합니다. 현재: {total}")
        for prob in v:
            if not (0.0 <= prob <= 1.0):
                raise ValueError("각 확률값은 0.0-1.0 사이여야 합니다.")
        return v

    @validator('selected_idx')
    def validate_selected_idx(cls, v):
        if v is None:
            return v
        if len(set(v)) != len(v):
            raise ValueError("selected_idx에 중복된 인덱스가 있습니다.")
        for idx in v:
            if idx < 0:
                raise ValueError("인덱스는 0 이상이어야 합니다.")
        return v


# ─── 🆕 누락된 AI 모델 호출 함수 구현 ─────────────────────────────────────────────────────────────
def call_ai_model_for_first_recommendation(user_preferences: dict) -> Dict[str, Any]:
    """
    🆕 1차 추천용 AI 모델 호출 함수

    Args:
        user_preferences: 사용자 선호도 딕셔너리

    Returns:
        AI 모델 결과 (cluster, confidence, emotion_proba, selected_idx)
    """
    try:
        logger.info("🤖 2차 추천에서 1차 AI 모델 호출 시작")

        # 1. 모델 가용성 확인
        if not _model_available:
            raise Exception("AI 모델이 사용 불가능합니다")

        model = get_model()
        if model is None:
            raise Exception("모델 로딩 실패")

        # 2. 입력 데이터 변환
        raw_features = [
            user_preferences["gender"],
            user_preferences["season_tags"],
            user_preferences["time_tags"],
            user_preferences["desired_impression"],
            user_preferences["activity"],
            user_preferences["weather"]
        ]

        logger.info(f"🔮 AI 모델 입력: {raw_features}")

        # 3. 안전한 입력 변환 (encoder.pkl 사용)
        x_input = safe_transform_input(raw_features)

        # 4. 모델 예측 (final_model.keras 사용)
        preds = model.predict(x_input, verbose=0)
        cluster_probabilities = preds[0]
        predicted_cluster = int(np.argmax(cluster_probabilities))
        confidence = float(cluster_probabilities[predicted_cluster])

        logger.info(f"🎯 AI 예측 결과: 클러스터 {predicted_cluster} (신뢰도: {confidence:.3f})")

        # 5. 해당 클러스터의 향수들 선택
        if 'emotion_cluster' in df.columns:
            cluster_perfumes = df[df['emotion_cluster'] == predicted_cluster].copy()

            # 추가 필터링 적용
            original_count = len(cluster_perfumes)

            # 성별 필터링
            if 'gender' in df.columns and user_preferences.get("gender"):
                gender_filtered = cluster_perfumes[
                    cluster_perfumes['gender'] == user_preferences["gender"]
                    ]
                if not gender_filtered.empty:
                    cluster_perfumes = gender_filtered
                    logger.info(f"  성별 필터링: {original_count} → {len(cluster_perfumes)}개")

            # 계절 필터링
            if 'season_tags' in df.columns and user_preferences.get("season_tags"):
                season_filtered = cluster_perfumes[
                    cluster_perfumes['season_tags'].str.contains(
                        user_preferences["season_tags"], na=False, case=False
                    )
                ]
                if not season_filtered.empty:
                    cluster_perfumes = season_filtered
                    logger.info(f"  계절 필터링: → {len(cluster_perfumes)}개")

            # 시간 필터링
            if 'time_tags' in df.columns and user_preferences.get("time_tags"):
                time_filtered = cluster_perfumes[
                    cluster_perfumes['time_tags'].str.contains(
                        user_preferences["time_tags"], na=False, case=False
                    )
                ]
                if not time_filtered.empty:
                    cluster_perfumes = time_filtered
                    logger.info(f"  시간 필터링: → {len(cluster_perfumes)}개")

            # 상위 10개 선택
            selected_indices = cluster_perfumes.head(10).index.tolist()

        else:
            # emotion_cluster 컬럼이 없으면 전체에서 선택
            logger.warning("⚠️ emotion_cluster 컬럼 없음, 전체 데이터에서 선택")
            selected_indices = df.sample(n=10, random_state=42).index.tolist()

        logger.info(f"✅ AI 모델 선택 완료: {len(selected_indices)}개 향수")

        return {
            "cluster": predicted_cluster,
            "confidence": confidence,
            "emotion_proba": [round(float(p), 4) for p in cluster_probabilities],
            "selected_idx": selected_indices
        }

    except Exception as e:
        logger.error(f"❌ AI 모델 1차 추천 실패: {e}")
        raise e


# ─── 노트 분석 유틸리티 함수들 ─────────────────────────────────────────────────
def parse_notes_from_string(notes_str: str) -> List[str]:
    """노트 문자열을 파싱하여 개별 노트 리스트로 변환"""
    if not notes_str or pd.isna(notes_str):
        return []

    # 콤마로 분리하고 앞뒤 공백 제거, 소문자 변환
    notes = [note.strip().lower() for note in str(notes_str).split(',')]
    notes = [note for note in notes if note and note != '']
    return notes


def normalize_note_name(note: str) -> str:
    """노트명을 정규화"""
    note = note.lower().strip()

    # 일반적인 노트명 정규화 규칙
    note_mappings = {
        'bergamot': ['bergamot', 'bergamotte'],
        'lemon': ['lemon', 'citron'],
        'orange': ['orange', 'sweet orange'],
        'rose': ['rose', 'bulgarian rose', 'damascus rose', 'tea rose'],
        'jasmine': ['jasmine', 'sambac jasmine', 'star jasmine'],
        'lavender': ['lavender', 'french lavender'],
        'cedar': ['cedar', 'cedarwood', 'atlas cedar'],
        'sandalwood': ['sandalwood', 'mysore sandalwood'],
        'amber': ['amber', 'grey amber'],
        'musk': ['musk', 'white musk', 'red musk'],
        'vanilla': ['vanilla', 'madagascar vanilla'],
        'pepper': ['pepper', 'black pepper', 'pink pepper'],
    }

    for normalized, variants in note_mappings.items():
        if note in variants:
            return normalized
    return note


def calculate_note_match_score(perfume_notes: List[str], user_note_scores: Dict[str, int]) -> float:
    """향수의 노트와 사용자 선호도를 비교하여 매칭 점수 계산"""
    if not perfume_notes or not user_note_scores:
        return 0.0

    normalized_perfume_notes = [normalize_note_name(note) for note in perfume_notes]

    total_score = 0.0
    matched_notes_count = 0
    total_preference_weight = sum(user_note_scores.values())

    if total_preference_weight == 0:
        return 0.0

    for user_note, preference_score in user_note_scores.items():
        normalized_user_note = normalize_note_name(user_note)

        # 정확한 매칭
        if normalized_user_note in normalized_perfume_notes:
            normalized_preference = preference_score / 5.0
            weight = preference_score / total_preference_weight
            contribution = normalized_preference * weight
            total_score += contribution
            matched_notes_count += 1
        # 부분 매칭
        else:
            partial_matches = []
            for perfume_note in normalized_perfume_notes:
                if normalized_user_note in perfume_note or perfume_note in normalized_user_note:
                    partial_matches.append(perfume_note)

            if partial_matches:
                normalized_preference = (preference_score / 5.0) * 0.5
                weight = preference_score / total_preference_weight
                contribution = normalized_preference * weight
                total_score += contribution
                matched_notes_count += 0.5

    if matched_notes_count == 0:
        return 0.0

    # 매칭 비율 보너스
    match_ratio = matched_notes_count / len(user_note_scores)
    match_bonus = match_ratio * 0.1
    final_score = min(1.0, total_score + match_bonus)

    return final_score


def calculate_emotion_cluster_weight(perfume_cluster: int, emotion_proba: List[float]) -> float:
    """향수의 감정 클러스터와 사용자의 감정 확률 분포를 기반으로 가중치 계산"""
    if perfume_cluster < 0 or perfume_cluster >= len(emotion_proba):
        logger.warning(f"⚠️ 잘못된 클러스터 ID: {perfume_cluster}")
        return 0.1

    cluster_weight = emotion_proba[perfume_cluster]
    cluster_weight = max(0.05, cluster_weight)
    return cluster_weight


def calculate_final_score(
        note_match_score: float,
        emotion_cluster_weight: float,
        diversity_bonus: float = 0.0
) -> float:
    """최종 추천 점수 계산"""
    final_score = (
            note_match_score * 0.70 +
            emotion_cluster_weight * 0.25 +
            diversity_bonus * 0.05
    )
    return max(0.0, min(1.0, final_score))


# ─── 메인 2차 추천 처리 함수 ─────────────────────────────────────────────────────────────
def process_second_recommendation_with_ai(
        user_preferences: dict,
        user_note_scores: Dict[str, int],
        emotion_proba: Optional[List[float]] = None,
        selected_idx: Optional[List[int]] = None
) -> List[Dict]:
    """AI 모델을 포함한 완전한 2차 추천 처리 함수"""

    start_time = datetime.now()
    logger.info(f"🎯 AI 포함 2차 추천 처리 시작")

    # 1. emotion_proba 또는 selected_idx가 없으면 AI 모델 호출
    if emotion_proba is None or selected_idx is None:
        logger.info("🤖 AI 모델로 1차 추천 수행")

        try:
            ai_result = call_ai_model_for_first_recommendation(user_preferences)

            if emotion_proba is None:
                emotion_proba = ai_result["emotion_proba"]
                logger.info(f"✅ AI에서 감정 확률 획득: 클러스터 {ai_result['cluster']}")

            if selected_idx is None:
                selected_idx = ai_result["selected_idx"]
                logger.info(f"✅ AI에서 선택 인덱스 획득: {len(selected_idx)}개")

        except Exception as e:
            logger.error(f"❌ AI 모델 1차 추천 실패: {e}")
            logger.info("📋 룰 기반 폴백으로 전환")

            # 룰 기반 폴백
            emotion_proba = [0.1, 0.15, 0.4, 0.15, 0.1, 0.1]

            # 기본 필터링으로 selected_idx 생성
            candidates = df.copy()
            if 'gender' in df.columns and user_preferences.get("gender"):
                gender_filtered = candidates[candidates['gender'] == user_preferences["gender"]]
                if not gender_filtered.empty:
                    candidates = gender_filtered

            selected_idx = candidates.head(10).index.tolist()
            logger.info(f"📋 룰 기반 폴백: {len(selected_idx)}개 인덱스 생성")

    # 2. 2차 추천 로직 수행
    return process_second_recommendation(user_note_scores, emotion_proba, selected_idx)


def process_second_recommendation(
        user_note_scores: Dict[str, int],
        emotion_proba: List[float],
        selected_idx: List[int]
) -> List[Dict]:
    """2차 추천 처리 메인 함수"""

    start_time = datetime.now()
    logger.info(f"🎯 2차 추천 처리 시작")

    # 선택된 인덱스에 해당하는 향수들 필터링
    valid_indices = [idx for idx in selected_idx if idx < len(df)]
    invalid_indices = [idx for idx in selected_idx if idx >= len(df)]

    if invalid_indices:
        logger.warning(f"⚠️ 잘못된 인덱스들: {invalid_indices}")

    if not valid_indices:
        raise ValueError("유효한 향수 인덱스가 없습니다.")

    selected_perfumes = df.iloc[valid_indices].copy()
    logger.info(f"✅ {len(selected_perfumes)}개 향수 선택됨")

    # 각 향수에 대한 점수 계산
    results = []
    brand_count = {}

    for idx, (_, row) in enumerate(selected_perfumes.iterrows()):
        try:
            # 향수 기본 정보
            perfume_name = str(row['name'])
            perfume_brand = str(row['brand'])
            perfume_cluster = int(row.get('emotion_cluster', 0))
            perfume_notes_str = str(row.get('notes', ''))
            perfume_image_url = str(row.get('image_url', ''))  # 🆕 이미지 URL 추가

            # 노트 파싱
            perfume_notes = parse_notes_from_string(perfume_notes_str)

            # 1. 노트 매칭 점수 계산
            note_match_score = calculate_note_match_score(perfume_notes, user_note_scores)

            # 2. 감정 클러스터 가중치 계산
            emotion_weight = calculate_emotion_cluster_weight(perfume_cluster, emotion_proba)

            # 3. 다양성 보너스 계산
            brand_count[perfume_brand] = brand_count.get(perfume_brand, 0) + 1
            diversity_bonus = max(0.0, 0.1 - (brand_count[perfume_brand] - 1) * 0.02)

            # 4. 최종 점수 계산
            final_score = calculate_final_score(note_match_score, emotion_weight, diversity_bonus)

            # 결과 저장
            result_item = {
                'name': perfume_name,
                'brand': perfume_brand,
                'final_score': round(final_score, 3),
                'emotion_cluster': perfume_cluster,
                'image_url': perfume_image_url,  # 🆕 이미지 URL 포함
                'note_match_score': round(note_match_score, 3),
                'emotion_weight': round(emotion_weight, 3),
                'diversity_bonus': round(diversity_bonus, 3),
                'perfume_notes': perfume_notes,
                'original_index': valid_indices[idx]
            }

            results.append(result_item)

        except Exception as e:
            logger.error(f"❌ 향수 '{row.get('name', 'Unknown')}' 처리 중 오류: {e}")
            continue

    # 최종 점수 기준으로 정렬
    results.sort(key=lambda x: x['final_score'], reverse=True)

    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"✅ 2차 추천 처리 완료: {len(results)}개 향수 (소요시간: {processing_time:.3f}초)")

    return results


# ─── 라우터 설정 ────────────────────────────────────────────────
router = APIRouter(prefix="/perfumes", tags=["Second Recommendation"])

# 시작 시 모델 가용성 확인
logger.info("🚀 2차 추천 시스템 (AI 모델 포함) 초기화 시작...")
try:
    check_model_availability()
    if _model_available:
        logger.info("🤖 AI 감정 클러스터 모델 사용 가능")
    else:
        logger.info("📋 룰 기반 폴백 시스템으로 동작")
except:
    logger.warning("⚠️ 모델 상태 확인 실패, 폴백 모드로 진행")
logger.info("✅ 2차 추천 시스템 초기화 완료")


# ─── 🆕 완전한 2차 추천 API ────────────────────────────────────────────────
@router.post(
    "/recommend-2nd",
    response_model=List[SecondRecommendItem],
    summary="2차 향수 추천 - AI 모델 + 노트 선호도 기반",
    description=(
            "🎯 **완전한 End-to-End 2차 향수 추천 API**\n\n"
            "1차 추천의 결과를 받아서 사용자의 노트 선호도와 결합하여 정밀한 2차 추천을 제공합니다.\n\n"
            "**📥 입력 정보:**\n"
            "- `user_preferences`: 사용자 기본 선호도 (AI 모델 입력용, 필요시에만)\n"
            "- `user_note_scores`: 사용자의 노트별 선호도 점수 (0-5)\n"
            "- `emotion_proba` (선택): 1차 추천의 감정 확률 배열\n"
            "- `selected_idx` (선택): 1차 추천의 선택된 향수 인덱스\n\n"
            "**🤖 처리 과정:**\n"
            "1. **선택적 AI 호출**: emotion_proba/selected_idx 없으면 AI 모델 호출\n"
            "2. **노트 매칭**: user_note_scores와 향수 노트 비교\n"
            "3. **점수 계산**: 노트 매칭(70%) + 감정 가중치(25%) + 다양성(5%)\n"
            "4. **최종 정렬**: 점수 기준 내림차순 정렬\n\n"
            "**📤 출력 정보:**\n"
            "- 향수별 최종 추천 점수, 감정 클러스터, 이미지 URL 포함\n"
            "- 점수 기준 내림차순 정렬"
    )
)
def recommend_second_perfumes(request: SecondRecommendRequest):
    """완전한 2차 향수 추천 API"""

    request_start_time = datetime.now()

    logger.info(f"🆕 2차 향수 추천 요청 접수")
    logger.info(f"  👤 사용자 선호도: {request.user_preferences.dict()}")
    logger.info(f"  📊 노트 선호도 개수: {len(request.user_note_scores)}개")

    # emotion_proba나 selected_idx 제공 여부 확인
    has_emotion_proba = request.emotion_proba is not None
    has_selected_idx = request.selected_idx is not None

    if has_emotion_proba and has_selected_idx:
        logger.info(f"  🧠 1차 추천 결과 제공됨")
        logger.info("  ⚡ 2차 추천 바로 실행 (AI 모델 호출 건너뜀)")
    else:
        logger.info("  🤖 1차 추천 결과 없음 → AI 모델 호출 예정")

    try:
        # 메인 추천 처리 (AI 모델 포함)
        results = process_second_recommendation_with_ai(
            user_preferences=request.user_preferences.dict(),
            user_note_scores=request.user_note_scores,
            emotion_proba=request.emotion_proba,
            selected_idx=request.selected_idx
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail="추천할 수 있는 향수가 없습니다."
            )

        # ✅ 응답 형태로 변환 (image_url 포함)
        response_items = []
        for result in results:
            response_items.append(
                SecondRecommendItem(
                    name=result['name'],
                    brand=result['brand'],
                    final_score=result['final_score'],
                    emotion_cluster=result['emotion_cluster'],
                    image_url=result['image_url']  # 🆕 이미지 URL 포함
                )
            )

        # 처리 시간 계산
        total_processing_time = (datetime.now() - request_start_time).total_seconds()

        logger.info(f"✅ 2차 추천 완료: {len(response_items)}개 향수")
        logger.info(f"⏱️ 총 처리 시간: {total_processing_time:.3f}초")
        logger.info(f"📊 최고 점수: {response_items[0].final_score:.3f} ({response_items[0].name})")

        # 클러스터별 분포 로깅
        cluster_distribution = {}
        for item in response_items:
            cluster_distribution[item.emotion_cluster] = cluster_distribution.get(item.emotion_cluster, 0) + 1
        logger.info(f"📊 클러스터별 분포: {cluster_distribution}")

        return response_items

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 2차 추천 처리 중 예외 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"2차 추천 처리 중 오류가 발생했습니다: {str(e)}"
        )


# ─── 추가 유틸리티 API들 ─────────────────────────────────────────────────────
@router.get(
    "/note-analysis/{perfume_index}",
    summary="향수 노트 분석",
    description="특정 향수의 노트 정보를 분석하여 반환합니다."
)
def analyze_perfume_notes(perfume_index: int):
    """향수 노트 분석 API"""

    try:
        if perfume_index < 0 or perfume_index >= len(df):
            raise HTTPException(
                status_code=404,
                detail=f"잘못된 향수 인덱스: {perfume_index} (범위: 0-{len(df) - 1})"
            )

        perfume = df.iloc[perfume_index]
        notes_str = str(perfume.get('notes', ''))
        parsed_notes = parse_notes_from_string(notes_str)
        normalized_notes = [normalize_note_name(note) for note in parsed_notes]

        return {
            "perfume_index": perfume_index,
            "name": str(perfume['name']),
            "brand": str(perfume['brand']),
            "image_url": str(perfume.get('image_url', '')),
            "raw_notes": notes_str,
            "parsed_notes": parsed_notes,
            "normalized_notes": normalized_notes,
            "note_count": len(parsed_notes),
            "emotion_cluster": int(perfume.get('emotion_cluster', 0))
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 노트 분석 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"노트 분석 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/system-status",
    summary="2차 추천 시스템 상태",
    description="2차 추천 시스템의 상태와 통계를 반환합니다."
)
def get_system_status():
    """시스템 상태 확인 API"""

    try:
        # 데이터셋 통계
        total_perfumes = len(df)
        unique_brands = df['brand'].nunique() if 'brand' in df.columns else 0

        # 감정 클러스터 분포
        cluster_distribution = {}
        if 'emotion_cluster' in df.columns:
            cluster_counts = df['emotion_cluster'].value_counts().to_dict()
            cluster_distribution = {int(k): int(v) for k, v in cluster_counts.items()}

        return {
            "system_status": "operational",
            "ai_model_available": _model_available,
            "dataset_info": {
                "total_perfumes": total_perfumes,
                "unique_brands": unique_brands,
                "columns": list(df.columns),
                "has_image_url": 'image_url' in df.columns
            },
            "emotion_clusters": {
                "available_clusters": list(EMOTION_CLUSTER_MAP.keys()),
                "cluster_descriptions": EMOTION_CLUSTER_MAP,
                "distribution": cluster_distribution
            },
            "features": [
                "1차-2차 추천 완전 연동",
                "AI 모델 자동 호출",
                "노트 선호도 기반 매칭",
                "감정 클러스터 가중치",
                "브랜드 다양성 보장",
                "이미지 URL 포함"
            ]
        }

    except Exception as e:
        logger.error(f"❌ 시스템 상태 확인 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"시스템 상태 확인 중 오류가 발생했습니다: {str(e)}"
        )