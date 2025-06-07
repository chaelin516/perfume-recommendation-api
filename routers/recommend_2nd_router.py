# routers/recommend_2nd_router.py
# 🆕 2차 향수 추천 API - 프론트엔드 데이터 매핑 수정 버전

import os
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from collections import Counter
import re

# ✅ 공통 함수들을 recommend_router에서 임포트
from routers.recommend_router import (
    check_model_availability,
    get_model,
    get_saved_encoder,
    get_fallback_encoder,
    safe_transform_input,
    predict_cluster_recommendation,
    EMOTION_CLUSTER_MAP,
    _model_available
)

# ─── 로거 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("recommend_2nd_router")

# ─── 1. 데이터 로딩 ───────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
try:
    df = pd.read_csv(DATA_PATH)
    df.fillna("", inplace=True)
    logger.info(f"✅ Perfume dataset loaded: {df.shape[0]} rows")

    # emotion_cluster 컬럼 정수형으로 변환
    if 'emotion_cluster' in df.columns:
        df['emotion_cluster'] = pd.to_numeric(df['emotion_cluster'], errors='coerce').fillna(0).astype(int)
        logger.info(f"📊 Emotion clusters: {sorted(df['emotion_cluster'].unique())}")

    logger.info(f"📋 Available columns: {list(df.columns)}")

except Exception as e:
    logger.error(f"❌ perfume_final_dataset.csv 로드 중 오류: {e}")
    raise RuntimeError(f"perfume_final_dataset.csv 로드 중 오류: {e}")


# ─── 2. 🆕 프론트엔드 데이터 매핑 함수 ─────────────────────────────────────────────────
def normalize_frontend_data(user_preferences: dict) -> dict:
    """
    프론트엔드에서 오는 데이터를 백엔드 AI 모델 형식에 맞게 변환
    """
    logger.info(f"🔄 프론트엔드 데이터 정규화 시작: {user_preferences}")

    # 1. 소문자 변환 매핑
    normalized = {}

    # gender 매핑
    gender = str(user_preferences.get('gender', '')).lower()
    if gender in ['unisex', 'men', 'women']:
        normalized['gender'] = gender
    else:
        # 기본값 또는 추론
        normalized['gender'] = 'unisex'
        logger.warning(f"⚠️ 알 수 없는 gender '{gender}', 기본값 'unisex' 사용")

    # season_tags 매핑
    season = str(user_preferences.get('season_tags', '')).lower()
    if season in ['spring', 'summer', 'fall', 'winter']:
        normalized['season_tags'] = season
    else:
        normalized['season_tags'] = 'summer'
        logger.warning(f"⚠️ 알 수 없는 season '{season}', 기본값 'summer' 사용")

    # time_tags 매핑
    time = str(user_preferences.get('time_tags', '')).lower()
    if time in ['day', 'night']:
        normalized['time_tags'] = time
    else:
        normalized['time_tags'] = 'day'
        logger.warning(f"⚠️ 알 수 없는 time '{time}', 기본값 'day' 사용")

    # activity 매핑
    activity = str(user_preferences.get('activity', '')).lower()
    if activity in ['casual', 'date', 'work']:
        normalized['activity'] = activity
    else:
        normalized['activity'] = 'casual'
        logger.warning(f"⚠️ 알 수 없는 activity '{activity}', 기본값 'casual' 사용")

    # weather 매핑
    weather = str(user_preferences.get('weather', '')).lower()
    if weather in ['any', 'cold', 'hot', 'rainy']:
        normalized['weather'] = weather
    else:
        normalized['weather'] = 'any'
        logger.warning(f"⚠️ 알 수 없는 weather '{weather}', 기본값 'any' 사용")

    # 2. 🚨 desired_impression 특별 처리 (가장 중요!)
    impression = str(user_preferences.get('desired_impression', '')).lower()

    # 지원되는 조합값들
    supported_impressions = [
        'confident, fresh',
        'confident, mysterious',
        'elegant, friendly',
        'pure, friendly'
    ]

    # 단일 값을 조합 값으로 매핑
    impression_mapping = {
        'pure': 'pure, friendly',
        'confident': 'confident, fresh',
        'fresh': 'confident, fresh',
        'mysterious': 'confident, mysterious',
        'elegant': 'elegant, friendly',
        'friendly': 'elegant, friendly'
    }

    if impression in supported_impressions:
        # 이미 올바른 조합 형태
        normalized['desired_impression'] = impression
    elif impression in impression_mapping:
        # 단일 값을 조합 값으로 변환
        normalized['desired_impression'] = impression_mapping[impression]
        logger.info(f"🔄 desired_impression 매핑: '{impression}' → '{normalized['desired_impression']}'")
    else:
        # 기본값 사용
        normalized['desired_impression'] = 'confident, fresh'
        logger.warning(f"⚠️ 알 수 없는 desired_impression '{impression}', 기본값 'confident, fresh' 사용")

    logger.info(f"✅ 정규화 완료: {normalized}")

    return normalized


def validate_ai_model_input(user_preferences: dict) -> bool:
    """AI 모델 입력 데이터 유효성 검사"""
    required_fields = ['gender', 'season_tags', 'time_tags', 'desired_impression', 'activity', 'weather']

    for field in required_fields:
        if field not in user_preferences:
            logger.error(f"❌ 필수 필드 누락: {field}")
            return False

    # 각 필드의 값 검증
    valid_values = {
        'gender': ['men', 'unisex', 'women'],
        'season_tags': ['fall', 'spring', 'summer', 'winter'],
        'time_tags': ['day', 'night'],
        'desired_impression': ['confident, fresh', 'confident, mysterious', 'elegant, friendly', 'pure, friendly'],
        'activity': ['casual', 'date', 'work'],
        'weather': ['any', 'cold', 'hot', 'rainy']
    }

    for field, valid_list in valid_values.items():
        value = user_preferences.get(field)
        if value not in valid_list:
            logger.error(f"❌ 잘못된 값: {field}='{value}', 유효한 값: {valid_list}")
            return False

    logger.info("✅ AI 모델 입력 데이터 유효성 검사 통과")
    return True


# ─── 3. 스키마 정의 ─────────────────────────────────────────────────────────────
class UserPreferences(BaseModel):
    """1차 추천을 위한 사용자 선호도 (AI 모델 입력) - 대소문자 관계없이 허용"""

    gender: str = Field(..., description="성별")
    season_tags: str = Field(..., description="계절")
    time_tags: str = Field(..., description="시간")
    desired_impression: str = Field(..., description="원하는 인상")
    activity: str = Field(..., description="활동")
    weather: str = Field(..., description="날씨")


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

    # Optional fields (기존 방식 호환성 유지)
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


class SecondRecommendItem(BaseModel):
    """2차 추천 결과 아이템"""

    name: str = Field(..., description="향수 이름")
    brand: str = Field(..., description="브랜드명")
    image_url: str = Field("", description="이미지 URL")
    notes: str = Field("", description="향수 노트")
    final_score: float = Field(..., description="최종 추천 점수 (0.0-1.0)", ge=0.0, le=1.0)
    emotion_cluster: int = Field(..., description="감정 클러스터 ID (0-5)", ge=0, le=5)
    reason: str = Field("", description="추천 이유")


# ─── 4. 노트 분석 유틸리티 함수들 ─────────────────────────────────────────────────
def parse_notes_from_string(notes_str: str) -> List[str]:
    """노트 문자열을 파싱하여 개별 노트 리스트로 변환"""
    if not notes_str or pd.isna(notes_str):
        return []

    # 콤마로 분리하고 앞뒤 공백 제거, 소문자 변환
    notes = [note.strip().lower() for note in str(notes_str).split(',')]
    notes = [note for note in notes if note and note != '']
    return notes


def normalize_note_name(note: str) -> str:
    """노트명을 정규화 (유사한 노트들을 매칭하기 위해)"""
    note = note.lower().strip()

    # 일반적인 노트명 정규화 규칙
    note_mappings = {
        # 시트러스 계열
        'bergamot': ['bergamot', 'bergamotte'],
        'lemon': ['lemon', 'citron'],
        'orange': ['orange', 'sweet orange'],
        'grapefruit': ['grapefruit', 'pink grapefruit'],
        'lime': ['lime', 'persian lime'],

        # 플로럴 계열
        'rose': ['rose', 'bulgarian rose', 'damascus rose', 'tea rose'],
        'jasmine': ['jasmine', 'sambac jasmine', 'star jasmine'],
        'lavender': ['lavender', 'french lavender'],
        'ylang-ylang': ['ylang-ylang', 'ylang ylang'],
        'iris': ['iris', 'orris'],

        # 우디 계열
        'cedar': ['cedar', 'cedarwood', 'atlas cedar'],
        'sandalwood': ['sandalwood', 'mysore sandalwood'],
        'oakmoss': ['oakmoss', 'oak moss'],
        'vetiver': ['vetiver', 'haitian vetiver'],

        # 앰버/오리엔탈 계열
        'amber': ['amber', 'grey amber'],
        'musk': ['musk', 'white musk', 'red musk'],
        'vanilla': ['vanilla', 'madagascar vanilla'],
        'benzoin': ['benzoin', 'siam benzoin'],

        # 스파이시 계열
        'pepper': ['pepper', 'black pepper', 'pink pepper'],
        'cinnamon': ['cinnamon', 'ceylon cinnamon'],
        'cardamom': ['cardamom', 'green cardamom'],
        'ginger': ['ginger', 'fresh ginger']
    }

    # 매핑 테이블에서 정규화된 이름 찾기
    for normalized, variants in note_mappings.items():
        if note in variants:
            return normalized

    return note


def calculate_note_match_score(perfume_notes: List[str], user_note_scores: Dict[str, int]) -> float:
    """향수의 노트와 사용자 선호도를 비교하여 매칭 점수 계산"""
    if not perfume_notes or not user_note_scores:
        return 0.0

    # 향수 노트를 정규화
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

        # 부분 매칭 (노트명이 포함되는 경우)
        else:
            partial_matches = []
            for perfume_note in normalized_perfume_notes:
                if normalized_user_note in perfume_note or perfume_note in normalized_user_note:
                    partial_matches.append(perfume_note)

            if partial_matches:
                # 부분 매칭은 50% 가중치
                normalized_preference = (preference_score / 5.0) * 0.5
                weight = preference_score / total_preference_weight
                contribution = normalized_preference * weight
                total_score += contribution
                matched_notes_count += 0.5

    if matched_notes_count == 0:
        return 0.0

    # 매칭 비율에 따른 보너스
    match_ratio = matched_notes_count / len(user_note_scores)
    match_bonus = match_ratio * 0.1

    final_score = min(1.0, total_score + match_bonus)
    return final_score


def calculate_emotion_cluster_weight(perfume_cluster: int, emotion_proba: List[float]) -> float:
    """향수의 감정 클러스터와 사용자의 감정 확률 분포를 기반으로 가중치 계산"""
    if perfume_cluster < 0 or perfume_cluster >= len(emotion_proba):
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
    # 노트 매칭 점수 70%, 감정 클러스터 가중치 25%, 다양성 보너스 5%
    final_score = (
            note_match_score * 0.70 +
            emotion_cluster_weight * 0.25 +
            diversity_bonus * 0.05
    )

    final_score = max(0.0, min(1.0, final_score))
    return final_score


def get_recommendation_reason(final_score: float, note_match_score: float, emotion_cluster: int) -> str:
    """추천 이유 생성"""
    cluster_desc = EMOTION_CLUSTER_MAP.get(emotion_cluster, f"클러스터 {emotion_cluster}")

    if final_score >= 0.8:
        return f"🎯 당신의 선호 노트와 {note_match_score:.1%} 일치하며, {cluster_desc} 감정에 완벽하게 어울립니다!"
    elif final_score >= 0.6:
        return f"✨ 당신의 취향과 {note_match_score:.1%} 일치하는 {cluster_desc} 스타일의 향수입니다."
    elif final_score >= 0.4:
        return f"🌟 {cluster_desc} 분위기의 향수로, 새로운 시도해볼 만한 선택입니다."
    else:
        return f"🔍 {cluster_desc} 계열의 색다른 매력을 가진 향수입니다."


# ─── 5. AI 모델 호출 함수 ─────────────────────────────────────────────────────────
def call_ai_model_for_first_recommendation(user_preferences: dict) -> Dict:
    """AI 모델을 호출하여 1차 추천 수행"""
    try:
        logger.info("🤖 AI 모델 1차 추천 호출 시작")

        # 🚨 데이터 정규화 적용
        normalized_preferences = normalize_frontend_data(user_preferences)

        # 🚨 유효성 검사
        if not validate_ai_model_input(normalized_preferences):
            raise ValueError("AI 모델 입력 데이터 유효성 검사 실패")

        logger.info(f"🔄 정규화된 선호도로 AI 모델 호출: {normalized_preferences}")

        # predict_cluster_recommendation 함수 호출
        result = predict_cluster_recommendation(normalized_preferences)

        # 결과 형태 변환
        return {
            "cluster": result["cluster"],
            "confidence": max(result["proba"]),
            "emotion_proba": result["proba"],
            "selected_idx": result["selected_idx"]
        }

    except Exception as e:
        logger.error(f"❌ AI 모델 1차 추천 실패: {e}")
        raise e


# ─── 6. 메인 추천 함수 ─────────────────────────────────────────────────────────
def process_second_recommendation_with_ai(
        user_preferences: dict,
        user_note_scores: Dict[str, int],
        emotion_proba: Optional[List[float]] = None,
        selected_idx: Optional[List[int]] = None
) -> List[Dict]:
    """AI 모델을 포함한 완전한 2차 추천 처리 함수"""
    start_time = datetime.now()

    logger.info(f"🎯 AI 모델 포함 2차 추천 처리 시작")

    # 1. emotion_proba 또는 selected_idx가 없으면 AI 모델 호출
    if emotion_proba is None or selected_idx is None:
        logger.info("🤖 AI 모델로 1차 추천 수행 (emotion_proba 또는 selected_idx 없음)")

        try:
            ai_result = call_ai_model_for_first_recommendation(user_preferences)

            if emotion_proba is None:
                emotion_proba = ai_result["emotion_proba"]
                logger.info(f"✅ AI 모델에서 감정 확률 획득: 클러스터 {ai_result['cluster']} (신뢰도: {ai_result['confidence']:.3f})")

            if selected_idx is None:
                selected_idx = ai_result["selected_idx"]
                logger.info(f"✅ AI 모델에서 선택 인덱스 획득: {len(selected_idx)}개")

        except Exception as e:
            logger.error(f"❌ AI 모델 1차 추천 실패: {e}")
            logger.info("📋 룰 기반 폴백으로 전환")

            # 룰 기반 폴백
            emotion_proba = [0.1, 0.15, 0.4, 0.15, 0.1, 0.1]

            # 🚨 정규화된 데이터로 기본 필터링
            try:
                normalized_preferences = normalize_frontend_data(user_preferences)
                candidates = df.copy()
                if 'gender' in df.columns and normalized_preferences.get("gender"):
                    gender_filtered = candidates[candidates['gender'] == normalized_preferences["gender"]]
                    if not gender_filtered.empty:
                        candidates = gender_filtered

                selected_idx = candidates.head(10).index.tolist()
                logger.info(f"📋 룰 기반 폴백으로 {len(selected_idx)}개 인덱스 생성")
            except Exception as fallback_e:
                logger.error(f"❌ 룰 기반 폴백도 실패: {fallback_e}")
                # 최종 안전장치
                selected_idx = list(range(10))

    # 2. 기존 2차 추천 로직 수행
    return process_second_recommendation(user_note_scores, emotion_proba, selected_idx)


def process_second_recommendation(
        user_note_scores: Dict[str, int],
        emotion_proba: List[float],
        selected_idx: List[int]
) -> List[Dict]:
    """2차 추천 처리 메인 함수"""
    start_time = datetime.now()

    logger.info(f"🎯 2차 추천 처리 시작")
    logger.info(f"  📝 사용자 노트 선호도: {user_note_scores}")
    logger.info(f"  🧠 감정 확률 분포: {[f'{p:.3f}' for p in emotion_proba]}")
    logger.info(f"  📋 선택된 인덱스: {selected_idx[:5]}... (총 {len(selected_idx)}개)")

    # 선택된 인덱스에 해당하는 향수들 필터링
    valid_indices = [idx for idx in selected_idx if idx < len(df)]

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
            perfume_image_url = str(row.get('image_url', ''))

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

            # 5. 추천 이유 생성
            reason = get_recommendation_reason(final_score, note_match_score, perfume_cluster)

            # 결과 저장
            result_item = {
                'name': perfume_name,
                'brand': perfume_brand,
                'image_url': perfume_image_url,
                'notes': perfume_notes_str,
                'final_score': round(final_score, 3),
                'emotion_cluster': perfume_cluster,
                'reason': reason,
                'note_match_score': round(note_match_score, 3),
                'emotion_weight': round(emotion_weight, 3),
                'diversity_bonus': round(diversity_bonus, 3),
                'original_index': valid_indices[idx]
            }

            results.append(result_item)

            # 상세 로깅 (상위 3개만)
            if idx < 3:
                logger.info(f"📊 #{idx + 1} {perfume_name}: 노트매칭={note_match_score:.3f}, "
                            f"감정가중치={emotion_weight:.3f}, 최종점수={final_score:.3f}")

        except Exception as e:
            logger.error(f"❌ 향수 '{row.get('name', 'Unknown')}' 처리 중 오류: {e}")
            continue

    # 최종 점수 기준으로 정렬
    results.sort(key=lambda x: x['final_score'], reverse=True)

    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"✅ 2차 추천 처리 완료: {len(results)}개 향수 (소요시간: {processing_time:.3f}초)")

    if results:
        top_scores = [r['final_score'] for r in results[:3]]
        logger.info(f"📊 상위 3개 점수: {top_scores}")

    return results


# ─── 7. 라우터 설정 ─────────────────────────────────────────────────────────────
router = APIRouter(prefix="/perfumes", tags=["Second Recommendation"])

# 시작 시 모델 가용성 확인
logger.info("🚀 2차 추천 시스템 (AI 모델 포함) 초기화 시작...")
try:
    global _model_available
    check_model_availability()
    if _model_available:
        logger.info("🤖 AI 감정 클러스터 모델 사용 가능")
    else:
        logger.info("📋 룰 기반 폴백 시스템으로 동작")
except Exception as e:
    logger.warning(f"⚠️ 모델 상태 확인 중 오류 (폴백으로 동작): {e}")
    _model_available = False

logger.info("✅ 2차 추천 시스템 초기화 완료")


@router.post(
    "/recommend-2nd",
    response_model=List[SecondRecommendItem],
    summary="2차 향수 추천 - AI 모델 + 노트 선호도 기반 (수정됨)",
    description=(
            "🎯 **완전한 End-to-End 2차 향수 추천 API (데이터 매핑 수정)**\n\n"
            "프론트엔드에서 오는 데이터를 자동으로 백엔드 AI 모델 형식에 맞게 변환하여\n"
            "정확한 1차 추천을 수행한 후, 노트 선호도와 결합하여 정밀한 2차 추천을 제공합니다.\n\n"
            "**🔄 자동 데이터 변환:**\n"
            "- 'Pure' → 'pure, friendly'\n"
            "- 'Unisex' → 'unisex'\n"
            "- 'Summer' → 'summer'\n"
            "- 기타 대소문자 및 형식 정규화\n\n"
            "**📥 입력 정보:**\n"
            "- `user_preferences`: 사용자 기본 선호도 (대소문자 무관)\n"
            "- `user_note_scores`: 사용자의 노트별 선호도 점수 (0-5)\n"
            "- `emotion_proba` (선택): 감정 확률 배열\n"
            "- `selected_idx` (선택): 선택된 향수 인덱스\n\n"
            "**🤖 처리 과정:**\n"
            "1. 데이터 정규화 → 프론트엔드 데이터를 AI 모델 형식으로 변환\n"
            "2. AI 모델 호출 → 감정 클러스터 예측 + 향수 선택\n"
            "3. 노트 매칭 → 사용자 선호도와 향수 노트 비교\n"
            "4. 점수 계산 → 노트 매칭(70%) + 감정 가중치(25%) + 다양성(5%)\n"
            "5. 최종 정렬 → 점수 기준 내림차순\n\n"
            "**✨ 특징:**\n"
            "- 🔄 자동 데이터 정규화\n"
            "- 🤖 AI 모델 자동 호출\n"
            "- 🎯 정확한 노트 매칭\n"
            "- 📊 감정 클러스터 기반 가중치\n"
            "- 🔄 다단계 폴백 시스템\n"
            "- 🌟 브랜드 다양성 보장"
    )
)
def recommend_second_perfumes(request: SecondRecommendRequest):
    """AI 모델 포함 완전한 2차 향수 추천 API (데이터 매핑 수정)"""

    request_start_time = datetime.now()

    logger.info(f"🆕 AI 모델 포함 2차 향수 추천 요청 접수")
    logger.info(f"  👤 원본 사용자 선호도: {request.user_preferences.dict()}")
    logger.info(f"  📊 노트 선호도 개수: {len(request.user_note_scores)}개")

    try:
        # 메인 추천 처리 (AI 모델 포함, 데이터 정규화 적용)
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

        # 응답 형태로 변환
        response_items = []
        for result in results:
            response_items.append(
                SecondRecommendItem(
                    name=result['name'],
                    brand=result['brand'],
                    image_url=result['image_url'],
                    notes=result['notes'],
                    final_score=result['final_score'],
                    emotion_cluster=result['emotion_cluster'],
                    reason=result['reason']
                )
            )

        # 처리 시간 계산
        total_processing_time = (datetime.now() - request_start_time).total_seconds()

        logger.info(f"✅ AI 모델 포함 2차 추천 완료: {len(response_items)}개 향수")
        logger.info(f"⏱️ 총 처리 시간: {total_processing_time:.3f}초")
        logger.info(f"📊 최고 점수: {response_items[0].final_score:.3f} ({response_items[0].name})")
        logger.info(f"📊 점수 범위: {response_items[-1].final_score:.3f} ~ {response_items[0].final_score:.3f}")

        # 클러스터별 분포 로깅
        cluster_distribution = {}
        for item in response_items:
            cluster_distribution[item.emotion_cluster] = cluster_distribution.get(item.emotion_cluster, 0) + 1
        logger.info(f"📊 클러스터별 분포: {cluster_distribution}")

        return response_items

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ AI 모델 포함 2차 추천 처리 중 예외 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"AI 모델 포함 2차 추천 처리 중 오류가 발생했습니다: {str(e)}"
        )


# ─── 8. 시스템 상태 API ─────────────────────────────────────────────────────
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

        # 노트 통계 (샘플링)
        all_notes = []
        for _, row in df.head(100).iterrows():
            notes = parse_notes_from_string(str(row.get('notes', '')))
            all_notes.extend(notes)

        note_frequency = Counter(all_notes)
        top_notes = dict(note_frequency.most_common(20))

        # AI 모델 상태 확인
        try:
            global _model_available
            model_status = "available" if _model_available else "fallback"
        except:
            model_status = "unknown"

        return {
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),  # ✅ timestamp 추가
            "ai_model_status": model_status,
            "dataset_info": {
                "total_perfumes": total_perfumes,
                "unique_brands": unique_brands,
                "columns": list(df.columns),
                "sample_size_for_notes": 100
            },
            "emotion_clusters": {
                "available_clusters": list(EMOTION_CLUSTER_MAP.keys()),
                "cluster_descriptions": EMOTION_CLUSTER_MAP,
                "distribution": cluster_distribution
            },
            "note_analysis": {
                "top_20_notes": top_notes,
                "unique_notes_in_sample": len(note_frequency),
                "total_note_occurrences": len(all_notes)
            },
            "supported_features": [
                "프론트엔드 데이터 자동 정규화",  # 🆕 추가
                "노트 선호도 기반 매칭",
                "감정 클러스터 가중치",
                "브랜드 다양성 보장",
                "노트명 정규화",
                "부분 매칭 지원",
                "AI 모델 자동 호출",
                "다단계 폴백 시스템"  # 🆕 추가
            ],
            "data_mapping": {  # 🆕 추가
                "frontend_to_backend": {
                    "Pure → pure, friendly": "단일 감정을 조합 감정으로 매핑",
                    "Unisex → unisex": "대소문자 정규화",
                    "Summer → summer": "대소문자 정규화",
                    "Day → day": "대소문자 정규화",
                    "Work → work": "대소문자 정규화",
                    "Any → any": "대소문자 정규화"
                }
            }
        }

    except Exception as e:
        logger.error(f"❌ 시스템 상태 확인 중 오류: {e}")
        return {
            "system_status": "error",
            "timestamp": datetime.now().isoformat(),
            "error_message": str(e)
        }


# ─── 9. 디버깅 API들 ─────────────────────────────────────────────────────
@router.post(
    "/debug/test-data-mapping",
    summary="데이터 매핑 테스트",
    description="프론트엔드 데이터가 백엔드 형식으로 올바르게 변환되는지 테스트합니다."
)
def test_data_mapping(frontend_data: Dict):
    """데이터 매핑 테스트 API"""
    try:
        logger.info(f"🧪 데이터 매핑 테스트 시작: {frontend_data}")

        # 정규화 수행
        normalized = normalize_frontend_data(frontend_data)

        # 유효성 검사
        is_valid = validate_ai_model_input(normalized)

        return {
            "original_data": frontend_data,
            "normalized_data": normalized,
            "is_valid_for_ai_model": is_valid,
            "mapping_applied": {
                field: {
                    "original": frontend_data.get(field, "N/A"),
                    "normalized": normalized.get(field, "N/A"),
                    "changed": frontend_data.get(field, "").lower() != normalized.get(field, "")
                }
                for field in ['gender', 'season_tags', 'time_tags', 'desired_impression', 'activity', 'weather']
            }
        }

    except Exception as e:
        logger.error(f"❌ 데이터 매핑 테스트 중 오류: {e}")
        return {
            "error": str(e),
            "original_data": frontend_data
        }


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


@router.post(
    "/test-note-matching",
    summary="노트 매칭 테스트",
    description="사용자 선호도와 향수 노트 간의 매칭 점수를 테스트합니다."
)
def test_note_matching(
        user_notes: Dict[str, int],
        perfume_notes: List[str]
):
    """노트 매칭 테스트 API"""

    try:
        # 노트 매칭 점수 계산
        match_score = calculate_note_match_score(perfume_notes, user_notes)

        # 정규화된 노트들
        normalized_perfume_notes = [normalize_note_name(note) for note in perfume_notes]
        normalized_user_notes = {normalize_note_name(k): v for k, v in user_notes.items()}

        # 매칭 상세 정보
        matches = []
        for user_note, preference in user_notes.items():
            normalized_user_note = normalize_note_name(user_note)
            if normalized_user_note in normalized_perfume_notes:
                matches.append({
                    "user_note": user_note,
                    "normalized": normalized_user_note,
                    "preference": preference,
                    "match_type": "exact"
                })

        return {
            "user_notes": user_notes,
            "perfume_notes": perfume_notes,
            "normalized_perfume_notes": normalized_perfume_notes,
            "match_score": round(match_score, 3),
            "matches": matches,
            "match_count": len(matches),
            "total_user_notes": len(user_notes)
        }

    except Exception as e:
        logger.error(f"❌ 노트 매칭 테스트 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"노트 매칭 테스트 중 오류가 발생했습니다: {str(e)}"
        )