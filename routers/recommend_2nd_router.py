# routers/recommend_2nd_router.py
# 🆕 2차 향수 추천 API - 사용자 노트 선호도 기반 정밀 추천

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from collections import Counter
import re

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


# ─── 2. 스키마 정의 ─────────────────────────────────────────────────────────────
class UserNoteScores(BaseModel):
    """사용자 노트 선호도 점수 (0-5)"""
    jasmine: Optional[int] = Field(None, ge=0, le=5, description="자스민 선호도 (0-5)")
    rose: Optional[int] = Field(None, ge=0, le=5, description="장미 선호도 (0-5)")
    amber: Optional[int] = Field(None, ge=0, le=5, description="앰버 선호도 (0-5)")
    musk: Optional[int] = Field(None, ge=0, le=5, description="머스크 선호도 (0-5)")
    citrus: Optional[int] = Field(None, ge=0, le=5, description="시트러스 선호도 (0-5)")
    vanilla: Optional[int] = Field(None, ge=0, le=5, description="바닐라 선호도 (0-5)")
    bergamot: Optional[int] = Field(None, ge=0, le=5, description="베르가못 선호도 (0-5)")
    cedar: Optional[int] = Field(None, ge=0, le=5, description="시더 선호도 (0-5)")
    sandalwood: Optional[int] = Field(None, ge=0, le=5, description="샌달우드 선호도 (0-5)")
    lavender: Optional[int] = Field(None, ge=0, le=5, description="라벤더 선호도 (0-5)")

    def to_dict(self) -> Dict[str, int]:
        """노트 스코어를 딕셔너리로 변환 (None 값 제외)"""
        return {k: v for k, v in self.dict().items() if v is not None}


class SecondRecommendRequest(BaseModel):
    """2차 추천 요청 스키마"""

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

    emotion_proba: List[float] = Field(
        ...,
        description="6개 감정 클러스터별 확률 배열 (AI 모델 출력)",
        min_items=6,
        max_items=6,
        example=[0.01, 0.03, 0.85, 0.02, 0.05, 0.04]
    )

    selected_idx: List[int] = Field(
        ...,
        description="1차 추천에서 선택된 향수 인덱스 목록",
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
        if len(v) != 6:
            raise ValueError("emotion_proba는 정확히 6개의 확률값을 가져야 합니다.")

        total = sum(v)
        if not (0.95 <= total <= 1.05):  # 소수점 오차 허용
            raise ValueError(f"emotion_proba의 합은 1.0에 가까워야 합니다. 현재: {total}")

        for prob in v:
            if not (0.0 <= prob <= 1.0):
                raise ValueError("각 확률값은 0.0-1.0 사이여야 합니다.")

        return v

    @validator('selected_idx')
    def validate_selected_idx(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("selected_idx에 중복된 인덱스가 있습니다.")

        for idx in v:
            if idx < 0:
                raise ValueError("인덱스는 0 이상이어야 합니다.")

        return v


class SecondRecommendItem(BaseModel):
    """2차 추천 결과 아이템"""

    name: str = Field(..., description="향수 이름")
    brand: str = Field(..., description="브랜드명")
    final_score: float = Field(..., description="최종 추천 점수 (0.0-1.0)", ge=0.0, le=1.0)
    emotion_cluster: int = Field(..., description="감정 클러스터 ID (0-5)", ge=0, le=5)


class SecondRecommendResponse(BaseModel):
    """2차 추천 응답"""

    recommendations: List[SecondRecommendItem] = Field(
        ...,
        description="2차 추천 향수 목록 (점수 내림차순)"
    )

    metadata: Dict = Field(
        ...,
        description="추천 메타데이터"
    )


# ─── 3. 감정 클러스터 매핑 ─────────────────────────────────────────────────────────
EMOTION_CLUSTER_MAP = {
    0: "차분한, 편안한",
    1: "자신감, 신선함",
    2: "우아함, 친근함",
    3: "순수함, 친근함",
    4: "신비로운, 매력적",
    5: "활기찬, 에너지"
}


# ─── 4. 노트 분석 유틸리티 함수들 ─────────────────────────────────────────────────
def parse_notes_from_string(notes_str: str) -> List[str]:
    """
    노트 문자열을 파싱하여 개별 노트 리스트로 변환
    """
    if not notes_str or pd.isna(notes_str):
        return []

    # 콤마로 분리하고 앞뒤 공백 제거, 소문자 변환
    notes = [note.strip().lower() for note in str(notes_str).split(',')]

    # 빈 문자열 제거
    notes = [note for note in notes if note and note != '']

    return notes


def normalize_note_name(note: str) -> str:
    """
    노트명을 정규화 (유사한 노트들을 매칭하기 위해)
    """
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
    """
    향수의 노트와 사용자 선호도를 비교하여 매칭 점수 계산
    """
    if not perfume_notes or not user_note_scores:
        return 0.0

    # 향수 노트를 정규화
    normalized_perfume_notes = [normalize_note_name(note) for note in perfume_notes]

    total_score = 0.0
    matched_notes_count = 0
    total_preference_weight = sum(user_note_scores.values())

    if total_preference_weight == 0:
        return 0.0

    logger.debug(f"🔍 향수 노트: {normalized_perfume_notes}")
    logger.debug(f"🔍 사용자 선호도: {user_note_scores}")

    for user_note, preference_score in user_note_scores.items():
        normalized_user_note = normalize_note_name(user_note)

        # 정확한 매칭
        if normalized_user_note in normalized_perfume_notes:
            # 선호도 점수를 0-1 범위로 정규화 (5점 만점)
            normalized_preference = preference_score / 5.0

            # 가중치 적용 (해당 노트의 선호도가 전체에서 차지하는 비중)
            weight = preference_score / total_preference_weight

            contribution = normalized_preference * weight
            total_score += contribution
            matched_notes_count += 1

            logger.debug(f"  ✅ 매칭: {normalized_user_note} (선호도: {preference_score}/5, 기여도: {contribution:.3f})")

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

                logger.debug(f"  🔸 부분 매칭: {normalized_user_note} → {partial_matches} (기여도: {contribution:.3f})")

    # 매칭된 노트가 없으면 0점
    if matched_notes_count == 0:
        return 0.0

    # 매칭 비율에 따른 보너스 (많은 노트가 매칭될수록 추가 점수)
    match_ratio = matched_notes_count / len(user_note_scores)
    match_bonus = match_ratio * 0.1  # 최대 10% 보너스

    final_score = min(1.0, total_score + match_bonus)

    logger.debug(f"  📊 매칭 결과: {matched_notes_count}/{len(user_note_scores)}개 노트, 최종 점수: {final_score:.3f}")

    return final_score


def calculate_emotion_cluster_weight(perfume_cluster: int, emotion_proba: List[float]) -> float:
    """
    향수의 감정 클러스터와 사용자의 감정 확률 분포를 기반으로 가중치 계산
    """
    if perfume_cluster < 0 or perfume_cluster >= len(emotion_proba):
        logger.warning(f"⚠️ 잘못된 클러스터 ID: {perfume_cluster}")
        return 0.1  # 최소 가중치

    # 해당 클러스터의 확률을 가중치로 사용
    cluster_weight = emotion_proba[perfume_cluster]

    # 너무 낮은 가중치는 최소값으로 보정 (완전히 배제하지 않음)
    cluster_weight = max(0.05, cluster_weight)

    logger.debug(f"📊 클러스터 {perfume_cluster} 가중치: {cluster_weight:.3f}")

    return cluster_weight


def calculate_final_score(
        note_match_score: float,
        emotion_cluster_weight: float,
        diversity_bonus: float = 0.0
) -> float:
    """
    최종 추천 점수 계산
    """
    # 노트 매칭 점수 70%, 감정 클러스터 가중치 25%, 다양성 보너스 5%
    final_score = (
            note_match_score * 0.70 +
            emotion_cluster_weight * 0.25 +
            diversity_bonus * 0.05
    )

    # 0.0 ~ 1.0 범위로 정규화
    final_score = max(0.0, min(1.0, final_score))

    return final_score


# ─── 5. 메인 추천 함수 ─────────────────────────────────────────────────────────
def process_second_recommendation(
        user_note_scores: Dict[str, int],
        emotion_proba: List[float],
        selected_idx: List[int]
) -> List[Dict]:
    """
    2차 추천 처리 메인 함수
    """
    start_time = datetime.now()

    logger.info(f"🎯 2차 추천 처리 시작")
    logger.info(f"  📝 사용자 노트 선호도: {user_note_scores}")
    logger.info(f"  🧠 감정 확률 분포: {[f'{p:.3f}' for p in emotion_proba]}")
    logger.info(f"  📋 선택된 인덱스: {selected_idx} (총 {len(selected_idx)}개)")

    # 선택된 인덱스에 해당하는 향수들 필터링
    valid_indices = [idx for idx in selected_idx if idx < len(df)]
    invalid_indices = [idx for idx in selected_idx if idx >= len(df)]

    if invalid_indices:
        logger.warning(f"⚠️ 잘못된 인덱스들: {invalid_indices} (데이터셋 크기: {len(df)})")

    if not valid_indices:
        raise ValueError("유효한 향수 인덱스가 없습니다.")

    selected_perfumes = df.iloc[valid_indices].copy()
    logger.info(f"✅ {len(selected_perfumes)}개 향수 선택됨")

    # 각 향수에 대한 점수 계산
    results = []
    brand_count = {}  # 브랜드별 개수 (다양성 보너스용)

    for idx, (_, row) in enumerate(selected_perfumes.iterrows()):
        try:
            # 향수 기본 정보
            perfume_name = str(row['name'])
            perfume_brand = str(row['brand'])
            perfume_cluster = int(row.get('emotion_cluster', 0))
            perfume_notes_str = str(row.get('notes', ''))

            # 노트 파싱
            perfume_notes = parse_notes_from_string(perfume_notes_str)

            # 1. 노트 매칭 점수 계산
            note_match_score = calculate_note_match_score(perfume_notes, user_note_scores)

            # 2. 감정 클러스터 가중치 계산
            emotion_weight = calculate_emotion_cluster_weight(perfume_cluster, emotion_proba)

            # 3. 다양성 보너스 계산 (같은 브랜드가 많으면 감점)
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
                'note_match_score': round(note_match_score, 3),
                'emotion_weight': round(emotion_weight, 3),
                'diversity_bonus': round(diversity_bonus, 3),
                'perfume_notes': perfume_notes,
                'original_index': valid_indices[idx]
            }

            results.append(result_item)

            logger.debug(f"📊 {perfume_name}: 노트매칭={note_match_score:.3f}, "
                         f"감정가중치={emotion_weight:.3f}, 다양성={diversity_bonus:.3f}, "
                         f"최종점수={final_score:.3f}")

        except Exception as e:
            logger.error(f"❌ 향수 '{row.get('name', 'Unknown')}' 처리 중 오류: {e}")
            continue

    # 최종 점수 기준으로 정렬
    results.sort(key=lambda x: x['final_score'], reverse=True)

    processing_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"✅ 2차 추천 처리 완료: {len(results)}개 향수 (소요시간: {processing_time:.3f}초)")

    if results:
        top_scores = [r['final_score'] for r in results[:5]]
        logger.info(f"📊 상위 5개 점수: {top_scores}")
        logger.info(f"📊 점수 범위: {results[-1]['final_score']:.3f} ~ {results[0]['final_score']:.3f}")

    return results


# ─── 6. 라우터 설정 ─────────────────────────────────────────────────────────────
router = APIRouter(prefix="/perfumes", tags=["Second Recommendation"])


@router.post(
    "/recommend-2nd",
    response_model=List[SecondRecommendItem],
    summary="2차 향수 추천 - 사용자 노트 선호도 기반",
    description=(
            "🎯 **2차 향수 추천 API**\n\n"
            "1차 추천 결과와 사용자의 노트 선호도를 기반으로 정밀한 2차 추천을 제공합니다.\n\n"
            "**📥 입력 정보:**\n"
            "- `user_note_scores`: 사용자의 노트별 선호도 점수 (0-5)\n"
            "- `emotion_proba`: AI 모델이 예측한 6개 감정 클러스터 확률\n"
            "- `selected_idx`: 1차 추천에서 선택된 향수 인덱스 목록\n\n"
            "**📤 출력 정보:**\n"
            "- 향수별 최종 추천 점수와 감정 클러스터 정보\n"
            "- 점수 기준 내림차순 정렬\n\n"
            "**🧮 점수 계산 방식:**\n"
            "- 노트 매칭 점수 (70%): 사용자 선호 노트와 향수 노트의 일치도\n"
            "- 감정 클러스터 가중치 (25%): AI 예측 확률 기반\n"
            "- 다양성 보너스 (5%): 브랜드 다양성 고려\n\n"
            "**✨ 특징:**\n"
            "- 정확한 노트 매칭 + 부분 매칭 지원\n"
            "- 노트명 정규화로 유사 노트 매칭\n"
            "- 브랜드 다양성 보장\n"
            "- 상세한 점수 분석 제공"
    )
)
def recommend_second_perfumes(request: SecondRecommendRequest):
    """2차 향수 추천 API"""

    request_start_time = datetime.now()

    logger.info(f"🆕 2차 향수 추천 요청 접수")
    logger.info(f"  📊 노트 선호도 개수: {len(request.user_note_scores)}개")
    logger.info(
        f"  🧠 감정 확률 최고: {max(request.emotion_proba):.3f} (클러스터 {request.emotion_proba.index(max(request.emotion_proba))})")
    logger.info(f"  📋 선택된 향수: {len(request.selected_idx)}개")

    try:
        # 메인 추천 처리
        results = process_second_recommendation(
            user_note_scores=request.user_note_scores,
            emotion_proba=request.emotion_proba,
            selected_idx=request.selected_idx
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail="추천할 수 있는 향수가 없습니다."
            )

        # 응답 형태로 변환 (상세 정보 제거)
        response_items = []
        for result in results:
            response_items.append(
                SecondRecommendItem(
                    name=result['name'],
                    brand=result['brand'],
                    final_score=result['final_score'],
                    emotion_cluster=result['emotion_cluster']
                )
            )

        # 처리 시간 계산
        total_processing_time = (datetime.now() - request_start_time).total_seconds()

        # 통계 정보
        note_preferences = list(request.user_note_scores.keys())
        top_emotion_cluster = request.emotion_proba.index(max(request.emotion_proba))

        logger.info(f"✅ 2차 추천 완료: {len(response_items)}개 향수")
        logger.info(f"⏱️ 총 처리 시간: {total_processing_time:.3f}초")
        logger.info(f"📊 최고 점수: {response_items[0].final_score:.3f} ({response_items[0].name})")
        logger.info(f"📊 최저 점수: {response_items[-1].final_score:.3f} ({response_items[-1].name})")

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


# ─── 7. 추가 유틸리티 API들 ─────────────────────────────────────────────────────
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
        for _, row in df.head(100).iterrows():  # 처음 100개만 샘플링
            notes = parse_notes_from_string(str(row.get('notes', '')))
            all_notes.extend(notes)

        note_frequency = Counter(all_notes)
        top_notes = dict(note_frequency.most_common(20))

        return {
            "system_status": "operational",
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
                "노트 선호도 기반 매칭",
                "감정 클러스터 가중치",
                "브랜드 다양성 보장",
                "노트명 정규화",
                "부분 매칭 지원"
            ]
        }

    except Exception as e:
        logger.error(f"❌ 시스템 상태 확인 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"시스템 상태 확인 중 오류가 발생했습니다: {str(e)}"
        )