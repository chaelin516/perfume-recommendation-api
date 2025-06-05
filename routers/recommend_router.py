import os
import pickle
import logging
import random
import sys
import numpy as np
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

# ✅ schemas/recommend.py에서 스키마 임포트
from schemas.recommend import (
    RecommendRequest,
    RecommendedPerfume,
    RecommendResponse,
    ClusterRecommendResponse,  # 🆕 새로운 스키마
    SUPPORTED_CATEGORIES,
    EMOTION_CLUSTER_DESCRIPTIONS,
    validate_request_categories,
    map_single_to_combined_impression
)

# ─── 로거 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("recommend_router")

# ─── 1. perfume_final_dataset.csv 로드 ───────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
try:
    df = pd.read_csv(DATA_PATH)
    df.fillna("", inplace=True)
    logger.info(f"✅ Perfume dataset loaded: {df.shape[0]} rows")
    logger.info(f"📋 Available columns: {list(df.columns)}")

    # ✅ 컬럼 존재 여부 확인
    required_columns = ['name', 'brand', 'image_url', 'notes']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"❌ Missing required columns: {missing_columns}")
        raise RuntimeError(f"Missing required columns: {missing_columns}")

    # ✅ emotion 관련 컬럼 확인
    if 'desired_impression' in df.columns:
        logger.info("✅ Using 'desired_impression' column for emotion data")
    if 'emotion_cluster' in df.columns:
        logger.info("✅ Using 'emotion_cluster' column for cluster data")
        # emotion_cluster 컬럼 정수형으로 변환
        df['emotion_cluster'] = pd.to_numeric(df['emotion_cluster'], errors='coerce').fillna(0).astype(int)
        logger.info(f"📊 Emotion clusters: {sorted(df['emotion_cluster'].unique())}")
    else:
        logger.warning("⚠️ No emotion_cluster column found")

    # 📊 데이터 샘플 로그
    if len(df) > 0:
        sample_row = df.iloc[0]
        logger.info(f"📝 Sample data: {sample_row['name']} by {sample_row['brand']}")

except Exception as e:
    logger.error(f"❌ perfume_final_dataset.csv 로드 중 오류: {e}")
    raise RuntimeError(f"perfume_final_dataset.csv 로드 중 오류: {e}")

# ─── 2. 모델 파일 경로 설정 ─────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/final_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "../models/encoder.pkl")

# ─── 3. 전역 변수 및 상태 관리 ─────────────────────────────────────
_model = None
_encoder = None
_model_available = False
_fallback_encoder = None

# ─── 4. 감정 클러스터 매핑 ─────────────────────────────────────
EMOTION_CLUSTER_MAP = {
    0: "차분한, 편안한",
    1: "자신감, 신선함",
    2: "우아함, 친근함",
    3: "순수함, 친근함",
    4: "신비로운, 매력적",
    5: "활기찬, 에너지"
}

# ✅ encoder.pkl과 호환되는 카테고리 매핑
API_TO_MODEL_MAPPING = {
    "gender": {
        "men": "men",
        "unisex": "unisex",
        "women": "women"
    },
    "season_tags": {
        "fall": "fall",
        "spring": "spring",
        "summer": "summer",
        "winter": "winter"
    },
    "time_tags": {
        "day": "day",
        "night": "night"
    },
    "desired_impression": {
        "confident, fresh": "confident, fresh",
        "confident, mysterious": "confident, mysterious",
        "elegant, friendly": "elegant, friendly",
        "pure, friendly": "pure, friendly"
    },
    "activity": {
        "casual": "casual",
        "date": "date",
        "work": "work"
    },
    "weather": {
        "any": "any",
        "cold": "cold",
        "hot": "hot",
        "rainy": "rainy"
    }
}


# ─── 5. 🆕 노트 분석 유틸리티 함수들 ─────────────────────────────────────
def parse_notes_from_string(notes_str: str) -> List[str]:
    """
    노트 문자열을 파싱하여 개별 노트 리스트로 변환

    Args:
        notes_str: "bergamot, jasmine, white musk, amber" 형태의 문자열

    Returns:
        개별 노트들의 리스트
    """
    if not notes_str or pd.isna(notes_str):
        return []

    # 콤마로 분리하고 앞뒤 공백 제거
    notes = [note.strip().lower() for note in str(notes_str).split(',')]

    # 빈 문자열 제거
    notes = [note for note in notes if note and note != '']

    return notes


def get_top_notes_from_cluster(cluster_perfumes: pd.DataFrame, top_k: int = 15) -> List[str]:
    """
    클러스터에 속한 향수들의 노트를 분석하여 상위 K개 노트 반환

    Args:
        cluster_perfumes: 클러스터에 속한 향수들의 DataFrame
        top_k: 반환할 상위 노트 개수

    Returns:
        빈도순으로 정렬된 상위 K개 노트 리스트
    """
    all_notes = []

    for _, row in cluster_perfumes.iterrows():
        notes = parse_notes_from_string(row.get('notes', ''))
        all_notes.extend(notes)

    if not all_notes:
        # 노트가 없으면 일반적인 향수 노트 반환
        return [
                   "bergamot", "jasmine", "rose", "vanilla", "sandalwood",
                   "cedar", "musk", "amber", "lavender", "citrus",
                   "woody", "floral", "fresh", "sweet", "spicy"
               ][:top_k]

    # 빈도 계산 및 상위 K개 선택
    note_counter = Counter(all_notes)
    top_notes = [note for note, count in note_counter.most_common(top_k)]

    logger.info(f"📊 클러스터 노트 분석: 총 {len(all_notes)}개 노트 → 상위 {len(top_notes)}개 선택")
    logger.info(f"📊 상위 5개 노트: {top_notes[:5]}")

    return top_notes


def get_perfume_indices(cluster_perfumes: pd.DataFrame, top_k: int = 10) -> List[int]:
    """
    추천 향수들의 원본 DataFrame 인덱스 반환

    Args:
        cluster_perfumes: 클러스터에 속한 향수들의 DataFrame
        top_k: 반환할 인덱스 개수

    Returns:
        원본 데이터셋에서의 인덱스 리스트
    """
    # 점수가 있으면 점수 기준으로 정렬, 없으면 원본 순서 유지
    if 'score' in cluster_perfumes.columns:
        sorted_perfumes = cluster_perfumes.nlargest(top_k, 'score')
    else:
        sorted_perfumes = cluster_perfumes.head(top_k)

    indices = sorted_perfumes.index.tolist()

    logger.info(f"📋 선택된 향수 인덱스: {indices}")

    return indices


# ─── 6. 모델 가용성 확인 (31KB 모델에 맞게 수정) ─────────────────────────────────────
def check_model_availability():
    """모델 파일들의 가용성을 확인합니다."""
    global _model_available

    logger.info("🔍 모델 파일 가용성 확인 중...")

    try:
        # 파일 존재 및 크기 확인
        model_exists = os.path.exists(MODEL_PATH)
        encoder_exists = os.path.exists(ENCODER_PATH)

        model_valid = False
        encoder_valid = False

        if model_exists:
            model_size = os.path.getsize(MODEL_PATH)
            # ✅ 실제 모델 파일 크기에 맞게 수정: 31KB 모델이므로 10KB 이상으로 체크
            model_valid = model_size > 10000  # 10KB 이상 (기존: 100KB)
            logger.info(f"📄 모델 파일: {model_size:,}B ({model_size / 1024:.1f}KB) {'✅' if model_valid else '❌'}")
        else:
            logger.warning(f"⚠️ 모델 파일이 없습니다: {MODEL_PATH}")

        if encoder_exists:
            encoder_size = os.path.getsize(ENCODER_PATH)
            # ✅ 인코더는 1KB이므로 500B 이상으로 체크 (기존: 100B)
            encoder_valid = encoder_size > 500  # 500B 이상
            logger.info(f"📄 인코더 파일: {encoder_size:,}B ({encoder_size}B) {'✅' if encoder_valid else '❌'}")
        else:
            logger.warning(f"⚠️ 인코더 파일이 없습니다: {ENCODER_PATH}")

        _model_available = model_valid and encoder_valid

        logger.info(f"🤖 모델 가용성: {'✅ 사용 가능' if _model_available else '❌ 사용 불가'}")

        if _model_available:
            logger.info(f"✨ 모델 사용 준비 완료 - 크기: {model_size / 1024:.1f}KB")
        else:
            if not model_valid:
                logger.warning(f"⚠️ 모델 파일 크기 부족: {model_size}B (최소 10KB 필요)")
            if not encoder_valid:
                logger.warning(f"⚠️ 인코더 파일 크기 부족: {encoder_size}B (최소 500B 필요)")

        return _model_available

    except Exception as e:
        logger.error(f"❌ 모델 가용성 확인 중 오류: {e}")
        _model_available = False
        return False


# ─── 7. 모델 로딩 함수들 (Keras 3.8.0 호환 및 크기 체크 수정) ─────────────────────────────────────
def get_model():
    """Keras 감정 클러스터 모델을 로드합니다."""
    global _model

    if _model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                logger.warning(f"⚠️ 모델 파일이 없습니다: {MODEL_PATH}")
                return None

            # ✅ 파일 크기 확인 - 31KB 모델에 맞게 수정
            model_size = os.path.getsize(MODEL_PATH)
            if model_size < 10000:  # 10KB 미만 (기존: 100KB)
                logger.warning(f"⚠️ 모델 파일이 너무 작습니다: {model_size} bytes ({model_size / 1024:.1f}KB)")
                return None

            logger.info(f"📦 모델 파일 크기 확인 완료: {model_size:,}B ({model_size / 1024:.1f}KB)")

            # TensorFlow 동적 임포트 및 Keras 3.x 호환성 고려
            try:
                tf_start = datetime.now()

                # ✅ Keras 3.x 지원을 위한 임포트 방식 개선
                try:
                    # Keras 3.x 방식 시도
                    import tensorflow as tf
                    from tensorflow import keras
                    load_model = keras.models.load_model
                    logger.info(f"📦 TensorFlow {tf.__version__} + Keras 3.x 스타일 로딩")
                except:
                    # 기존 방식 폴백
                    from tensorflow.keras.models import load_model
                    logger.info(f"📦 TensorFlow 기존 스타일 로딩")

                tf_load_time = (datetime.now() - tf_start).total_seconds()

                logger.info(f"📦 Keras 모델 로딩 시도 (TF 로딩: {tf_load_time:.3f}초)")
                logger.info(f"📊 예상 모델 구조: 입력(6) → Dense(64,relu) → Dense(6,softmax)")

                model_start = datetime.now()

                # ✅ compile=False로 빠른 로딩, Keras 3.x 호환
                _model = load_model(MODEL_PATH, compile=False)
                model_load_time = (datetime.now() - model_start).total_seconds()

                # ✅ 모델 구조 검증
                logger.info(f"✅ Keras 모델 로드 성공 (모델 로딩: {model_load_time:.3f}초)")
                logger.info(f"📊 실제 모델 입력 shape: {_model.input_shape}")
                logger.info(f"📊 실제 모델 출력 shape: {_model.output_shape}")

                # ✅ 레이어 정보 출력
                logger.info(f"📊 모델 레이어 수: {len(_model.layers)}")
                for i, layer in enumerate(_model.layers):
                    layer_info = f"  Layer {i + 1}: {layer.__class__.__name__}"
                    if hasattr(layer, 'units'):
                        layer_info += f" (units: {layer.units})"
                    if hasattr(layer, 'activation'):
                        layer_info += f" (activation: {layer.activation.__name__})"
                    logger.info(layer_info)

                # ✅ 출력 크기 검증 (6개 감정 클러스터)
                output_size = _model.output_shape[-1]
                if output_size == 6:
                    logger.info("🎯 6개 감정 클러스터 분류 모델로 확인됨")
                else:
                    logger.warning(f"⚠️ 예상과 다른 출력 크기: {output_size} (예상: 6)")

                # ✅ 입력 크기 검증 (6개 특성)
                input_size = _model.input_shape[-1]
                if input_size == 6:
                    logger.info("🎯 6개 입력 특성 모델로 확인됨")
                else:
                    logger.warning(f"⚠️ 예상과 다른 입력 크기: {input_size} (예상: 6)")

                # ✅ 간단한 테스트 추론 (모델 동작 확인)
                try:
                    test_input = np.random.random((1, 6)).astype(np.float32)
                    test_output = _model.predict(test_input, verbose=0)
                    logger.info(f"🧪 테스트 추론 성공: 입력{test_input.shape} → 출력{test_output.shape}")
                    logger.info(f"🧪 출력 합계: {test_output.sum():.3f} (softmax이면 ~1.0)")

                    # 가장 높은 확률의 클러스터 확인
                    predicted_cluster = int(np.argmax(test_output[0]))
                    confidence = float(test_output[0][predicted_cluster])
                    logger.info(f"🧪 테스트 예측: 클러스터 {predicted_cluster} (신뢰도: {confidence:.3f})")
                except Exception as test_e:
                    logger.warning(f"⚠️ 테스트 추론 실패: {test_e}")

            except ImportError as e:
                logger.error(f"❌ TensorFlow를 찾을 수 없습니다: {e}")
                return None
            except Exception as e:
                logger.error(f"❌ Keras 모델 로드 실패: {e}")
                logger.error(f"  파일 경로: {MODEL_PATH}")
                logger.error(f"  파일 크기: {model_size}B")
                return None

        except Exception as e:
            logger.error(f"❌ 모델 로딩 중 예외: {e}")
            return None

    return _model


def get_saved_encoder():
    """저장된 encoder.pkl을 로드합니다."""
    global _encoder

    if _encoder is None:
        try:
            if not os.path.exists(ENCODER_PATH):
                logger.warning(f"⚠️ 인코더 파일이 없습니다: {ENCODER_PATH}")
                return None

            encoder_size = os.path.getsize(ENCODER_PATH)
            logger.info(f"📦 인코더 로딩 시도: {ENCODER_PATH} ({encoder_size}B)")

            with open(ENCODER_PATH, "rb") as f:
                _encoder = pickle.load(f)
            logger.info("✅ encoder.pkl 로드 성공")

        except Exception as e:
            logger.error(f"❌ encoder.pkl 로드 실패: {e}")
            return None

    return _encoder


def get_fallback_encoder():
    """✅ encoder.pkl과 호환되는 Fallback OrdinalEncoder를 생성합니다."""
    global _fallback_encoder

    if _fallback_encoder is None:
        try:
            logger.info("🔧 Fallback OrdinalEncoder 생성 중...")

            # ✅ encoder.pkl과 동일한 카테고리 정의
            from sklearn.preprocessing import OrdinalEncoder

            CATEGORIES = [
                ["men", "unisex", "women"],  # gender
                ["fall", "spring", "summer", "winter"],  # season_tags
                ["day", "night"],  # time_tags
                ["confident, fresh", "confident, mysterious", "elegant, friendly", "pure, friendly"],
                # desired_impression
                ["casual", "date", "work"],  # activity
                ["any", "cold", "hot", "rainy"]  # weather
            ]

            # ✅ OrdinalEncoder 생성 (encoder.pkl과 동일한 타입)
            _fallback_encoder = OrdinalEncoder(
                categories=CATEGORIES,
                handle_unknown="error"
            )

            # 더미 데이터로 fit (encoder.pkl과 완전 일치)
            dummy_data = [
                ["men", "fall", "day", "confident, fresh", "casual", "any"],
                ["unisex", "spring", "night", "confident, mysterious", "date", "cold"],
                ["women", "summer", "day", "elegant, friendly", "work", "hot"],
                ["men", "winter", "night", "pure, friendly", "casual", "rainy"]
            ]

            _fallback_encoder.fit(dummy_data)
            logger.info("✅ Fallback OrdinalEncoder 생성 및 훈련 완료")

            # ✅ 인코더 검증 테스트
            test_input = ["women", "spring", "day", "confident, fresh", "casual", "hot"]
            test_encoded = _fallback_encoder.transform([test_input])
            logger.info(f"🧪 Fallback 인코더 테스트 성공: 입력 6개 → 출력 {test_encoded.shape[1]}개")

        except Exception as e:
            logger.error(f"❌ Fallback encoder 생성 실패: {e}")
            return None

    return _fallback_encoder


def safe_transform_input(raw_features: list) -> np.ndarray:
    """✅ 안전한 입력 변환 함수"""
    try:
        # 1. 저장된 인코더 시도
        encoder = get_saved_encoder()
        if encoder:
            try:
                logger.info(f"🔍 저장된 인코더로 변환 시도: {raw_features}")
                transformed = encoder.transform([raw_features])
                logger.info(f"✅ 저장된 인코더 변환 성공: {transformed.shape}")
                return transformed
            except Exception as e:
                logger.warning(f"⚠️ 저장된 인코더 실패: {e}")

        # 2. Fallback 인코더 시도
        fallback_encoder = get_fallback_encoder()
        if fallback_encoder:
            logger.info(f"🔄 Fallback 인코더로 변환: {raw_features}")
            transformed = fallback_encoder.transform([raw_features])
            logger.info(f"✅ Fallback 인코더 변환 성공: {transformed.shape}")
            return transformed
        else:
            raise Exception("Fallback 인코더 생성 실패")

    except Exception as e:
        logger.error(f"❌ 입력 변환 완전 실패: {e}")
        raise e


# ─── 8. 🆕 클러스터 기반 추천 함수 ─────────────────────────────────────
def predict_cluster_recommendation(request_dict: dict) -> Dict[str, Any]:
    """
    ✅ 클러스터 기반 추천 - 새로운 응답 형태

    Returns:
        {
            "cluster": int,
            "description": str,
            "proba": List[float],
            "recommended_notes": List[str],
            "selected_idx": List[int]
        }
    """
    try:
        start_time = datetime.now()

        # 모델 가져오기
        model = get_model()
        if model is None:
            raise Exception("모델 로드 실패")

        # ✅ API 입력을 모델 호환 형식으로 변환
        raw_features = [
            request_dict["gender"],
            request_dict["season_tags"],
            request_dict["time_tags"],
            request_dict["desired_impression"],
            request_dict["activity"],
            request_dict["weather"]
        ]

        logger.info(f"🔮 클러스터 추천 입력 데이터: {raw_features}")

        # ✅ 안전한 입력 변환 사용
        x_input = safe_transform_input(raw_features)
        logger.info(f"🔮 감정 클러스터 예측 시작 (입력 shape: {x_input.shape})")

        # 모델 예측 (감정 클러스터)
        preds = model.predict(x_input, verbose=0)  # (1, 6) 출력
        cluster_probabilities = preds[0]  # [0.1, 0.8, 0.05, 0.02, 0.02, 0.01]
        predicted_cluster = int(np.argmax(cluster_probabilities))  # 가장 높은 확률의 클러스터
        confidence = float(cluster_probabilities[predicted_cluster])

        # 클러스터 설명
        cluster_description = EMOTION_CLUSTER_MAP.get(predicted_cluster, f"클러스터 {predicted_cluster}")

        logger.info(f"🎯 예측된 감정 클러스터: {predicted_cluster} ({cluster_description}) - 신뢰도: {confidence:.3f}")

        # 모든 클러스터 확률 로그
        for i, prob in enumerate(cluster_probabilities):
            cluster_desc = EMOTION_CLUSTER_MAP.get(i, f"클러스터 {i}")
            logger.info(f"  클러스터 {i} ({cluster_desc}): {prob:.3f}")

        # 감정 클러스터에 해당하는 향수 필터링
        if 'emotion_cluster' in df.columns:
            cluster_perfumes = df[df['emotion_cluster'] == predicted_cluster].copy()
            logger.info(f"📋 클러스터 {predicted_cluster} 향수 개수: {len(cluster_perfumes)}개")
        else:
            logger.warning("⚠️ emotion_cluster 컬럼이 없어 전체 데이터 사용")
            cluster_perfumes = df.copy()

        # 클러스터에 해당하는 향수가 없으면 대체 클러스터 사용
        if cluster_perfumes.empty:
            logger.warning(f"⚠️ 클러스터 {predicted_cluster}에 해당하는 향수가 없음")
            # 두 번째로 높은 확률의 클러스터 찾기
            second_best = int(np.argsort(cluster_probabilities)[-2])
            cluster_perfumes = df[df['emotion_cluster'] == second_best].copy()
            predicted_cluster = second_best
            confidence = float(cluster_probabilities[second_best])
            cluster_description = EMOTION_CLUSTER_MAP.get(predicted_cluster, f"클러스터 {predicted_cluster}")
            logger.info(f"📋 대체 클러스터 {second_best} 사용: {len(cluster_perfumes)}개")

        # ✅ 추가 필터링 (성별, 계절 등)
        original_count = len(cluster_perfumes)

        # 성별 필터링
        if 'gender' in cluster_perfumes.columns:
            gender_filtered = cluster_perfumes[
                cluster_perfumes['gender'] == request_dict["gender"]
                ]
            if not gender_filtered.empty:
                cluster_perfumes = gender_filtered
                logger.info(f"  성별 '{request_dict['gender']}' 필터링: {original_count} → {len(cluster_perfumes)}개")

        # 계절 필터링
        if 'season_tags' in cluster_perfumes.columns:
            season_filtered = cluster_perfumes[
                cluster_perfumes['season_tags'].str.contains(
                    request_dict["season_tags"], na=False, case=False
                )
            ]
            if not season_filtered.empty:
                cluster_perfumes = season_filtered
                logger.info(f"  계절 '{request_dict['season_tags']}' 필터링: → {len(cluster_perfumes)}개")

        # 시간 필터링
        if 'time_tags' in cluster_perfumes.columns:
            time_filtered = cluster_perfumes[
                cluster_perfumes['time_tags'].str.contains(
                    request_dict["time_tags"], na=False, case=False
                )
            ]
            if not time_filtered.empty:
                cluster_perfumes = time_filtered
                logger.info(f"  시간 '{request_dict['time_tags']}' 필터링: → {len(cluster_perfumes)}개")

        # ✅ 상위 15개 노트 추출
        recommended_notes = get_top_notes_from_cluster(cluster_perfumes, top_k=15)

        # ✅ 상위 10개 향수 인덱스 추출
        selected_indices = get_perfume_indices(cluster_perfumes, top_k=10)

        # 처리 시간 계산
        processing_time = (datetime.now() - start_time).total_seconds()

        # ✅ 새로운 형태의 응답 구성
        result = {
            "cluster": predicted_cluster,
            "description": cluster_description,
            "proba": [round(float(prob), 4) for prob in cluster_probabilities],  # 소수점 4자리로 반올림
            "recommended_notes": recommended_notes,
            "selected_idx": selected_indices,
            "metadata": {
                "processing_time_seconds": round(processing_time, 3),
                "total_cluster_perfumes": len(cluster_perfumes),
                "confidence": round(confidence, 3),
                "method": "AI 감정 클러스터 모델",
                "filters_applied": {
                    "gender": request_dict["gender"],
                    "season": request_dict["season_tags"],
                    "time": request_dict["time_tags"]
                }
            }
        }

        logger.info(f"✅ 클러스터 기반 추천 완료: 클러스터 {predicted_cluster} (소요시간: {processing_time:.3f}초)")

        return result

    except Exception as e:
        logger.error(f"❌ 클러스터 기반 추천 실패: {e}")
        raise e


# ─── 9. AI 감정 클러스터 모델 추천 (기존 유지) ─────────────────────────────────────
def predict_with_emotion_cluster_model(request_dict: dict) -> pd.DataFrame:
    """✅ 수정된 감정 클러스터 모델을 사용한 AI 추천 (기존 방식 유지)"""

    try:
        # 모델 가져오기
        model = get_model()
        if model is None:
            raise Exception("모델 로드 실패")

        # ✅ API 입력을 모델 호환 형식으로 변환
        raw_features = [
            request_dict["gender"],
            request_dict["season_tags"],
            request_dict["time_tags"],
            request_dict["desired_impression"],
            request_dict["activity"],
            request_dict["weather"]
        ]

        logger.info(f"🔮 AI 모델 입력 데이터: {raw_features}")

        # ✅ 안전한 입력 변환 사용
        x_input = safe_transform_input(raw_features)
        logger.info(f"🔮 감정 클러스터 예측 시작 (입력 shape: {x_input.shape})")

        # 모델 예측 (감정 클러스터)
        preds = model.predict(x_input, verbose=0)  # (1, 6) 출력
        cluster_probabilities = preds[0]  # [0.1, 0.8, 0.05, 0.02, 0.02, 0.01]
        predicted_cluster = int(np.argmax(cluster_probabilities))  # 가장 높은 확률의 클러스터
        confidence = float(cluster_probabilities[predicted_cluster])

        cluster_name = EMOTION_CLUSTER_MAP.get(predicted_cluster, f"클러스터 {predicted_cluster}")
        logger.info(f"🎯 예측된 감정 클러스터: {predicted_cluster} ({cluster_name}) - 신뢰도: {confidence:.3f}")

        # 모든 클러스터 확률 로그
        for i, prob in enumerate(cluster_probabilities):
            cluster_desc = EMOTION_CLUSTER_MAP.get(i, f"클러스터 {i}")
            logger.info(f"  클러스터 {i} ({cluster_desc}): {prob:.3f}")

        # 감정 클러스터에 해당하는 향수 필터링
        if 'emotion_cluster' in df.columns:
            cluster_perfumes = df[df['emotion_cluster'] == predicted_cluster].copy()
            logger.info(f"📋 클러스터 {predicted_cluster} 향수 개수: {len(cluster_perfumes)}개")
        else:
            logger.warning("⚠️ emotion_cluster 컬럼이 없어 전체 데이터 사용")
            cluster_perfumes = df.copy()

        # 클러스터에 해당하는 향수가 없으면 대체 클러스터 사용
        if cluster_perfumes.empty:
            logger.warning(f"⚠️ 클러스터 {predicted_cluster}에 해당하는 향수가 없음")
            # 두 번째로 높은 확률의 클러스터 찾기
            second_best = int(np.argsort(cluster_probabilities)[-2])
            cluster_perfumes = df[df['emotion_cluster'] == second_best].copy()
            predicted_cluster = second_best
            confidence = float(cluster_probabilities[second_best])
            logger.info(f"📋 대체 클러스터 {second_best} 사용: {len(cluster_perfumes)}개")

        # 추가 필터링 (성별, 계절 등)
        original_count = len(cluster_perfumes)

        # 성별 필터링
        if 'gender' in cluster_perfumes.columns:
            gender_filtered = cluster_perfumes[
                cluster_perfumes['gender'] == request_dict["gender"]
                ]
            if not gender_filtered.empty:
                cluster_perfumes = gender_filtered
                logger.info(f"  성별 '{request_dict['gender']}' 필터링: {original_count} → {len(cluster_perfumes)}개")

        # 계절 필터링
        if 'season_tags' in cluster_perfumes.columns:
            season_filtered = cluster_perfumes[
                cluster_perfumes['season_tags'].str.contains(
                    request_dict["season_tags"], na=False, case=False
                )
            ]
            if not season_filtered.empty:
                cluster_perfumes = season_filtered
                logger.info(f"  계절 '{request_dict['season_tags']}' 필터링: → {len(cluster_perfumes)}개")

        # 시간 필터링
        if 'time_tags' in cluster_perfumes.columns:
            time_filtered = cluster_perfumes[
                cluster_perfumes['time_tags'].str.contains(
                    request_dict["time_tags"], na=False, case=False
                )
            ]
            if not time_filtered.empty:
                cluster_perfumes = time_filtered
                logger.info(f"  시간 '{request_dict['time_tags']}' 필터링: → {len(cluster_perfumes)}개")

        # AI 신뢰도 기반 점수 할당
        cluster_perfumes = cluster_perfumes.copy()

        # 클러스터 신뢰도를 기본 점수로 사용
        base_score = confidence * 0.8  # AI 신뢰도의 80%를 기본 점수로

        scores = []
        for idx, (_, row) in enumerate(cluster_perfumes.iterrows()):
            score = base_score

            # 추가 조건 일치 보너스
            if 'season_tags' in row and request_dict["season_tags"].lower() in str(row['season_tags']).lower():
                score += 0.08
            if 'time_tags' in row and request_dict["time_tags"].lower() in str(row['time_tags']).lower():
                score += 0.06
            if 'desired_impression' in row and request_dict["desired_impression"].lower() in str(
                    row['desired_impression']).lower():
                score += 0.05

            # 브랜드 인기도 보너스
            popular_brands = ['Creed', 'Tom Ford', 'Chanel', 'Dior', 'Jo Malone', 'Diptyque']
            if any(popular in str(row.get('brand', '')) for popular in popular_brands):
                score += 0.03

            # 다양성을 위한 위치 기반 점수 (앞쪽일수록 약간 높은 점수)
            position_bonus = (len(cluster_perfumes) - idx) / len(cluster_perfumes) * 0.05
            score += position_bonus

            # 랜덤 요소 (다양성 확보)
            score += random.uniform(-0.03, 0.05)

            # 정규화 (0.4 ~ 0.95 범위)
            score = max(0.4, min(0.95, score))
            scores.append(score)

        cluster_perfumes['score'] = scores

        # 상위 10개 선택
        top_10 = cluster_perfumes.nlargest(10, 'score')

        logger.info(f"✅ AI 클러스터 모델 추천 완료: {len(top_10)}개")
        if not top_10.empty:
            logger.info(f"📊 점수 범위: {top_10['score'].min():.3f} ~ {top_10['score'].max():.3f}")
            logger.info(f"📊 평균 점수: {top_10['score'].mean():.3f}")

        return top_10

    except Exception as e:
        logger.error(f"❌ AI 클러스터 모델 추천 실패: {e}")
        raise e


# ─── 10. 룰 기반 추천 시스템 (기존 유지) ─────────────────────────────────────
def rule_based_recommendation(request_data: dict, top_k: int = 10) -> List[dict]:
    """룰 기반 향수 추천 시스템 (AI 모델 대체)"""
    logger.info("🎯 룰 기반 추천 시스템 시작")

    try:
        # 필터링 조건
        gender = request_data["gender"]
        season_tags = request_data["season_tags"]
        time_tags = request_data["time_tags"]
        desired_impression = request_data["desired_impression"]
        activity = request_data["activity"]
        weather = request_data["weather"]

        logger.info(f"🔍 필터링 조건: gender={gender}, season_tags={season_tags}, time_tags={time_tags}, "
                    f"desired_impression={desired_impression}, activity={activity}, weather={weather}")

        # 성별 매핑
        gender_map = {"women": "women", "men": "men", "unisex": "unisex"}
        mapped_gender = gender_map.get(gender, "unisex")

        # 1단계: 기본 필터링
        candidates = df.copy()
        original_count = len(candidates)

        # 성별 필터링
        if 'gender' in df.columns:
            gender_filtered = candidates[candidates['gender'] == mapped_gender]
            if not gender_filtered.empty:
                candidates = gender_filtered
                logger.info(f"  성별 '{mapped_gender}' 필터링: {original_count} → {len(candidates)}개")

        # 계절 필터링
        if 'season_tags' in df.columns:
            season_filtered = candidates[
                candidates['season_tags'].str.contains(season_tags, na=False, case=False)
            ]
            if not season_filtered.empty:
                candidates = season_filtered
                logger.info(f"  계절 '{season_tags}' 필터링: → {len(candidates)}개")

        # 시간 필터링
        if 'time_tags' in df.columns:
            time_filtered = candidates[
                candidates['time_tags'].str.contains(time_tags, na=False, case=False)
            ]
            if not time_filtered.empty:
                candidates = time_filtered
                logger.info(f"  시간 '{time_tags}' 필터링: → {len(candidates)}개")

        # 인상 필터링
        if 'desired_impression' in df.columns:
            impression_filtered = candidates[
                candidates['desired_impression'].str.contains(desired_impression, na=False, case=False)
            ]
            if not impression_filtered.empty:
                candidates = impression_filtered
                logger.info(f"  인상 '{desired_impression}' 필터링: → {len(candidates)}개")

        # 활동 필터링 (있는 경우)
        if 'activity' in df.columns:
            activity_filtered = candidates[
                candidates['activity'].str.contains(activity, na=False, case=False)
            ]
            if not activity_filtered.empty:
                candidates = activity_filtered
                logger.info(f"  활동 '{activity}' 필터링: → {len(candidates)}개")

        # 날씨 필터링 (있는 경우)
        if 'weather' in df.columns and weather != 'any':
            weather_filtered = candidates[
                candidates['weather'].str.contains(weather, na=False, case=False)
            ]
            if not weather_filtered.empty:
                candidates = weather_filtered
                logger.info(f"  날씨 '{weather}' 필터링: → {len(candidates)}개")

        # 2단계: 스코어링
        if candidates.empty:
            logger.warning("⚠️ 필터링 결과가 없어 전체 데이터에서 다양성 기반 선택")
            # 다양한 브랜드에서 고르게 선택
            if 'brand' in df.columns:
                unique_brands = df['brand'].unique()
                candidates_list = []
                per_brand = max(1, top_k // len(unique_brands))

                for brand in unique_brands:
                    brand_perfumes = df[df['brand'] == brand].sample(
                        n=min(per_brand, len(df[df['brand'] == brand])),
                        random_state=42
                    )
                    candidates_list.append(brand_perfumes)

                candidates = pd.concat(candidates_list).head(top_k)
            else:
                candidates = df.sample(n=min(top_k, len(df)), random_state=42)

        # 점수 계산
        candidates = candidates.copy()
        scores = []

        # 브랜드별 가중치 (인기 브랜드 예시)
        popular_brands = ['Creed', 'Tom Ford', 'Chanel', 'Dior', 'Jo Malone', 'Diptyque']

        for idx, (_, row) in enumerate(candidates.iterrows()):
            score = 0.3  # 기본 점수

            # 1. 조건 일치도 점수
            brand_name = str(row.get('brand', ''))
            notes_text = str(row.get('notes', ''))

            # 브랜드 인기도 보너스
            if any(popular in brand_name for popular in popular_brands):
                score += 0.15

            # 노트 복잡성 (더 많은 노트 = 더 복잡한 향수)
            note_count = len([n.strip() for n in notes_text.split(',') if n.strip()])
            if note_count >= 8:
                score += 0.10
            elif note_count >= 5:
                score += 0.05

            # 텍스트 매칭 정확도
            impression_match_count = 0
            if 'desired_impression' in row:
                impressions = str(row['desired_impression']).lower().split(',')
                impression_match_count = sum(1 for imp in impressions if desired_impression.lower() in imp.strip())
                score += impression_match_count * 0.08

            # 계절/시간 매칭 정확도
            if 'season_tags' in row:
                season_tags_data = str(row['season_tags']).lower()
                if season_tags.lower() in season_tags_data:
                    score += 0.12 if f' {season_tags.lower()} ' in f' {season_tags_data} ' else 0.08

            if 'time_tags' in row:
                time_tags_data = str(row['time_tags']).lower()
                if time_tags.lower() in time_tags_data:
                    score += 0.12 if f' {time_tags.lower()} ' in f' {time_tags_data} ' else 0.08

            # 활동 매칭
            if 'activity' in row and activity.lower() in str(row['activity']).lower():
                score += 0.08

            # 날씨 매칭
            if 'weather' in row and weather != 'any':
                if weather.lower() in str(row['weather']).lower():
                    score += 0.06
            elif weather == 'any':
                score += 0.03

            # 다양성을 위한 위치 기반 점수
            position_bonus = (len(candidates) - idx) / len(candidates) * 0.05
            score += position_bonus

            # 랜덤 요소 (다양성 확보)
            score += random.uniform(-0.15, 0.15)

            # 점수 정규화 (0.2 ~ 0.95 범위)
            score = max(0.2, min(0.95, score))
            scores.append(score)

        candidates['score'] = scores

        # 상위 K개 선택
        top_candidates = candidates.nlargest(top_k, 'score')

        logger.info(f"✅ 룰 기반 추천 완료: 최종 {len(top_candidates)}개 선택")
        if not top_candidates.empty:
            logger.info(f"📊 점수 범위: {top_candidates['score'].min():.3f} ~ {top_candidates['score'].max():.3f}")
            logger.info(f"📊 평균 점수: {top_candidates['score'].mean():.3f}")

        return top_candidates.to_dict('records')

    except Exception as e:
        logger.error(f"❌ 룰 기반 추천 중 오류: {e}")
        # 최종 안전장치: 완전 랜덤 추천
        logger.info("🎲 완전 랜덤 추천으로 대체")
        random_sample = df.sample(n=min(top_k, len(df)), random_state=42)
        random_sample = random_sample.copy()
        random_sample['score'] = [random.uniform(0.4, 0.7) for _ in range(len(random_sample))]
        return random_sample.to_dict('records')


# ─── 11. 유틸리티 함수들 ─────────────────────────────────────
def get_emotion_text(row):
    """감정 정보를 추출합니다."""
    # 1순위: desired_impression
    if 'desired_impression' in df.columns and pd.notna(row.get('desired_impression')):
        return str(row['desired_impression'])

    # 2순위: emotion_cluster를 텍스트로 변환
    if 'emotion_cluster' in df.columns and pd.notna(row.get('emotion_cluster')):
        cluster_id = int(row['emotion_cluster']) if str(row['emotion_cluster']).isdigit() else 0
        return EMOTION_CLUSTER_MAP.get(cluster_id, "균형잡힌")

    return "다양한 감정"


def get_recommendation_reason(score: float, method: str) -> str:
    """점수와 방법에 따른 추천 이유 생성"""

    if method.startswith("AI"):
        if score >= 0.9:
            return f"🤖 AI가 {score:.1%} 확률로 당신의 완벽한 향수라고 분석했습니다!"
        elif score >= 0.8:
            return f"🤖 AI가 {score:.1%} 확률로 당신에게 잘 맞을 것이라 예측했습니다."
        elif score >= 0.6:
            return f"🤖 AI가 {score:.1%} 확률로 새로운 시도해볼 만한 향수로 추천했습니다."
        else:
            return f"🤖 AI가 {score:.1%} 확률로 색다른 매력을 제안합니다."
    else:
        # 룰 기반 - 더 다양한 메시지
        if score >= 0.9:
            return f"🎯 조건 완벽 일치 (일치도 {score:.1%}) - 이보다 완벽할 순 없어요!"
        elif score >= 0.8:
            return f"⭐ 조건 높은 일치 (일치도 {score:.1%}) - 강력 추천!"
        elif score >= 0.6:
            return f"✨ 조건 적합 (일치도 {score:.1%}) - 고려해보세요!"
        elif score >= 0.4:
            return f"🔍 부분 일치 (일치도 {score:.1%}) - 새로운 발견이 될 수도!"
        else:
            return f"🎲 새로운 스타일 제안 (일치도 {score:.1%}) - 도전해보세요!"


# ─── 12. 레거시 스키마 정의 (하위 호환성) ────────────────────────────────────────────────
class PerfumeRecommendItem(BaseModel):
    name: str
    brand: str
    image_url: str
    notes: str
    emotions: str
    reason: str
    score: Optional[float] = None
    method: Optional[str] = None


# ─── 13. 라우터 설정 ────────────────────────────────────────────────
router = APIRouter(prefix="/perfumes", tags=["Perfume"])

# 시작 시 모델 가용성 확인
logger.info("🚀 추천 시스템 초기화 시작...")
check_model_availability()
if _model_available:
    logger.info("🤖 AI 감정 클러스터 모델 사용 가능")
else:
    logger.info("📋 룰 기반 추천 시스템으로 동작")
logger.info("✅ 추천 시스템 초기화 완료")


# ─── 14. 🆕 새로운 클러스터 기반 추천 API ────────────────────────────────────────────────

@router.post(
    "/recommend-cluster",
    response_model=ClusterRecommendResponse,
    summary="클러스터 기반 향수 추천 (새로운 응답 형태)",
    description=(
            "🆕 **새로운 클러스터 기반 추천 API**\n\n"
            "사용자의 선호도를 기반으로 AI 모델이 감정 클러스터를 예측하고,\n"
            "해당 클러스터의 정보와 추천 향수 인덱스를 반환합니다.\n\n"
            "**🤖 응답 형태:**\n"
            "- `cluster`: 예측된 감정 클러스터 인덱스 (0-5)\n"
            "- `description`: 클러스터 설명 (감정 특성)\n"
            "- `proba`: 6개 클러스터별 softmax 확률 배열\n"
            "- `recommended_notes`: 해당 클러스터의 상위 15개 인기 노트\n"
            "- `selected_idx`: 추천 향수들의 데이터셋 인덱스 10개\n\n"
            "**📋 입력 파라미터:**\n"
            "- encoder.pkl과 완전 호환되는 6개 특성 입력\n"
            "- AI 모델 우선, 실패 시 룰 기반 폴백\n\n"
            "**✨ 활용 방법:**\n"
            "- 클라이언트에서 `selected_idx`로 해당 향수들의 상세 정보 조회\n"
            "- `proba` 정보로 사용자 선호도 분석 가능\n"
            "- `recommended_notes`로 향수 노트 기반 UI 구성 가능"
    )
)
def recommend_cluster_based(request: RecommendRequest):
    request_start_time = datetime.now()
    logger.info(f"🆕 클러스터 기반 향수 추천 요청: {request}")

    # ✅ 입력 검증
    if not validate_request_categories(request):
        logger.error("❌ 잘못된 카테고리 값 입력")
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 카테고리 값입니다. 지원되는 값: {SUPPORTED_CATEGORIES}"
        )

    # 요청 데이터를 딕셔너리로 변환
    request_dict = request.dict()

    try:
        # AI 모델 시도
        if _model_available:
            try:
                logger.info("🤖 AI 감정 클러스터 모델로 클러스터 기반 추천 시도")
                result = predict_cluster_recommendation(request_dict)

                # 처리 시간 업데이트
                total_processing_time = (datetime.now() - request_start_time).total_seconds()
                result["metadata"]["total_processing_time_seconds"] = round(total_processing_time, 3)

                logger.info(f"✅ 클러스터 기반 추천 성공 (클러스터: {result['cluster']}, 소요시간: {total_processing_time:.3f}초)")

                return ClusterRecommendResponse(**result)

            except Exception as e:
                logger.warning(f"⚠️ AI 모델 클러스터 추천 실패: {e}")
                # 룰 기반으로 폴백하되, 클러스터 형태로 변환
                logger.info("📋 룰 기반으로 폴백하여 클러스터 형태 응답 생성")

        else:
            logger.info("📋 AI 모델 사용 불가, 룰 기반으로 클러스터 형태 응답 생성")

        # 룰 기반 폴백 - 클러스터 형태로 변환
        rule_results = rule_based_recommendation(request_dict, 10)
        rule_df = pd.DataFrame(rule_results)

        # 가상의 클러스터 정보 생성 (룰 기반이므로)
        fallback_cluster = 2  # 기본 클러스터 (우아함, 친근함)
        fallback_proba = [0.1, 0.15, 0.4, 0.15, 0.1, 0.1]  # 가상 확률

        # 룰 기반 결과에서 노트 추출
        fallback_notes = get_top_notes_from_cluster(rule_df, top_k=15)

        # 룰 기반 결과에서 인덱스 추출
        fallback_indices = get_perfume_indices(rule_df, top_k=10)

        total_processing_time = (datetime.now() - request_start_time).total_seconds()

        fallback_result = {
            "cluster": fallback_cluster,
            "description": EMOTION_CLUSTER_MAP[fallback_cluster] + " (룰 기반 추정)",
            "proba": fallback_proba,
            "recommended_notes": fallback_notes,
            "selected_idx": fallback_indices,
            "metadata": {
                "processing_time_seconds": round(total_processing_time, 3),
                "total_cluster_perfumes": len(rule_df),
                "confidence": 0.4,  # 룰 기반이므로 낮은 신뢰도
                "method": "룰 기반 (AI 모델 대체)",
                "fallback_used": True,
                "filters_applied": {
                    "gender": request_dict["gender"],
                    "season": request_dict["season_tags"],
                    "time": request_dict["time_tags"]
                }
            }
        }

        logger.info(f"✅ 룰 기반 클러스터 형태 추천 완료 (소요시간: {total_processing_time:.3f}초)")

        return ClusterRecommendResponse(**fallback_result)

    except Exception as e:
        logger.error(f"❌ 클러스터 기반 추천 중 예외 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"클러스터 기반 추천 중 오류가 발생했습니다: {str(e)}"
        )


# ─── 15. 기존 API들 (하위 호환성 유지) ────────────────────────────────────────────────

@router.post(
    "/recommend",
    response_model=List[PerfumeRecommendItem],
    summary="향수 추천 (기존 방식, 하위 호환성)",
    description=(
            "**🔄 기존 향수 추천 API (하위 호환성 유지)**\n\n"
            "사용자의 선호도를 기반으로 향수를 추천합니다.\n\n"
            "**🤖 추천 방식:**\n"
            "1. **AI 감정 클러스터 모델**: 6개 입력 → 6개 감정 클러스터 분류 → 해당 클러스터 향수 추천\n"
            "2. **룰 기반 Fallback**: 조건부 필터링 + 스코어링 (모델이 없거나 실패한 경우)\n"
            "3. **다양성 보장**: 브랜드별 균형 잡힌 추천\n\n"
            "**⚠️ 권장사항:**\n"
            "- 새로운 프로젝트는 `/recommend-cluster` API 사용 권장\n"
            "- 더 구조화된 응답과 클러스터 정보 제공\n"
            "- 이 API는 기존 클라이언트 호환성을 위해 유지"
    )
)
def recommend_perfumes(request: RecommendRequest):
    request_start_time = datetime.now()
    logger.info(f"🎯 향수 추천 요청 시작 (기존 방식): {request}")

    # ✅ 입력 검증
    if not validate_request_categories(request):
        logger.error("❌ 잘못된 카테고리 값 입력")
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 카테고리 값입니다. 지원되는 값: {SUPPORTED_CATEGORIES}"
        )

    # 요청 데이터를 딕셔너리로 변환
    request_dict = request.dict()

    method_used = "알 수 없음"

    # 1) AI 모델 시도
    if _model_available:
        model_start_time = datetime.now()
        try:
            logger.info("🤖 AI 감정 클러스터 모델 추천 시도")

            # 감정 클러스터 모델로 추천
            top_10 = predict_with_emotion_cluster_model(request_dict)
            method_used = "AI 감정 클러스터 모델"

            model_time = (datetime.now() - model_start_time).total_seconds()
            logger.info(f"✅ AI 모델 추천 성공 (방법: {method_used}, 소요시간: {model_time:.3f}초)")

        except Exception as e:
            model_time = (datetime.now() - model_start_time).total_seconds()
            logger.warning(f"⚠️ AI 모델 추천 실패 (소요시간: {model_time:.3f}초): {e}")
            logger.info("📋 룰 기반 추천으로 전환")

            rule_start_time = datetime.now()
            rule_results = rule_based_recommendation(request_dict, 10)
            top_10 = pd.DataFrame(rule_results)
            rule_time = (datetime.now() - rule_start_time).total_seconds()
            method_used = "룰 기반 (AI 모델 실패)"
            logger.info(f"📋 룰 기반 추천 완료 (소요시간: {rule_time:.3f}초)")
    else:
        logger.info("📋 룰 기반 추천 사용 (모델 파일 크기 부족)")
        rule_start_time = datetime.now()
        rule_results = rule_based_recommendation(request_dict, 10)
        top_10 = pd.DataFrame(rule_results)
        rule_time = (datetime.now() - rule_start_time).total_seconds()
        method_used = "룰 기반 (모델 크기 부족)"
        logger.info(f"📋 룰 기반 추천 완료 (소요시간: {rule_time:.3f}초)")

    # 2) 결과 가공
    response_list: List[PerfumeRecommendItem] = []
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        emotions_text = get_emotion_text(row)
        score = float(row.get('score', 0.0))

        # 추천 이유 생성
        reason = get_recommendation_reason(score, method_used)

        response_list.append(
            PerfumeRecommendItem(
                name=str(row["name"]),
                brand=str(row["brand"]),
                image_url=str(row["image_url"]),
                notes=str(row["notes"]),
                emotions=emotions_text,
                reason=reason,
                score=score,
                method=method_used
            )
        )

    # 처리 시간 계산
    total_processing_time = (datetime.now() - request_start_time).total_seconds()

    logger.info(f"✅ 향수 추천 완료: {len(response_list)}개 ({method_used})")
    logger.info(f"⏱️ 총 처리 시간: {total_processing_time:.3f}초")
    if response_list:
        logger.info(
            f"📊 점수 범위: {min(item.score for item in response_list):.3f} ~ {max(item.score for item in response_list):.3f}")
        logger.info(f"📊 평균 점수: {sum(item.score for item in response_list) / len(response_list):.3f}")

    return response_list


@router.get(
    "/model-status",
    summary="모델 상태 확인",
    description="AI 모델과 관련 파일들의 상태를 확인합니다."
)
def get_model_status():
    """모델 및 시스템 상태를 반환합니다."""

    # 파일 상태 확인
    model_exists = os.path.exists(MODEL_PATH)
    encoder_exists = os.path.exists(ENCODER_PATH)

    model_size = os.path.getsize(MODEL_PATH) if model_exists else 0
    encoder_size = os.path.getsize(ENCODER_PATH) if encoder_exists else 0

    # ✅ 수정된 유효성 기준
    model_size_valid = model_size > 10000  # 10KB 이상 (기존: 100KB)
    encoder_size_valid = encoder_size > 500  # 500B 이상 (기존: 100B)

    # 모델 구조 정보 (모델이 로드된 경우)
    model_structure = None
    if _model is not None:
        try:
            model_structure = {
                "input_shape": str(_model.input_shape),
                "output_shape": str(_model.output_shape),
                "total_params": _model.count_params(),
                "layers": len(_model.layers),
                "layer_details": [
                    {
                        "name": layer.name,
                        "type": layer.__class__.__name__,
                        "units": getattr(layer, 'units', None),
                        "activation": getattr(layer.activation, '__name__', None) if hasattr(layer,
                                                                                             'activation') else None
                    } for layer in _model.layers
                ]
            }
        except:
            model_structure = "모델 정보 읽기 실패"

    # ✅ scikit-learn 버전 정보 추가
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except:
        sklearn_version = "불명"

    return {
        "timestamp": datetime.now().isoformat(),
        "model_available": _model_available,
        "files": {
            "keras_model": {
                "path": MODEL_PATH,
                "exists": model_exists,
                "size_bytes": model_size,
                "size_kb": round(model_size / 1024, 2),  # KB 단위 추가
                "size_mb": round(model_size / (1024 * 1024), 2),
                "valid": model_size_valid,
                "min_required_kb": 10,  # 최소 요구사항 명시
                "status": "✅ 유효" if model_size_valid else "❌ 크기 부족"
            },
            "encoder": {
                "path": ENCODER_PATH,
                "exists": encoder_exists,
                "size_bytes": encoder_size,
                "size_kb": round(encoder_size / 1024, 2),
                "valid": encoder_size_valid,
                "min_required_bytes": 500,  # 최소 요구사항 명시
                "status": "✅ 유효" if encoder_size_valid else "❌ 크기 부족"
            }
        },
        "model_info": {
            "expected_structure": {
                "input_shape": "(None, 6)",
                "layers": [
                    "Dense(64, activation='relu')",
                    "Dense(6, activation='softmax')"
                ],
                "output_shape": "(None, 6)",
                "purpose": "6개 감정 클러스터 분류"
            },
            "keras_version": "3.8.0 (detected from file)",
            "saved_date": "2025-06-05@05:09:56"
        },
        "model_structure": model_structure,
        "emotion_clusters": EMOTION_CLUSTER_MAP,
        "recommendation_method": "AI 감정 클러스터 모델" if _model_available else "룰 기반",
        "fallback_encoder_ready": _fallback_encoder is not None,
        "supported_categories": SUPPORTED_CATEGORIES,  # ✅ encoder.pkl 호환 카테고리
        "system": {
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn_version,  # ✅ 추가
            "current_directory": os.getcwd(),
            "router_location": BASE_DIR,
            "dataset_loaded": len(df) > 0,
            "dataset_size": len(df)
        },
        "dataset_info": {
            "total_perfumes": len(df),
            "columns": list(df.columns),
            "sample_brands": df['brand'].unique()[:5].tolist() if 'brand' in df.columns else [],
            "emotion_cluster_distribution": dict(
                df['emotion_cluster'].value_counts()) if 'emotion_cluster' in df.columns else None
        },
        "compatibility": {
            "encoder_type": "OrdinalEncoder (6 output features)",
            "api_schema_categories": SUPPORTED_CATEGORIES,
            "encoder_fallback_available": _fallback_encoder is not None,
            "sklearn_compatibility": "OrdinalEncoder 사용 (encoder.pkl 호환)"
        },
        "🆕_new_features": {
            "cluster_based_api": "/perfumes/recommend-cluster",
            "cluster_response_format": {
                "cluster": "int (0-5)",
                "description": "str",
                "proba": "List[float] (length 6)",
                "recommended_notes": "List[str] (max 15)",
                "selected_idx": "List[int] (max 10)"
            }
        }
    }


@router.get(
    "/health",
    summary="추천 시스템 헬스 체크",
    description="추천 시스템의 전반적인 건강 상태를 확인합니다."
)
def health_check():
    """추천 시스템 헬스 체크"""

    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {}
    }

    # 데이터셋 확인
    try:
        health_status["checks"]["dataset"] = {
            "status": "ok" if len(df) > 0 else "error",
            "perfume_count": len(df),
            "columns_available": len(df.columns)
        }
    except Exception as e:
        health_status["checks"]["dataset"] = {
            "status": "error",
            "error": str(e)
        }

    # 모델 파일 확인 (수정된 기준)
    try:
        model_exists = os.path.exists(MODEL_PATH)
        encoder_exists = os.path.exists(ENCODER_PATH)
        model_size = os.path.getsize(MODEL_PATH) if model_exists else 0
        encoder_size = os.path.getsize(ENCODER_PATH) if encoder_exists else 0

        # ✅ 수정된 크기 기준
        model_valid = model_exists and model_size > 10000  # 10KB 이상
        encoder_valid = encoder_exists and encoder_size > 500  # 500B 이상

        health_status["checks"]["model_files"] = {
            "status": "ok" if model_valid and encoder_valid else "warning",
            "model_available": model_valid,
            "encoder_available": encoder_valid,
            "model_size_kb": round(model_size / 1024, 2),
            "encoder_size_bytes": encoder_size,
            "fallback_ready": _fallback_encoder is not None
        }
    except Exception as e:
        health_status["checks"]["model_files"] = {
            "status": "error",
            "error": str(e)
        }

    # ✅ 인코더 호환성 체크 추가
    try:
        fallback_encoder = get_fallback_encoder()
        health_status["checks"]["encoder_compatibility"] = {
            "status": "ok" if fallback_encoder is not None else "error",
            "fallback_encoder_available": fallback_encoder is not None,
            "encoder_type": "OrdinalEncoder",
            "output_features": 6,
            "sklearn_compatible": True  # fallback 생성 성공하면 호환됨
        }
    except Exception as e:
        health_status["checks"]["encoder_compatibility"] = {
            "status": "error",
            "error": str(e),
            "sklearn_compatible": False
        }

    # 🆕 클러스터 기반 추천 테스트
    try:
        test_request = {
            "gender": "women",
            "season_tags": "spring",
            "time_tags": "day",
            "desired_impression": "confident, fresh",
            "activity": "casual",
            "weather": "any"
        }

        start_time = datetime.now()
        if _model_available:
            try:
                cluster_result = predict_cluster_recommendation(test_request)
                method = "AI 클러스터 모델"
                test_success = True
                result_count = len(cluster_result.get("selected_idx", []))
            except:
                rule_results = rule_based_recommendation(test_request, 3)
                test_results = pd.DataFrame(rule_results)
                method = "룰 기반 (AI 실패)"
                test_success = True
                result_count = len(test_results)
        else:
            rule_results = rule_based_recommendation(test_request, 3)
            test_results = pd.DataFrame(rule_results)
            method = "룰 기반 (모델 크기 부족)"
            test_success = True
            result_count = len(test_results)

        processing_time = (datetime.now() - start_time).total_seconds()

        health_status["checks"]["recommendation_system"] = {
            "status": "ok" if test_success and result_count > 0 else "error",
            "test_result_count": result_count,
            "processing_time_seconds": round(processing_time, 3),
            "method": method
        }

        # 🆕 클러스터 API 테스트 추가
        health_status["checks"]["cluster_api"] = {
            "status": "ok" if _model_available else "warning",
            "cluster_api_available": _model_available,
            "fallback_available": True
        }

    except Exception as e:
        health_status["checks"]["recommendation_system"] = {
            "status": "error",
            "error": str(e)
        }

    # 전체 상태 결정
    all_checks = health_status["checks"].values()
    if any(check.get("status") == "error" for check in all_checks):
        health_status["status"] = "unhealthy"
    elif any(check.get("status") == "warning" for check in all_checks):
        health_status["status"] = "degraded"

    return health_status


@router.post(
    "/test-recommendation",
    summary="추천 시스템 테스트 (개발용)",
    description="다양한 조건으로 추천 시스템을 테스트합니다."
)
def test_recommendation_system():
    """추천 시스템을 다양한 조건으로 테스트합니다."""

    test_cases = [
        {
            "name": "여성용 봄 데이 향수",
            "request": {
                "gender": "women",
                "season_tags": "spring",
                "time_tags": "day",
                "desired_impression": "confident, fresh",
                "activity": "casual",
                "weather": "any"
            }
        },
        {
            "name": "남성용 겨울 나이트 향수",
            "request": {
                "gender": "men",
                "season_tags": "winter",
                "time_tags": "night",
                "desired_impression": "confident, mysterious",
                "activity": "date",
                "weather": "cold"
            }
        },
        {
            "name": "유니섹스 여름 향수",
            "request": {
                "gender": "unisex",
                "season_tags": "summer",
                "time_tags": "day",
                "desired_impression": "elegant, friendly",
                "activity": "work",
                "weather": "hot"
            }
        }
    ]

    results = []

    for test_case in test_cases:
        try:
            start_time = datetime.now()

            # 🆕 클러스터 기반 추천 테스트
            cluster_result = None
            cluster_success = False

            if _model_available:
                try:
                    cluster_result = predict_cluster_recommendation(test_case["request"])
                    cluster_success = True
                    cluster_method = "AI 감정 클러스터 모델"
                except Exception as e:
                    cluster_success = False
                    cluster_error = str(e)
                    cluster_method = "실패"

            # 기존 방식 추천 테스트
            if _model_available:
                try:
                    ai_results = predict_with_emotion_cluster_model(test_case["request"])
                    result_data = ai_results.to_dict('records')[:3]
                    method = "AI 감정 클러스터 모델"
                except:
                    rule_results = rule_based_recommendation(test_case["request"], 5)
                    result_data = rule_results[:3]
                    method = "룰 기반 (AI 실패)"
            else:
                rule_results = rule_based_recommendation(test_case["request"], 5)
                result_data = rule_results[:3]
                method = "룰 기반 (모델 크기 부족)"

            processing_time = (datetime.now() - start_time).total_seconds()

            # 결과 구성
            test_result = {
                "test_name": test_case["name"],
                "request": test_case["request"],
                "success": True,
                "legacy_api": {
                    "method": method,
                    "result_count": len(result_data),
                    "sample_results": [
                        {
                            "name": r.get("name", ""),
                            "brand": r.get("brand", ""),
                            "score": round(r.get("score", 0), 3)
                        } for r in result_data
                    ]
                },
                "🆕_cluster_api": {
                    "success": cluster_success,
                    "method": cluster_method if cluster_success else "실패",
                    "cluster": cluster_result.get("cluster") if cluster_result else None,
                    "description": cluster_result.get("description") if cluster_result else None,
                    "confidence": cluster_result["proba"][cluster_result["cluster"]] if cluster_result else None,
                    "notes_count": len(cluster_result.get("recommended_notes", [])) if cluster_result else 0,
                    "selected_perfumes_count": len(cluster_result.get("selected_idx", [])) if cluster_result else 0,
                    "error": cluster_error if not cluster_success else None
                },
                "processing_time_seconds": round(processing_time, 3)
            }

            results.append(test_result)

        except Exception as e:
            results.append({
                "test_name": test_case["name"],
                "request": test_case["request"],
                "success": False,
                "error": str(e),
                "processing_time_seconds": 0
            })

    # ✅ scikit-learn 버전 정보 추가
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except:
        sklearn_version = "불명"

    return {
        "timestamp": datetime.now().isoformat(),
        "model_available": _model_available,
        "model_size_kb": round(os.path.getsize(MODEL_PATH) / 1024, 2) if os.path.exists(MODEL_PATH) else 0,
        "encoder_size_bytes": os.path.getsize(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else 0,
        "fallback_encoder_available": _fallback_encoder is not None,
        "dataset_size": len(df),
        "emotion_clusters": EMOTION_CLUSTER_MAP,
        "supported_categories": SUPPORTED_CATEGORIES,  # ✅ encoder.pkl 호환 카테고리
        "sklearn_version": sklearn_version,  # ✅ 추가
        "🆕_new_features": {
            "cluster_api_endpoint": "/perfumes/recommend-cluster",
            "cluster_response_includes": [
                "cluster (int)", "description (str)", "proba (List[float])",
                "recommended_notes (List[str])", "selected_idx (List[int])"
            ]
        },
        "test_results": results,
        "summary": {
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r["success"]),
            "cluster_api_successful_tests": sum(1 for r in results if r.get("🆕_cluster_api", {}).get("success", False)),
            "average_processing_time": round(
                sum(r.get("processing_time_seconds", 0) for r in results) / len(results), 3
            )
        }
    }


# ─── 16. 🆕 노트 분석 전용 API (추가 기능) ────────────────────────────────────────────────

@router.get(
    "/notes/analysis",
    summary="향수 노트 분석 (클러스터별)",
    description="각 감정 클러스터별로 인기 있는 향수 노트를 분석합니다."
)
def analyze_notes_by_cluster():
    """클러스터별 노트 분석 정보를 반환합니다."""

    cluster_notes_analysis = {}

    try:
        for cluster_id in range(6):  # 0-5 클러스터
            if 'emotion_cluster' in df.columns:
                cluster_perfumes = df[df['emotion_cluster'] == cluster_id]

                if len(cluster_perfumes) > 0:
                    # 상위 15개 노트 추출
                    top_notes = get_top_notes_from_cluster(cluster_perfumes, top_k=15)

                    # 노트 빈도 계산
                    all_notes = []
                    for _, row in cluster_perfumes.iterrows():
                        notes = parse_notes_from_string(row.get('notes', ''))
                        all_notes.extend(notes)

                    note_counter = Counter(all_notes)

                    cluster_notes_analysis[cluster_id] = {
                        "cluster_name": EMOTION_CLUSTER_MAP.get(cluster_id, f"클러스터 {cluster_id}"),
                        "perfume_count": len(cluster_perfumes),
                        "total_notes_found": len(all_notes),
                        "unique_notes_count": len(set(all_notes)),
                        "top_15_notes": top_notes,
                        "top_5_with_frequency": [
                            {"note": note, "frequency": count}
                            for note, count in note_counter.most_common(5)
                        ]
                    }
                else:
                    cluster_notes_analysis[cluster_id] = {
                        "cluster_name": EMOTION_CLUSTER_MAP.get(cluster_id, f"클러스터 {cluster_id}"),
                        "perfume_count": 0,
                        "error": "해당 클러스터에 향수가 없습니다."
                    }

        return {
            "timestamp": datetime.now().isoformat(),
            "total_perfumes": len(df),
            "cluster_analysis": cluster_notes_analysis,
            "summary": {
                "clusters_with_data": sum(1 for analysis in cluster_notes_analysis.values() if "error" not in analysis),
                "total_unique_notes": len(set(
                    note for analysis in cluster_notes_analysis.values()
                    if "top_15_notes" in analysis
                    for note in analysis["top_15_notes"]
                ))
            }
        }

    except Exception as e:
        logger.error(f"❌ 노트 분석 중 오류: {e}")
        return {
            "error": f"노트 분석 중 오류가 발생했습니다: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@router.get(
    "/notes/search/{note_name}",
    summary="특정 노트가 포함된 향수 검색",
    description="지정된 노트가 포함된 향수들을 검색합니다."
)
def search_perfumes_by_note(note_name: str, limit: int = 20):
    """특정 노트가 포함된 향수들을 검색합니다."""

    try:
        note_name = note_name.lower().strip()
        matching_perfumes = []

        for idx, row in df.iterrows():
            notes = parse_notes_from_string(row.get('notes', ''))

            if note_name in notes:
                perfume_info = {
                    "index": int(idx),
                    "name": str(row.get('name', '')),
                    "brand": str(row.get('brand', '')),
                    "notes": str(row.get('notes', '')),
                    "emotion_cluster": int(row.get('emotion_cluster', 0)),
                    "cluster_description": EMOTION_CLUSTER_MAP.get(int(row.get('emotion_cluster', 0)), "알 수 없음"),
                    "gender": str(row.get('gender', '')),
                    "season_tags": str(row.get('season_tags', '')),
                    "time_tags": str(row.get('time_tags', ''))
                }
                matching_perfumes.append(perfume_info)

        # 클러스터별로 정렬
        matching_perfumes.sort(key=lambda x: x['emotion_cluster'])

        # 제한된 개수만 반환
        limited_results = matching_perfumes[:limit]

        # 클러스터별 분포 계산
        cluster_distribution = {}
        for perfume in matching_perfumes:
            cluster = perfume['emotion_cluster']
            cluster_distribution[cluster] = cluster_distribution.get(cluster, 0) + 1

        return {
            "timestamp": datetime.now().isoformat(),
            "search_note": note_name,
            "total_matches": len(matching_perfumes),
            "returned_count": len(limited_results),
            "cluster_distribution": cluster_distribution,
            "matching_perfumes": limited_results
        }

    except Exception as e:
        logger.error(f"❌ 노트 검색 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"노트 검색 중 오류가 발생했습니다: {str(e)}"
        )


# ─── 17. 🆕 클러스터 정보 API ────────────────────────────────────────────────

@router.get(
    "/clusters/info",
    summary="감정 클러스터 정보",
    description="모든 감정 클러스터의 정보와 통계를 반환합니다."
)
def get_cluster_info():
    """감정 클러스터 정보와 통계를 반환합니다."""

    try:
        cluster_info = {}

        for cluster_id in range(6):
            cluster_perfumes = df[
                df['emotion_cluster'] == cluster_id] if 'emotion_cluster' in df.columns else pd.DataFrame()

            cluster_info[cluster_id] = {
                "cluster_id": cluster_id,
                "description": EMOTION_CLUSTER_MAP.get(cluster_id, f"클러스터 {cluster_id}"),
                "perfume_count": len(cluster_perfumes),
                "percentage": round(len(cluster_perfumes) / len(df) * 100, 2) if len(df) > 0 else 0
            }

            if len(cluster_perfumes) > 0:
                # 성별 분포
                gender_dist = cluster_perfumes[
                    'gender'].value_counts().to_dict() if 'gender' in cluster_perfumes.columns else {}

                # 브랜드 분포 (상위 5개)
                brand_dist = cluster_perfumes['brand'].value_counts().head(
                    5).to_dict() if 'brand' in cluster_perfumes.columns else {}

                cluster_info[cluster_id].update({
                    "gender_distribution": gender_dist,
                    "top_5_brands": brand_dist,
                    "sample_perfumes": [
                        {
                            "name": row['name'],
                            "brand": row['brand']
                        } for _, row in cluster_perfumes.head(3).iterrows()
                    ]
                })

        return {
            "timestamp": datetime.now().isoformat(),
            "total_perfumes": len(df),
            "total_clusters": 6,
            "cluster_details": cluster_info,
            "emotion_cluster_map": EMOTION_CLUSTER_MAP
        }

    except Exception as e:
        logger.error(f"❌ 클러스터 정보 조회 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"클러스터 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/clusters/{cluster_id}/perfumes",
    summary="특정 클러스터의 향수 목록",
    description="지정된 감정 클러스터에 속한 향수들의 목록을 반환합니다."
)
def get_cluster_perfumes(cluster_id: int, limit: int = 50, offset: int = 0):
    """특정 클러스터의 향수 목록을 반환합니다."""

    if cluster_id < 0 or cluster_id > 5:
        raise HTTPException(
            status_code=400,
            detail="클러스터 ID는 0-5 범위여야 합니다."
        )

    try:
        if 'emotion_cluster' not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="데이터셋에 emotion_cluster 컬럼이 없습니다."
            )

        cluster_perfumes = df[df['emotion_cluster'] == cluster_id]

        if len(cluster_perfumes) == 0:
            return {
                "cluster_id": cluster_id,
                "description": EMOTION_CLUSTER_MAP.get(cluster_id, f"클러스터 {cluster_id}"),
                "total_count": 0,
                "perfumes": [],
                "message": "해당 클러스터에 향수가 없습니다."
            }

        # 페이지네이션 적용
        paginated_perfumes = cluster_perfumes.iloc[offset:offset + limit]

        perfume_list = []
        for idx, row in paginated_perfumes.iterrows():
            perfume_list.append({
                "index": int(idx),
                "name": str(row.get('name', '')),
                "brand": str(row.get('brand', '')),
                "image_url": str(row.get('image_url', '')),
                "notes": str(row.get('notes', '')),
                "gender": str(row.get('gender', '')),
                "season_tags": str(row.get('season_tags', '')),
                "time_tags": str(row.get('time_tags', '')),
                "desired_impression": str(row.get('desired_impression', ''))
            })

        return {
            "cluster_id": cluster_id,
            "description": EMOTION_CLUSTER_MAP.get(cluster_id, f"클러스터 {cluster_id}"),
            "total_count": len(cluster_perfumes),
            "returned_count": len(perfume_list),
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < len(cluster_perfumes),
            "perfumes": perfume_list
        }

    except Exception as e:
        logger.error(f"❌ 클러스터 향수 목록 조회 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"클러스터 향수 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )