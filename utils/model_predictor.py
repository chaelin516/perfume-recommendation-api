# utils/model_predictor.py
# 🤖 향수 추천 모델 전용 (감정 분석 모델은 emotion_model_loader.py로 분리)

import numpy as np
import pickle
import tensorflow as tf
import os
import logging

logger = logging.getLogger(__name__)

# ─── 향수 추천 모델 파일 경로 ────────────────────────────────────────────────
MODEL_PATH = "./models/final_model.keras"
ENCODER_PATH = "./models/encoder.pkl"

# 전역 변수로 모델과 인코더를 저장 (lazy loading)
_recommendation_model = None
_recommendation_encoder = None


def load_recommendation_model_and_encoder():
    """향수 추천 모델과 인코더를 lazy loading으로 로드"""
    global _recommendation_model, _recommendation_encoder

    if _recommendation_model is None or _recommendation_encoder is None:
        try:
            # compile=False로 모델 로드 (optimizer 문제 회피)
            logger.info("📦 향수 추천 모델을 로딩 중...")

            if not os.path.exists(MODEL_PATH):
                logger.error(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
                raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

            if not os.path.exists(ENCODER_PATH):
                logger.error(f"❌ 인코더 파일이 없습니다: {ENCODER_PATH}")
                raise FileNotFoundError(f"인코더 파일이 없습니다: {ENCODER_PATH}")

            # 모델 파일 크기 확인
            model_size = os.path.getsize(MODEL_PATH)
            encoder_size = os.path.getsize(ENCODER_PATH)

            logger.info(f"📊 모델 파일 크기: {model_size:,} bytes ({model_size / 1024:.1f}KB)")
            logger.info(f"📊 인코더 파일 크기: {encoder_size:,} bytes")

            # TensorFlow 모델 로드
            _recommendation_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("✅ TensorFlow 향수 추천 모델 로딩 완료")

            # 모델 구조 정보
            logger.info(f"📊 모델 입력 shape: {_recommendation_model.input_shape}")
            logger.info(f"📊 모델 출력 shape: {_recommendation_model.output_shape}")
            logger.info(f"📊 모델 레이어 수: {len(_recommendation_model.layers)}")

            # 인코더 로드
            logger.info("📦 향수 추천 인코더를 로딩 중...")
            with open(ENCODER_PATH, "rb") as f:
                _recommendation_encoder = pickle.load(f)
            logger.info("✅ 향수 추천 인코더 로딩 완료")

            # 🧪 통합 테스트
            logger.info("🧪 향수 추천 모델 통합 테스트 시작...")
            test_input = ["women", "spring", "day", "elegant", "casual", "any"]
            test_result = predict_emotion_cluster(test_input)
            logger.info(f"✅ 향수 추천 모델 테스트 성공: 클러스터 {test_result}")

        except FileNotFoundError as e:
            logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
            raise e
        except Exception as e:
            logger.error(f"❌ 향수 추천 모델/인코더 로딩 중 오류 발생: {e}")
            raise e

    return _recommendation_model, _recommendation_encoder


def predict_emotion_cluster(user_input: list) -> int:
    """
    향수 추천을 위한 감정 클러스터 예측

    Args:
        user_input: [gender, season, time, desired_impression, activity, weather]

    Returns:
        int: 예측된 감정 클러스터 (0~5)
    """
    try:
        model, encoder = load_recommendation_model_and_encoder()

        logger.debug(f"🔮 향수 추천 입력: {user_input}")

        # 입력 데이터 전처리
        X = encoder.transform([user_input])
        logger.debug(f"📊 인코딩된 입력 shape: {X.shape}")

        # 예측 수행
        pred = model.predict(X, verbose=0)  # verbose=0으로 로그 출력 제거
        cluster_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        logger.info(f"🎯 향수 추천 감정 클러스터 예측: {cluster_id} (신뢰도: {confidence:.3f})")

        # 모든 클러스터 확률 로깅
        for i, prob in enumerate(pred[0]):
            logger.debug(f"  클러스터 {i}: {prob:.3f}")

        return cluster_id

    except Exception as e:
        logger.error(f"❌ 향수 추천 예측 중 오류 발생: {e}")
        # 기본값 반환 (에러 발생 시)
        logger.warning("⚠️ 오류로 인해 기본 클러스터 0 반환")
        return 0


def predict_emotion_cluster_with_probabilities(user_input: list) -> dict:
    """
    향수 추천을 위한 감정 클러스터 예측 (확률 포함)

    Args:
        user_input: [gender, season, time, desired_impression, activity, weather]

    Returns:
        dict: {
            "cluster": int,
            "confidence": float,
            "probabilities": list,
            "method": str
        }
    """
    try:
        model, encoder = load_recommendation_model_and_encoder()

        logger.debug(f"🔮 향수 추천 상세 예측 입력: {user_input}")

        # 입력 데이터 전처리
        X = encoder.transform([user_input])

        # 예측 수행
        pred = model.predict(X, verbose=0)
        probabilities = pred[0].tolist()  # numpy array를 list로 변환
        cluster_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        result = {
            "cluster": cluster_id,
            "confidence": confidence,
            "probabilities": probabilities,
            "method": "TensorFlow 향수 추천 모델",
            "input_shape": X.shape,
            "model_output_shape": pred.shape
        }

        logger.info(f"🎯 향수 추천 상세 예측 완료: 클러스터 {cluster_id} (신뢰도: {confidence:.3f})")

        return result

    except Exception as e:
        logger.error(f"❌ 향수 추천 상세 예측 중 오류 발생: {e}")
        # 에러 시 기본값 반환
        return {
            "cluster": 0,
            "confidence": 0.3,
            "probabilities": [0.3, 0.15, 0.15, 0.15, 0.15, 0.1],  # 기본 확률 분포
            "method": "기본값 (에러 발생)",
            "error": str(e)
        }


def validate_recommendation_input(user_input: list) -> bool:
    """
    향수 추천 입력 데이터 검증

    Args:
        user_input: [gender, season, time, desired_impression, activity, weather]

    Returns:
        bool: 유효한 입력인지 여부
    """
    try:
        if len(user_input) != 6:
            logger.error(f"❌ 입력 길이가 잘못됨: {len(user_input)} (예상: 6)")
            return False

        # 기본적인 타입 체크
        for i, item in enumerate(user_input):
            if not isinstance(item, str):
                logger.error(f"❌ 입력 항목 {i}가 문자열이 아님: {type(item)}")
                return False
            if not item.strip():
                logger.error(f"❌ 입력 항목 {i}가 비어있음")
                return False

        logger.debug(f"✅ 향수 추천 입력 검증 통과: {user_input}")
        return True

    except Exception as e:
        logger.error(f"❌ 입력 검증 중 오류: {e}")
        return False


def check_recommendation_model_files() -> dict:
    """향수 추천 모델 파일들이 존재하는지 확인"""
    status = {
        "model_exists": False,
        "encoder_exists": False,
        "model_size": 0,
        "encoder_size": 0,
        "model_path": MODEL_PATH,
        "encoder_path": ENCODER_PATH
    }

    try:
        # 모델 파일 확인
        if os.path.exists(MODEL_PATH):
            status["model_exists"] = True
            status["model_size"] = os.path.getsize(MODEL_PATH)
            logger.info(f"✅ 향수 추천 모델 파일 존재: {status['model_size']:,} bytes")
        else:
            logger.warning(f"❌ 향수 추천 모델 파일 없음: {MODEL_PATH}")

        # 인코더 파일 확인
        if os.path.exists(ENCODER_PATH):
            status["encoder_exists"] = True
            status["encoder_size"] = os.path.getsize(ENCODER_PATH)
            logger.info(f"✅ 향수 추천 인코더 파일 존재: {status['encoder_size']:,} bytes")
        else:
            logger.warning(f"❌ 향수 추천 인코더 파일 없음: {ENCODER_PATH}")

        status["all_files_ready"] = status["model_exists"] and status["encoder_exists"]

        return status

    except Exception as e:
        logger.error(f"❌ 향수 추천 모델 파일 확인 중 오류: {e}")
        status["error"] = str(e)
        return status


def get_recommendation_model_info() -> dict:
    """향수 추천 모델 정보 반환"""
    try:
        # 파일 상태 확인
        file_status = check_recommendation_model_files()

        info = {
            "model_type": "TensorFlow Keras",
            "purpose": "향수 추천 (감정 클러스터 예측)",
            "input_features": [
                "gender", "season", "time",
                "desired_impression", "activity", "weather"
            ],
            "output_clusters": 6,
            "cluster_descriptions": {
                0: "차분한, 편안한",
                1: "자신감, 신선함",
                2: "우아함, 친근함",
                3: "순수함, 친근함",
                4: "신비로운, 매력적",
                5: "활기찬, 에너지"
            },
            "file_status": file_status,
            "model_loaded": _recommendation_model is not None,
            "encoder_loaded": _recommendation_encoder is not None
        }

        # 로드된 모델이 있으면 추가 정보
        if _recommendation_model is not None:
            info["model_details"] = {
                "input_shape": str(_recommendation_model.input_shape),
                "output_shape": str(_recommendation_model.output_shape),
                "num_layers": len(_recommendation_model.layers),
                "trainable_params": _recommendation_model.count_params()
            }

        return info

    except Exception as e:
        logger.error(f"❌ 향수 추천 모델 정보 조회 중 오류: {e}")
        return {"error": str(e)}


def reset_recommendation_model():
    """향수 추천 모델 메모리 초기화"""
    global _recommendation_model, _recommendation_encoder

    logger.info("🔄 향수 추천 모델 메모리 초기화...")

    _recommendation_model = None
    _recommendation_encoder = None

    logger.info("✅ 향수 추천 모델 메모리 초기화 완료")


# 🧪 테스트 함수
def test_recommendation_model():
    """향수 추천 모델 테스트"""
    logger.info("🧪 향수 추천 모델 테스트 시작...")

    try:
        # 파일 상태 확인
        file_status = check_recommendation_model_files()
        logger.info(f"📊 파일 상태: {file_status}")

        if not file_status.get("all_files_ready"):
            logger.error("❌ 모델 파일이 준비되지 않음")
            return False

        # 테스트 케이스들
        test_cases = [
            ["women", "spring", "day", "confident, fresh", "casual", "hot"],
            ["men", "winter", "night", "mysterious", "date", "cold"],
            ["unisex", "summer", "day", "elegant", "work", "any"]
        ]

        for i, test_input in enumerate(test_cases, 1):
            logger.info(f"🧪 테스트 케이스 {i}: {test_input}")

            # 입력 검증
            if not validate_recommendation_input(test_input):
                logger.error(f"❌ 테스트 케이스 {i} 입력 검증 실패")
                continue

            # 기본 예측
            cluster = predict_emotion_cluster(test_input)
            logger.info(f"✅ 테스트 케이스 {i} 기본 예측: 클러스터 {cluster}")

            # 상세 예측
            detailed_result = predict_emotion_cluster_with_probabilities(test_input)
            logger.info(f"✅ 테스트 케이스 {i} 상세 예측: {detailed_result['cluster']} (신뢰도: {detailed_result['confidence']:.3f})")

        logger.info("✅ 향수 추천 모델 테스트 완료")
        return True

    except Exception as e:
        logger.error(f"❌ 향수 추천 모델 테스트 중 오류: {e}")
        return False


if __name__ == "__main__":
    # 직접 실행 시 테스트
    logging.basicConfig(level=logging.INFO)

    logger.info("🚀 향수 추천 모델 테스트 실행...")

    # 모델 정보 출력
    model_info = get_recommendation_model_info()
    logger.info(f"📋 모델 정보: {model_info}")

    # 테스트 실행
    test_success = test_recommendation_model()

    if test_success:
        logger.info("🎉 모든 테스트 성공!")
    else:
        logger.error("❌ 테스트 실패")