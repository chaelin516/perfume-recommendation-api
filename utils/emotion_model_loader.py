# utils/emotion_model_loader.py
# 🎭 감정 분석 모델 로더 - Google Drive 연동 버전

import os
import pickle
import requests
import logging
from pathlib import Path
import gdown
from typing import Optional, Tuple, Any
import hashlib

logger = logging.getLogger(__name__)

# 🔗 Google Drive 파일 ID 설정
EMOTION_MODEL_FILE_ID = "1JYUJvKVb44p63ctWe3c1G_qmdcDr-Xix"  # 감정 모델 파일 ID
VECTORIZER_FILE_ID = None  # 벡터라이저는 로컬 파일 사용 (Git에 포함)

# 로컬 모델 파일 경로
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "emotion_models"
EMOTION_MODEL_PATH = MODELS_DIR / "scent_emotion_model_v6.keras"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

# 전역 변수로 모델과 벡터라이저 저장
_emotion_model = None
_vectorizer = None
_model_loaded = False


def create_models_directory():
    """모델 디렉토리 생성"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 모델 디렉토리 확인: {MODELS_DIR}")


def download_from_google_drive_gdown(file_id: str, output_path: str) -> bool:
    """gdown 라이브러리를 사용한 Google Drive 다운로드"""
    try:
        logger.info(f"📥 Google Drive에서 파일 다운로드 시작 (gdown): {file_id}")

        # gdown을 사용한 다운로드
        url = f"https://drive.google.com/uc?id={file_id}"

        # 다운로드 실행
        output = gdown.download(url, output_path, quiet=False)

        if output and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ gdown 다운로드 완료: {output_path} ({file_size:,} bytes)")
            return True
        else:
            logger.error("❌ gdown 다운로드 실패")
            return False

    except Exception as e:
        logger.error(f"❌ gdown 다운로드 오류: {e}")
        return False


def download_from_google_drive_requests(file_id: str, destination: str) -> bool:
    """requests를 사용한 Google Drive 다운로드 (폴백)"""
    try:
        logger.info(f"📥 Google Drive에서 파일 다운로드 시작 (requests): {file_id}")

        # Google Drive 다운로드 URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"

        # 세션 생성
        session = requests.Session()
        response = session.get(url, stream=True)

        # 큰 파일의 경우 확인 토큰 처리
        if response.status_code == 200:
            # 바이러스 스캔 경고가 있는지 확인
            if "virus scan warning" in response.text.lower() or "download_warning" in response.text:
                logger.info("🔍 대용량 파일 확인 토큰 처리 중...")

                # 확인 페이지에서 실제 다운로드 링크 찾기
                confirm_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
                response = session.get(confirm_url, stream=True)

        # 파일 저장
        if response.status_code == 200:
            os.makedirs(os.path.dirname(destination), exist_ok=True)

            total_size = 0
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)

                        # 진행 상황 로그 (10MB마다)
                        if total_size % (10 * 1024 * 1024) == 0:
                            logger.info(f"📥 다운로드 진행: {total_size / 1024 / 1024:.1f}MB")

            file_size = os.path.getsize(destination)
            logger.info(f"✅ requests 다운로드 완료: {destination} ({file_size:,} bytes)")
            return True
        else:
            logger.error(f"❌ 다운로드 실패: HTTP {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"❌ requests 다운로드 오류: {e}")
        return False


def download_model_file(file_id: str, output_path: str) -> bool:
    """모델 파일 다운로드 (여러 방법 시도)"""
    if not file_id:
        logger.warning("⚠️ 파일 ID가 제공되지 않음")
        return False

    create_models_directory()

    # 파일이 이미 존재하는지 확인
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1024:  # 1KB 이상이면 유효한 파일로 간주
            logger.info(f"✅ 파일이 이미 존재: {output_path} ({file_size:,} bytes)")
            return True

    # 방법 1: gdown 시도
    try:
        import gdown
        if download_from_google_drive_gdown(file_id, output_path):
            return True
    except ImportError:
        logger.warning("⚠️ gdown 라이브러리가 없음. requests로 시도합니다.")
    except Exception as e:
        logger.warning(f"⚠️ gdown 방법 실패: {e}")

    # 방법 2: requests 시도
    if download_from_google_drive_requests(file_id, output_path):
        return True

    logger.error("❌ 모든 다운로드 방법 실패")
    return False


def verify_model_file(file_path: str) -> bool:
    """모델 파일 유효성 검증"""
    try:
        if not os.path.exists(file_path):
            return False

        file_size = os.path.getsize(file_path)
        if file_size < 1024:  # 1KB 미만이면 무효
            logger.warning(f"⚠️ 파일 크기가 너무 작음: {file_size} bytes")
            return False

        # Keras 모델 파일인지 확인
        if file_path.endswith('.keras'):
            try:
                import tensorflow as tf
                # 모델 헤더만 확인 (전체 로딩하지 않음)
                with open(file_path, 'rb') as f:
                    header = f.read(100)
                    if b'keras' in header.lower() or b'tensorflow' in header.lower():
                        logger.info(f"✅ Keras 모델 파일 검증 완료: {file_size:,} bytes")
                        return True
            except Exception as e:
                logger.warning(f"⚠️ Keras 모델 검증 실패: {e}")

        # Pickle 파일인지 확인
        elif file_path.endswith('.pkl'):
            try:
                with open(file_path, 'rb') as f:
                    # pickle 헤더 확인
                    header = f.read(10)
                    if header.startswith(b'\x80'):  # pickle protocol
                        logger.info(f"✅ Pickle 파일 검증 완료: {file_size:,} bytes")
                        return True
            except Exception as e:
                logger.warning(f"⚠️ Pickle 파일 검증 실패: {e}")

        # 기본적으로 크기가 충분하면 유효하다고 판단
        logger.info(f"✅ 파일 기본 검증 완료: {file_size:,} bytes")
        return True

    except Exception as e:
        logger.error(f"❌ 파일 검증 중 오류: {e}")
        return False


def load_emotion_model():
    """감정 분석 모델 로딩"""
    global _emotion_model

    if _emotion_model is not None:
        return _emotion_model

    try:
        # 모델 파일 다운로드 (필요한 경우)
        if not verify_model_file(str(EMOTION_MODEL_PATH)):
            logger.info("📥 감정 모델 파일 다운로드 중...")
            if not download_model_file(EMOTION_MODEL_FILE_ID, str(EMOTION_MODEL_PATH)):
                raise Exception("감정 모델 파일 다운로드 실패")

        # 모델 로딩
        logger.info(f"🤖 감정 모델 로딩 시작: {EMOTION_MODEL_PATH}")

        import tensorflow as tf

        # Keras 모델 로딩
        _emotion_model = tf.keras.models.load_model(str(EMOTION_MODEL_PATH), compile=False)

        logger.info(f"✅ 감정 모델 로딩 완료")
        logger.info(f"  - 입력 shape: {_emotion_model.input_shape}")
        logger.info(f"  - 출력 shape: {_emotion_model.output_shape}")

        return _emotion_model

    except Exception as e:
        logger.error(f"❌ 감정 모델 로딩 실패: {e}")
        _emotion_model = None
        return None


def load_vectorizer():
    """벡터라이저 로딩 (로컬 파일 사용)"""
    global _vectorizer

    if _vectorizer is not None:
        return _vectorizer

    try:
        if os.path.exists(VECTORIZER_PATH):
            logger.info(f"📊 벡터라이저 로딩 시작 (로컬 파일): {VECTORIZER_PATH}")

            # 파일 크기 확인
            file_size = os.path.getsize(VECTORIZER_PATH)
            logger.info(f"📊 벡터라이저 파일 크기: {file_size:,} bytes")

            with open(VECTORIZER_PATH, 'rb') as f:
                _vectorizer = pickle.load(f)

            logger.info("✅ 벡터라이저 로딩 완료 (로컬 파일)")
            return _vectorizer
        else:
            logger.warning("⚠️ 벡터라이저 파일이 없음 - 모델 내장 전처리 사용")
            logger.warning(f"  예상 경로: {VECTORIZER_PATH}")
            return None

    except Exception as e:
        logger.error(f"❌ 벡터라이저 로딩 실패: {e}")
        _vectorizer = None
        return None


def get_emotion_model():
    """감정 모델 getter"""
    return load_emotion_model()


def get_vectorizer():
    """벡터라이저 getter"""
    return load_vectorizer()


def initialize_emotion_models() -> Tuple[bool, str]:
    """감정 분석 모델 초기화"""
    global _model_loaded

    try:
        logger.info("🎭 감정 분석 모델 초기화 시작...")

        # 모델 파일 확인 (감정 모델은 다운로드, 벡터라이저는 로컬)
        model_available = verify_model_file(str(EMOTION_MODEL_PATH))
        vectorizer_available = os.path.exists(VECTORIZER_PATH)  # 로컬 파일만 확인

        logger.info(f"📋 모델 파일 상태:")
        logger.info(f"  - 감정 모델: {'✅ 존재' if model_available else '❌ 다운로드 필요'}")
        logger.info(f"  - 벡터라이저: {'✅ 존재 (로컬)' if vectorizer_available else '❌ 없음 (로컬)'}")

        # 감정 모델 다운로드 (필요한 경우)
        if not model_available:
            logger.info("📥 감정 모델 파일 다운로드 시도...")
            if EMOTION_MODEL_FILE_ID:
                if download_model_file(EMOTION_MODEL_FILE_ID, str(EMOTION_MODEL_PATH)):
                    model_available = True
                    logger.info("✅ 감정 모델 다운로드 완료")
                else:
                    return False, "감정 모델 파일 다운로드 실패"
            else:
                return False, "감정 모델 파일 ID가 설정되지 않음"

        # 벡터라이저 확인 (로컬 파일)
        if not vectorizer_available:
            logger.warning(f"⚠️ 벡터라이저 파일이 없습니다: {VECTORIZER_PATH}")
            logger.warning("  모델 내장 전처리를 사용하거나 룰 기반으로 동작합니다")

        # 모델 로딩 테스트
        if model_available:
            model = load_emotion_model()
            vectorizer = load_vectorizer()  # 실패해도 계속 진행

            if model is not None:
                logger.info("✅ 감정 모델 초기화 완료")
                logger.info(f"  - 감정 모델: ✅ 로드됨")
                logger.info(f"  - 벡터라이저: {'✅ 로드됨' if vectorizer is not None else '❌ 없음 (모델 내장 사용)'}")
                _model_loaded = True
                return True, "감정 모델 초기화 성공"
            else:
                return False, "감정 모델 로딩 실패"
        else:
            return False, "감정 모델 파일 준비 실패"

    except Exception as e:
        logger.error(f"❌ 감정 모델 초기화 실패: {e}")
        return False, f"초기화 중 오류: {str(e)}"


def is_model_available() -> bool:
    """모델 사용 가능 여부 확인"""
    return _model_loaded and _emotion_model is not None


def get_model_status() -> dict:
    """모델 상태 정보 반환"""
    return {
        "model_loaded": _model_loaded,
        "emotion_model_available": _emotion_model is not None,
        "vectorizer_available": _vectorizer is not None,
        "emotion_model_path": str(EMOTION_MODEL_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
        "emotion_model_exists": os.path.exists(EMOTION_MODEL_PATH),
        "vectorizer_exists": os.path.exists(VECTORIZER_PATH),
        "emotion_model_file_id": EMOTION_MODEL_FILE_ID,
        "vectorizer_source": "로컬 파일 (Git 포함)",  # 🆕 벡터라이저 소스 명시
        "emotion_model_source": "Google Drive 다운로드",  # 🆕 감정 모델 소스 명시
        "vectorizer_file_id": "N/A (로컬 파일)"  # 🆕 File ID 없음 표시
    }


# 모듈 임포트 시 자동 초기화 시도 (선택사항)
def auto_initialize():
    """자동 초기화 (필요시)"""
    if not _model_loaded:
        success, message = initialize_emotion_models()
        if success:
            logger.info("🎯 감정 모델 자동 초기화 성공")
        else:
            logger.warning(f"⚠️ 감정 모델 자동 초기화 실패: {message}")

# 필요한 경우 주석 해제
# auto_initialize()