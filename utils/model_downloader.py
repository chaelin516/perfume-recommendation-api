# utils/model_downloader.py
# Google Drive에서 감정 분석 모델 자동 다운로드

import os
import requests
import logging
from typing import Tuple, Optional
from pathlib import Path
import gdown

logger = logging.getLogger(__name__)

# 🔗 Google Drive 파일 정보
EMOTION_MODEL_CONFIG = {
    "file_id": "1JYUJvKVb44p63ctWe3c1G_qmdcDr-Xix",
    "filename": "emotion_analysis_model.pkl",
    "local_path": "./models/emotion_analysis_model.pkl",
    "description": "한국어 텍스트 감정 분석 모델 (scikit-learn)",
    "expected_size_mb": 50,  # 예상 파일 크기 (MB)
}


class EmotionModelDownloader:
    """Google Drive에서 감정 분석 모델을 다운로드하는 클래스"""

    def __init__(self):
        self.config = EMOTION_MODEL_CONFIG
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)

    def check_model_exists(self) -> Tuple[bool, Optional[str]]:
        """로컬에 모델이 존재하는지 확인"""
        model_path = Path(self.config["local_path"])

        if model_path.exists():
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 로컬 모델 발견: {model_path} ({file_size_mb:.1f}MB)")

            # 파일 크기 검증
            expected_size = self.config["expected_size_mb"]
            if file_size_mb < expected_size * 0.1:  # 예상 크기의 10% 미만이면 손상된 것으로 간주
                logger.warning(f"⚠️ 모델 파일이 너무 작습니다: {file_size_mb:.1f}MB (예상: {expected_size}MB)")
                return False, "파일 크기 부족"

            return True, str(model_path)

        logger.info("❌ 로컬 모델이 없습니다")
        return False, None

    def download_model_from_gdrive(self) -> Tuple[bool, str]:
        """Google Drive에서 모델 다운로드"""
        try:
            file_id = self.config["file_id"]
            output_path = self.config["local_path"]

            logger.info(f"📥 Google Drive에서 감정 분석 모델 다운로드 시작...")
            logger.info(f"  - 파일 ID: {file_id}")
            logger.info(f"  - 저장 경로: {output_path}")

            # Google Drive 직접 다운로드 URL 생성
            download_url = f"https://drive.google.com/uc?id={file_id}"

            # gdown 라이브러리 사용 (Google Drive 다운로드에 최적화)
            try:
                import gdown

                logger.info("📦 gdown 라이브러리로 다운로드 시도...")
                gdown.download(download_url, output_path, quiet=False)

            except ImportError:
                logger.warning("⚠️ gdown 라이브러리가 없습니다. requests로 시도...")

                # requests를 사용한 대안 방법
                session = requests.Session()

                # 먼저 다운로드 확인 페이지 접근
                response = session.get(download_url)

                # 다운로드 확인 토큰 추출
                for line in response.text.split('\n'):
                    if 'confirm=' in line and 'download' in line:
                        confirm_token = line.split('confirm=')[1].split('&')[0]
                        break
                else:
                    confirm_token = None

                # 실제 다운로드
                if confirm_token:
                    download_url = f"https://drive.google.com/uc?confirm={confirm_token}&id={file_id}"

                with session.get(download_url, stream=True) as r:
                    r.raise_for_status()

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    with open(output_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

            # 다운로드 검증
            if os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✅ 모델 다운로드 완료: {file_size_mb:.1f}MB")

                # 파일 크기 검증
                expected_size = self.config["expected_size_mb"]
                if file_size_mb < expected_size * 0.1:
                    logger.error(f"❌ 다운로드된 파일 크기가 비정상적입니다: {file_size_mb:.1f}MB")
                    os.remove(output_path)
                    return False, f"파일 크기 오류: {file_size_mb:.1f}MB"

                return True, output_path
            else:
                return False, "파일이 생성되지 않았습니다"

        except Exception as e:
            logger.error(f"❌ 모델 다운로드 실패: {e}")
            return False, str(e)

    def download_model_with_fallback(self) -> Tuple[bool, str]:
        """여러 방법으로 모델 다운로드 시도"""
        file_id = self.config["file_id"]
        output_path = self.config["local_path"]

        # 방법 1: gdown 라이브러리
        try:
            import gdown
            logger.info("📦 방법 1: gdown 라이브러리로 다운로드 시도...")

            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

            if os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✅ gdown으로 다운로드 성공: {file_size_mb:.1f}MB")
                return True, output_path

        except Exception as e:
            logger.warning(f"⚠️ gdown 다운로드 실패: {e}")

        # 방법 2: requests 직접 다운로드
        try:
            logger.info("📦 방법 2: requests로 직접 다운로드 시도...")

            download_urls = [
                f"https://drive.google.com/uc?export=download&id={file_id}",
                f"https://drive.google.com/uc?id={file_id}&export=download",
                f"https://docs.google.com/uc?export=download&id={file_id}"
            ]

            for i, url in enumerate(download_urls, 1):
                try:
                    logger.info(f"  시도 {i}: {url[:60]}...")

                    session = requests.Session()
                    response = session.get(url, stream=True)

                    if response.status_code == 200:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=32768):
                                f.write(chunk)

                        if os.path.exists(output_path):
                            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

                            if file_size_mb > 1:  # 최소 1MB 이상
                                logger.info(f"✅ requests로 다운로드 성공 (방법 {i}): {file_size_mb:.1f}MB")
                                return True, output_path
                            else:
                                logger.warning(f"⚠️ 다운로드된 파일이 너무 작습니다: {file_size_mb:.1f}MB")
                                os.remove(output_path)

                except Exception as e:
                    logger.warning(f"⚠️ 방법 {i} 실패: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ requests 다운로드 실패: {e}")

        return False, "모든 다운로드 방법 실패"

    def ensure_model_available(self) -> Tuple[bool, str]:
        """모델이 사용 가능하도록 보장 (없으면 다운로드)"""
        logger.info("🔍 감정 분석 모델 가용성 확인...")

        # 1. 로컬에 이미 있는지 확인
        exists, path = self.check_model_exists()
        if exists:
            logger.info("✅ 로컬 모델 사용 가능")
            return True, path

        # 2. 없으면 다운로드
        logger.info("📥 로컬에 모델이 없어 Google Drive에서 다운로드...")

        success, result = self.download_model_with_fallback()

        if success:
            logger.info(f"✅ 감정 분석 모델 준비 완료: {result}")
            return True, result
        else:
            logger.error(f"❌ 감정 분석 모델 다운로드 실패: {result}")
            return False, result

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        exists, path = self.check_model_exists()

        info = {
            "file_id": self.config["file_id"],
            "filename": self.config["filename"],
            "description": self.config["description"],
            "local_path": self.config["local_path"],
            "exists": exists,
            "download_url": f"https://drive.google.com/uc?id={self.config['file_id']}"
        }

        if exists and path:
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            info.update({
                "file_size_mb": round(file_size_mb, 2),
                "absolute_path": os.path.abspath(path)
            })

        return info


# 🌟 전역 인스턴스
emotion_model_downloader = EmotionModelDownloader()


def download_emotion_model() -> Tuple[bool, str]:
    """감정 분석 모델 다운로드 (편의 함수)"""
    return emotion_model_downloader.ensure_model_available()


def get_emotion_model_path() -> Optional[str]:
    """감정 분석 모델 경로 반환"""
    exists, path = emotion_model_downloader.check_model_exists()
    return path if exists else None


def install_gdown_if_missing():
    """gdown 라이브러리가 없으면 설치 시도"""
    try:
        import gdown
        logger.info("✅ gdown 라이브러리 사용 가능")
        return True
    except ImportError:
        logger.warning("⚠️ gdown 라이브러리가 설치되지 않았습니다")
        logger.info("📦 gdown 설치 시도...")

        try:
            import subprocess
            import sys

            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            logger.info("✅ gdown 설치 완료")
            return True
        except Exception as e:
            logger.error(f"❌ gdown 설치 실패: {e}")
            logger.info("💡 수동 설치: pip install gdown")
            return False


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)

    print("🧪 감정 분석 모델 다운로더 테스트")
    print("=" * 50)

    downloader = EmotionModelDownloader()

    # 모델 정보 출력
    info = downloader.get_model_info()
    print(f"📋 모델 정보:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)

    # 다운로드 테스트
    success, message = downloader.ensure_model_available()

    if success:
        print(f"✅ 성공: {message}")
    else:
        print(f"❌ 실패: {message}")