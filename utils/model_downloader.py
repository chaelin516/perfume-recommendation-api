# utils/model_downloader.py
# 🤖 Google Drive에서 AI 모델 자동 다운로드 시스템

import os
import requests
import logging
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import pickle
import json

logger = logging.getLogger("model_downloader")


class ModelDownloader:
    """AI 모델 자동 다운로드 클래스"""

    def __init__(self):
        # 🔗 Google Drive 파일 설정들
        self.file_configs = {
            "emotion_model": {
                "file_id": "1JYUJvKVb44p63ctWe3c1G_qmdcDr-Xix",  # ✅ 실제 감정 모델 파일 ID
                "local_path": "emotion_models/scent_emotion_model_v6.keras",
                "expected_size_mb": 1200,  # 1.2GB
                "description": "감정 태깅 AI 모델",
                "required": True,
                "backup_urls": []  # 백업 다운로드 URL (필요시)
            },
            "vectorizer": {
                "file_id": "CREATE_DUMMY_VECTORIZER",  # 🔧 더미 벡터라이저 생성
                "local_path": "emotion_models/vectorizer.pkl",
                "expected_size_mb": 1,
                "description": "텍스트 벡터라이저",
                "required": False,  # 없어도 더미로 생성 가능
                "backup_urls": []
            }
        }

        # 다운로드 설정
        self.chunk_size = 8192 * 8  # 64KB chunks
        self.timeout = 30  # 30초 타임아웃
        self.max_retries = 3
        self.retry_delay = 5  # 5초 대기

    def check_file_integrity(self, file_path: str, expected_size_mb: float) -> bool:
        """파일 무결성 검사"""
        try:
            if not os.path.exists(file_path):
                return False

            file_size = os.path.getsize(file_path)
            expected_size_bytes = expected_size_mb * 1024 * 1024

            # 크기 검증 (±10% 허용)
            size_tolerance = 0.1
            min_size = expected_size_bytes * (1 - size_tolerance)
            max_size = expected_size_bytes * (1 + size_tolerance)

            if min_size <= file_size <= max_size:
                logger.info(f"✅ 파일 크기 검증 통과: {file_size:,} bytes (예상: {expected_size_bytes:,})")
                return True
            else:
                logger.warning(f"⚠️ 파일 크기 불일치: {file_size:,} bytes, 예상: {expected_size_bytes:,}")
                return False

        except Exception as e:
            logger.error(f"❌ 파일 검증 중 오류: {e}")
            return False

    def get_google_drive_download_url(self, file_id: str) -> str:
        """Google Drive 직접 다운로드 URL 생성"""
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    def handle_google_drive_virus_warning(self, response_text: str, file_id: str) -> Optional[str]:
        """Google Drive 바이러스 경고 처리"""
        try:
            # confirm 토큰 찾기
            if "download_warning" in response_text or "virus scan warning" in response_text:
                # 여러 방법으로 confirm 토큰 추출 시도
                confirm_patterns = [
                    'name="confirm" value="',
                    'confirm=',
                    'confirm&amp;'
                ]

                for pattern in confirm_patterns:
                    if pattern in response_text:
                        start = response_text.find(pattern) + len(pattern)
                        end = response_text.find('"', start) if '"' in response_text[start:start + 50] else start + 10
                        confirm_token = response_text[start:end].split('&')[0]

                        if confirm_token and len(confirm_token) < 50:  # 유효한 토큰인지 간단 검증
                            logger.info(f"🔓 Google Drive 확인 토큰 발견: {confirm_token[:10]}...")
                            return f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"

                # 기본 확인 토큰 시도
                logger.info("🔓 기본 확인 토큰 사용")
                return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"

        except Exception as e:
            logger.warning(f"⚠️ 바이러스 경고 처리 중 오류: {e}")

        return None

    def download_with_progress(self, url: str, destination: str, description: str) -> bool:
        """진행률 표시와 함께 파일 다운로드"""
        try:
            session = requests.Session()

            # User-Agent 설정 (일부 제한 우회)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            session.headers.update(headers)

            logger.info(f"📡 다운로드 시작: {description}")

            response = session.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()

            # Content-Length 헤더에서 파일 크기 가져오기
            total_size = int(response.headers.get('content-length', 0))

            if total_size == 0:
                logger.warning("⚠️ 파일 크기 정보를 가져올 수 없음")

            downloaded_size = 0
            last_progress_time = time.time()
            last_downloaded_size = 0

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # 진행률 표시 (5초마다 또는 10MB마다)
                        current_time = time.time()
                        if (current_time - last_progress_time >= 5) or (
                                downloaded_size - last_downloaded_size >= 10 * 1024 * 1024):
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                speed_mbps = (downloaded_size - last_downloaded_size) / (
                                            current_time - last_progress_time) / 1024 / 1024
                                eta_seconds = (total_size - downloaded_size) / (
                                            speed_mbps * 1024 * 1024) if speed_mbps > 0 else 0

                                logger.info(f"📥 {description}: {progress:.1f}% "
                                            f"({downloaded_size:,}/{total_size:,} bytes) "
                                            f"속도: {speed_mbps:.1f}MB/s, "
                                            f"남은시간: {int(eta_seconds // 60)}분 {int(eta_seconds % 60)}초")
                            else:
                                logger.info(f"📥 {description}: {downloaded_size:,} bytes 다운로드됨")

                            last_progress_time = current_time
                            last_downloaded_size = downloaded_size

            logger.info(f"✅ {description} 다운로드 완료: {downloaded_size:,} bytes")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 네트워크 오류: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 다운로드 오류: {e}")
            return False

    def download_from_google_drive(self, file_id: str, destination: str, description: str,
                                   expected_size_mb: float) -> bool:
        """Google Drive에서 파일 다운로드 (재시도 로직 포함)"""

        # 이미 파일이 존재하고 유효한지 확인
        if self.check_file_integrity(destination, expected_size_mb):
            logger.info(f"✅ {description} 파일이 이미 존재하고 유효함: {destination}")
            return True

        # 디렉토리 생성
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # 기존 불완전한 파일 제거
        if os.path.exists(destination):
            os.remove(destination)
            logger.info(f"🗑️ 기존 불완전한 파일 제거: {destination}")

        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 {description} 다운로드 시도 {attempt + 1}/{self.max_retries}")

                # 첫 번째 요청으로 다운로드 URL 확인
                session = requests.Session()
                initial_url = self.get_google_drive_download_url(file_id)

                response = session.get(initial_url, stream=True, timeout=self.timeout)

                # Google Drive 바이러스 경고 처리
                if response.status_code == 200 and len(response.content) < 1024 * 1024:  # 1MB 미만이면 경고 페이지일 수 있음
                    response_text = response.text
                    if "download_warning" in response_text or "virus scan warning" in response_text:
                        logger.info("🦠 Google Drive 바이러스 스캔 경고 감지, 확인 토큰 처리...")
                        confirmed_url = self.handle_google_drive_virus_warning(response_text, file_id)

                        if confirmed_url:
                            # 확인 토큰이 포함된 URL로 다시 시도
                            if self.download_with_progress(confirmed_url, destination, description):
                                if self.check_file_integrity(destination, expected_size_mb):
                                    return True
                                else:
                                    logger.warning(f"⚠️ 다운로드된 파일 검증 실패: {destination}")
                        else:
                            logger.error("❌ 확인 토큰을 찾을 수 없음")
                else:
                    # 일반적인 다운로드
                    response.close()  # 기존 연결 닫기
                    if self.download_with_progress(initial_url, destination, description):
                        if self.check_file_integrity(destination, expected_size_mb):
                            return True
                        else:
                            logger.warning(f"⚠️ 다운로드된 파일 검증 실패: {destination}")

            except Exception as e:
                logger.error(f"❌ 다운로드 시도 {attempt + 1} 실패: {e}")

            # 재시도 전 대기
            if attempt < self.max_retries - 1:
                logger.info(f"⏳ {self.retry_delay}초 후 재시도...")
                time.sleep(self.retry_delay)

        logger.error(f"❌ {description} 다운로드 최종 실패 (모든 재시도 소진)")
        return False

    def create_dummy_vectorizer(self) -> bool:
        """더미 벡터라이저 생성 (vectorizer.pkl이 없을 때)"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            logger.info("🔧 더미 벡터라이저 생성 중...")

            # 간단한 TF-IDF 벡터라이저 생성
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words=None,
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )

            # 더미 데이터로 fit
            dummy_texts = [
                "좋아요 향수 기분 좋아 행복해 사랑해",
                "나빠요 싫어 화나 짜증나 별로야",
                "슬퍼요 우울해 눈물 외로워 쓸쓸해",
                "신나요 흥미로워 에너지 활기차 즐거워",
                "불안해 걱정돼 두려워 긴장돼 스트레스",
                "당황스러워 놀라워 혼란스러워 의외야",
                "상처받아 아파 실망스러워 서운해",
                "완벽해 최고야 훌륭해 멋져 환상적"
            ]

            vectorizer.fit(dummy_texts)

            # 저장
            vectorizer_path = "emotion_models/vectorizer.pkl"
            os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)

            logger.info(f"✅ 더미 벡터라이저 생성 완료: {vectorizer_path}")
            return True

        except Exception as e:
            logger.error(f"❌ 더미 벡터라이저 생성 실패: {e}")
            return False

    def ensure_models_available(self) -> Dict[str, bool]:
        """모든 필요한 모델이 사용 가능한지 확인하고 다운로드"""
        results = {}

        logger.info("🤖 AI 모델 가용성 확인 시작...")

        for model_name, config in self.file_configs.items():
            file_id = config["file_id"]
            local_path = config["local_path"]
            expected_size_mb = config["expected_size_mb"]
            description = config["description"]
            required = config.get("required", True)

            logger.info(f"🔍 {description} 확인 중...")

            # 파일 ID가 설정되지 않은 경우
            if file_id.startswith("YOUR_"):
                logger.warning(f"⚠️ {description} 파일 ID가 설정되지 않음: {file_id}")

                if model_name == "vectorizer":
                    # 벡터라이저는 더미로 생성 가능
                    results[model_name] = self.create_dummy_vectorizer()
                else:
                    results[model_name] = False
                continue

            # 다운로드 시도
            success = self.download_from_google_drive(file_id, local_path, description, expected_size_mb)
            results[model_name] = success

            if success:
                logger.info(f"✅ {description} 준비 완료")
            else:
                if required:
                    logger.error(f"❌ 필수 모델 {description} 다운로드 실패")
                else:
                    logger.warning(f"⚠️ 선택적 모델 {description} 다운로드 실패")

                    # 벡터라이저가 실패한 경우 더미 생성 시도
                    if model_name == "vectorizer":
                        logger.info("🔧 벡터라이저 더미 생성 시도...")
                        results[model_name] = self.create_dummy_vectorizer()

        # 결과 요약
        successful_models = sum(1 for success in results.values() if success)
        total_models = len(results)

        logger.info(f"📊 모델 준비 완료: {successful_models}/{total_models}")

        # 필수 모델 확인
        emotion_model_ready = results.get("emotion_model", False)

        if emotion_model_ready:
            logger.info("🎉 감정 태깅 시스템 준비 완료!")
        else:
            logger.error("❌ 감정 태깅 시스템 준비 실패 - 핵심 모델 없음")

        return results

    def get_download_status(self) -> Dict[str, Any]:
        """다운로드 상태 정보 반환"""
        status = {
            "models": {},
            "system_info": {
                "timestamp": datetime.now().isoformat(),
                "python_version": os.sys.version,
                "working_directory": os.getcwd()
            }
        }

        for model_name, config in self.file_configs.items():
            local_path = config["local_path"]
            expected_size_mb = config["expected_size_mb"]

            model_status = {
                "local_path": local_path,
                "exists": os.path.exists(local_path),
                "file_size_mb": 0,
                "is_valid": False,
                "description": config["description"]
            }

            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                model_status["file_size_mb"] = round(file_size / 1024 / 1024, 2)
                model_status["is_valid"] = self.check_file_integrity(local_path, expected_size_mb)

            status["models"][model_name] = model_status

        return status

    def update_file_id(self, model_name: str, new_file_id: str) -> bool:
        """파일 ID 업데이트 (런타임에서)"""
        if model_name in self.file_configs:
            old_id = self.file_configs[model_name]["file_id"]
            self.file_configs[model_name]["file_id"] = new_file_id
            logger.info(f"🔄 {model_name} 파일 ID 업데이트: {old_id[:10]}... → {new_file_id[:10]}...")
            return True
        else:
            logger.error(f"❌ 알 수 없는 모델: {model_name}")
            return False


# ─── 전역 다운로더 인스턴스 ─────────────────────────────────────────────────────
model_downloader = ModelDownloader()


# ─── 유틸리티 함수들 ─────────────────────────────────────────────────────────────
async def ensure_emotion_model_available() -> bool:
    """감정 모델이 사용 가능한지 확인 (비동기 래퍼)"""
    try:
        results = model_downloader.ensure_models_available()
        return results.get("emotion_model", False)
    except Exception as e:
        logger.error(f"❌ 감정 모델 확인 중 오류: {e}")
        return False


def check_emotion_model_status() -> Dict[str, Any]:
    """감정 모델 상태 확인"""
    return model_downloader.get_download_status()


def download_emotion_model_sync() -> bool:
    """감정 모델 동기 다운로드"""
    try:
        config = model_downloader.file_configs["emotion_model"]
        return model_downloader.download_from_google_drive(
            file_id=config["file_id"],
            destination=config["local_path"],
            description=config["description"],
            expected_size_mb=config["expected_size_mb"]
        )
    except Exception as e:
        logger.error(f"❌ 감정 모델 다운로드 중 오류: {e}")
        return False


# ─── 테스트 함수 ─────────────────────────────────────────────────────────────────
def test_downloader():
    """다운로더 테스트 함수"""
    print("🧪 모델 다운로더 테스트 시작...")

    # 현재 상태 확인
    status = model_downloader.get_download_status()
    print(f"📊 현재 상태: {status}")

    # 모델 다운로드 시도
    results = model_downloader.ensure_models_available()
    print(f"📥 다운로드 결과: {results}")

    # 최종 상태 확인
    final_status = model_downloader.get_download_status()
    print(f"📊 최종 상태: {final_status}")

    print("✅ 테스트 완료!")


if __name__ == "__main__":
    # 직접 실행 시 테스트
    test_downloader()