# utils/file_utils.py
import json
import os
import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)


def load_json_file(file_path: str) -> List[Dict]:
    """JSON 파일 로딩"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 리스트가 아닌 경우 빈 리스트 반환
                if not isinstance(data, list):
                    logger.warning(f"⚠️ {file_path}가 리스트 형태가 아닙니다. 빈 리스트를 반환합니다.")
                    return []
                return data
        except json.JSONDecodeError as e:
            logger.error(f"❌ {file_path} JSON 파싱 오류: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ {file_path} 로딩 실패: {e}")
            return []
    else:
        logger.info(f"📁 {file_path} 파일이 존재하지 않습니다. 빈 리스트를 반환합니다.")
        return []


def save_json_file(file_path: str, data: List[Dict]) -> bool:
    """JSON 파일 저장"""
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ {file_path} 저장 완료")
        return True
    except Exception as e:
        logger.error(f"❌ {file_path} 저장 실패: {e}")
        return False


def load_json_dict(file_path: str) -> Dict:
    """JSON 파일을 딕셔너리로 로딩"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.warning(f"⚠️ {file_path}가 딕셔너리 형태가 아닙니다. 빈 딕셔너리를 반환합니다.")
                    return {}
                return data
        except json.JSONDecodeError as e:
            logger.error(f"❌ {file_path} JSON 파싱 오류: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ {file_path} 로딩 실패: {e}")
            return {}
    else:
        logger.info(f"📁 {file_path} 파일이 존재하지 않습니다. 빈 딕셔너리를 반환합니다.")
        return {}


def save_json_dict(file_path: str, data: Dict) -> bool:
    """딕셔너리를 JSON 파일로 저장"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ {file_path} 저장 완료")
        return True
    except Exception as e:
        logger.error(f"❌ {file_path} 저장 실패: {e}")
        return False


def ensure_data_directory(base_path: str = "data") -> bool:
    """데이터 디렉토리 및 기본 파일들 생성"""
    try:
        # 데이터 디렉토리 생성
        os.makedirs(base_path, exist_ok=True)

        # 기본 JSON 파일들 생성 (없는 경우)
        default_files = {
            os.path.join(base_path, "users.json"): [],
            os.path.join(base_path, "diary_data.json"): [],
            os.path.join(base_path, "temp_users.json"): [],
            os.path.join(base_path, "withdraw_logs.json"): []
        }

        for file_path, default_data in default_files.items():
            if not os.path.exists(file_path):
                save_json_file(file_path, default_data)
                logger.info(f"📁 기본 파일 생성: {file_path}")

        return True
    except Exception as e:
        logger.error(f"❌ 데이터 디렉토리 생성 실패: {e}")
        return False