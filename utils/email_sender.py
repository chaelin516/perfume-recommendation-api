# utils/email_sender.py - 향상된 로깅과 에러 처리
import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple
from datetime import datetime

# 로거 설정
logger = logging.getLogger(__name__)


class EmailSender:
    def __init__(self):
        # 환경변수에서 SMTP 설정 가져오기
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = int('587')
        self.smtp_username = 'cswingzz74@gmail.com'
        self.smtp_password = 'jrjz fent efzd huri'
        self.from_email = 'no-reply@perfume-recommendation-api-1.onrender.com'

        # 설정 로깅
        logger.info("📧 이메일 발송자 초기화")
        logger.info(f"  - SMTP 서버: {self.smtp_server}:{self.smtp_port}")
        logger.info(f"  - 발신 이메일: {self.from_email}")
        logger.info(f"  - SMTP 사용자명: {'설정됨' if self.smtp_username else '없음'}")
        logger.info(f"  - SMTP 비밀번호: {'설정됨' if self.smtp_password else '없음'}")

    def check_smtp_config(self) -> Tuple[bool, str]:
        """SMTP 설정 유효성 검사"""
        if not self.smtp_username:
            return False, "SMTP_USERNAME 환경변수가 설정되지 않았습니다."

        if not self.smtp_password:
            return False, "SMTP_PASSWORD 환경변수가 설정되지 않았습니다."

        if not self.from_email:
            return False, "FROM_EMAIL 환경변수가 설정되지 않았습니다."

        return True, "SMTP 설정이 올바릅니다."

    def send_verification_email(self, to_email: str, verification_link: str, user_name: str = "사용자") -> Tuple[
        bool, str]:
        """이메일 인증 메일 발송 - 상세 로깅 포함"""
        start_time = datetime.now()
        logger.info(f"📧 이메일 인증 발송 시작")
        logger.info(f"  - 수신자: {to_email}")
        logger.info(f"  - 사용자명: {user_name}")
        logger.info(f"  - 시작 시간: {start_time}")

        try:
            # SMTP 설정 검증
            config_valid, config_message = self.check_smtp_config()
            if not config_valid:
                logger.error(f"❌ SMTP 설정 오류: {config_message}")
                return False, config_message

            logger.info(f"✅ SMTP 설정 검증 완료")

            # 이메일 내용 구성
            subject = "ScentRoute - 이메일 인증을 완료해주세요"

            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>이메일 인증</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ color: #4CAF50; text-align: center; }}
                    .button {{ 
                        background-color: #4CAF50; 
                        color: white; 
                        padding: 12px 30px; 
                        text-decoration: none; 
                        border-radius: 5px; 
                        display: inline-block;
                        font-weight: bold;
                        margin: 20px 0;
                    }}
                    .link-box {{ 
                        word-break: break-all; 
                        background-color: #f5f5f5; 
                        padding: 10px; 
                        border-radius: 3px; 
                        margin: 10px 0;
                    }}
                    .footer {{ font-size: 12px; color: #666; margin-top: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="header">🌸 ScentRoute 이메일 인증</h2>

                    <p>안녕하세요, <strong>{user_name}</strong>님!</p>

                    <p>ScentRoute에 가입해주셔서 감사합니다. 아래 버튼을 클릭하여 이메일 인증을 완료해주세요.</p>

                    <div style="text-align: center;">
                        <a href="{verification_link}" class="button">
                            ✅ 이메일 인증하기
                        </a>
                    </div>

                    <p>버튼이 작동하지 않는다면, 아래 링크를 복사하여 브라우저에 직접 입력해주세요:</p>
                    <div class="link-box">
                        {verification_link}
                    </div>

                    <div class="footer">
                        <hr>
                        <p>이 이메일을 요청하지 않으셨다면 무시해주세요.</p>
                        <p><strong>ScentRoute 팀</strong> 드림 🌹</p>
                        <p>발송 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
            </html>
            """

            text_body = f"""
ScentRoute 이메일 인증

안녕하세요, {user_name}님!

ScentRoute에 가입해주셔서 감사합니다. 
아래 링크를 클릭하여 이메일 인증을 완료해주세요.

{verification_link}

이 이메일을 요청하지 않으셨다면 무시해주세요.

ScentRoute 팀 드림
발송 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
            """

            logger.info(f"📝 이메일 내용 생성 완료 (HTML: {len(html_body)}자, TEXT: {len(text_body)}자)")

            # 실제 이메일 발송
            success, message = self._send_email(to_email, subject, html_body, text_body)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if success:
                logger.info(f"✅ 이메일 발송 성공!")
                logger.info(f"  - 수신자: {to_email}")
                logger.info(f"  - 소요 시간: {duration:.2f}초")
                logger.info(f"  - 완료 시간: {end_time}")
                return True, f"이메일이 성공적으로 발송되었습니다. (소요시간: {duration:.2f}초)"
            else:
                logger.error(f"❌ 이메일 발송 실패!")
                logger.error(f"  - 오류 메시지: {message}")
                logger.error(f"  - 소요 시간: {duration:.2f}초")
                return False, f"이메일 발송 실패: {message}"

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            error_msg = f"이메일 발송 중 예외 발생: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"  - Exception Type: {type(e).__name__}")
            logger.error(f"  - 소요 시간: {duration:.2f}초")
            return False, error_msg

    def _send_email(self, to_email: str, subject: str, html_body: str, text_body: str) -> Tuple[bool, str]:
        """실제 이메일 발송 - SMTP 연결 및 전송"""
        try:
            logger.info(f"🔌 SMTP 서버 연결 시도: {self.smtp_server}:{self.smtp_port}")

            # 메시지 구성
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            # 텍스트 및 HTML 파트 추가
            text_part = MIMEText(text_body, 'plain', 'utf-8')
            html_part = MIMEText(html_body, 'html', 'utf-8')

            msg.attach(text_part)
            msg.attach(html_part)

            logger.info(f"📨 이메일 메시지 구성 완료")
            logger.info(f"  - From: {self.from_email}")
            logger.info(f"  - To: {to_email}")
            logger.info(f"  - Subject: {subject}")

            # SMTP 서버 연결 및 발송
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                logger.info(f"🔗 SMTP 서버 연결 성공")

                # TLS 시작
                server.starttls()
                logger.info(f"🔒 TLS 암호화 활성화")

                # 로그인
                server.login(self.smtp_username, self.smtp_password)
                logger.info(f"🔑 SMTP 로그인 성공 (사용자: {self.smtp_username})")

                # 메일 발송
                result = server.send_message(msg)
                logger.info(f"📬 이메일 전송 완료")

                if result:
                    logger.warning(f"⚠️ 일부 수신자에게 발송 실패: {result}")
                    return False, f"일부 수신자 발송 실패: {result}"
                else:
                    logger.info(f"✅ 모든 수신자에게 발송 성공")
                    return True, "발송 성공"

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP 인증 실패: {str(e)} - Gmail 앱 비밀번호를 확인하세요."
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        except smtplib.SMTPRecipientsRefused as e:
            error_msg = f"수신자 거부됨: {str(e)} - 이메일 주소를 확인하세요."
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        except smtplib.SMTPServerDisconnected as e:
            error_msg = f"SMTP 서버 연결 끊김: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        except ConnectionRefusedError as e:
            error_msg = f"SMTP 서버 연결 거부: {str(e)} - 서버 주소와 포트를 확인하세요."
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"SMTP 발송 중 예외: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"  - Exception Type: {type(e).__name__}")
            return False, error_msg

    def test_smtp_connection(self) -> Tuple[bool, str]:
        """SMTP 연결 테스트"""
        logger.info(f"🧪 SMTP 연결 테스트 시작")

        try:
            # 설정 검증
            config_valid, config_message = self.check_smtp_config()
            if not config_valid:
                return False, config_message

            # 연결 테스트
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                logger.info(f"✅ SMTP 서버 연결 성공")

                server.starttls()
                logger.info(f"✅ TLS 암호화 성공")

                server.login(self.smtp_username, self.smtp_password)
                logger.info(f"✅ SMTP 인증 성공")

                return True, "SMTP 연결 테스트 성공"

        except Exception as e:
            error_msg = f"SMTP 연결 테스트 실패: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg


# 전역 이메일 발송자 인스턴스
email_sender = EmailSender()