# utils/email_sender.py
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional


class EmailSender:
    def __init__(self):
        # 환경변수에서 SMTP 설정 가져오기
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', 'cswingzz83@gmail.com')
        self.smtp_password = os.getenv('SMTP_PASSWORD', 'mdwk owvs ffgg rusy')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)

    def send_verification_email(self, to_email: str, verification_link: str, user_name: str = "사용자") -> bool:
        """이메일 인증 메일 발송"""
        try:
            # 이메일 내용 구성
            subject = "ScentRoute - 이메일 인증을 완료해주세요"

            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>이메일 인증</title>
            </head>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #4CAF50;">ScentRoute 이메일 인증</h2>

                    <p>안녕하세요, {user_name}님!</p>

                    <p>ScentRoute에 가입해주셔서 감사합니다. 아래 버튼을 클릭하여 이메일 인증을 완료해주세요.</p>

                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{verification_link}" 
                           style="background-color: #4CAF50; color: white; padding: 12px 30px; 
                                  text-decoration: none; border-radius: 5px; display: inline-block;
                                  font-weight: bold;">
                            이메일 인증하기
                        </a>
                    </div>

                    <p>버튼이 작동하지 않는다면, 아래 링크를 복사하여 브라우저에 직접 입력해주세요:</p>
                    <p style="word-break: break-all; background-color: #f5f5f5; padding: 10px; border-radius: 3px;">
                        {verification_link}
                    </p>

                    <hr style="margin: 30px 0;">
                    <p style="font-size: 12px; color: #666;">
                        이 이메일을 요청하지 않으셨다면 무시해주세요.<br>
                        ScentRoute 팀 드림
                    </p>
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
            """

            return self._send_email(to_email, subject, html_body, text_body)

        except Exception as e:
            print(f"❌ 인증 이메일 발송 실패: {e}")
            return False

    def _send_email(self, to_email: str, subject: str, html_body: str, text_body: str) -> bool:
        """실제 이메일 발송"""
        try:
            # SMTP 설정 확인
            if not self.smtp_username or not self.smtp_password:
                print("❌ SMTP 인증 정보가 설정되지 않았습니다.")
                return False

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

            # SMTP 서버 연결 및 발송
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            print(f"✅ 이메일 발송 성공: {to_email}")
            return True

        except Exception as e:
            print(f"❌ 이메일 발송 실패: {e}")
            return False


# 전역 이메일 발송자 인스턴스
email_sender = EmailSender()