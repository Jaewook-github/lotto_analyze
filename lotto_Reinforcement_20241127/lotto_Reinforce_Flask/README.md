## Synology NAS에서 Docker를 사용

1. 도커 이미지 생성을 위한 준비:

먼저 다음과 같은 파일 구조를 만듭니다:
```
lotto-prediction/
├── Dockerfile
├── requirements.txt
├── app.py
|-- config.py
|-- wsgi.py
├── templates/
│   └── index.html
└── data/
    ├── logs/
    ├── predictions/
    ├── models/
    └── lotto.db
```

2. Dockerfile 생성:
```dockerfile
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p /app/data/logs /app/data/predictions /app/data/models

# 포트 설정
EXPOSE 5000

# 애플리케이션 실행
CMD ["python", "app.py"]
```

3. requirements.txt 작성:
```
flask==2.0.1
numpy==1.21.0
pandas==1.3.0
matplotlib==3.4.2
openpyxl==3.0.7
SQLAlchemy==1.4.23
gunicorn==20.1.0
```

4. 도커 이미지 빌드 및 푸시:
```bash
# 이미지 빌드
docker build -t your-docker-hub-username/lotto-prediction:latest .

# Docker Hub에 푸시
docker push your-docker-hub-username/lotto-prediction:latest
```

5. Synology NAS에서 설정:

a) Docker 패키지 설치:
- Synology Package Center 열기
- Docker 검색 및 설치

b) Docker 컨테이너 생성:
1. Synology Docker 앱 실행
2. "Registry" 탭에서 이미지 다운로드:
   - 검색창에 `your-docker-hub-username/lotto-prediction` 입력
   - 이미지 다운로드

3. "Container" 탭에서 새 컨테이너 생성:
   - 다운로드한 이미지 선택
   - "Advanced Settings" 클릭
   
4. 고급 설정:
   - Volume 설정:
     ```
     호스트 경로: /volume1/docker/lotto-prediction/data
     컨테이너 경로: /app/data
     ```
   - Port 설정:
     ```
     호스트 포트: 5000
     컨테이너 포트: 5000
     ```
   - Environment 설정:
     ```
     FLASK_APP=app.py
     FLASK_ENV=production
     ```

5. 컨테이너 시작:
   - 설정 완료 후 "Apply" 클릭
   - 컨테이너 시작

6. 접속 테스트:
- 브라우저에서 `http://[시놀로지IP]:5000` 접속

주의사항:
1. 데이터 지속성:
   - 중요 데이터는 볼륨 마운트를 통해 호스트에 저장
   - DB 파일은 볼륨에 위치

2. 보안:
   - 방화벽 설정 확인
   - 필요한 포트만 개방
   - HTTPS 설정 권장

3. 리소스 관리:
   - CPU/메모리 제한 설정
   - 로그 rotate 설정

4. 모니터링:
   - Docker 로그 확인
   - 리소스 사용량 모니터링
-------------------------------------------------------
버전 호환성을 고려한 requirements.txt

```text
Flask==2.0.1
Werkzeug==2.0.1
numpy==1.21.0
pandas==1.3.0
matplotlib==3.4.2
python-dateutil==2.8.2
SQLAlchemy==1.4.23
openpyxl==3.0.7
Pillow==8.3.1
six==1.16.0
pytz==2021.1
click==8.0.1
itsdangerous==2.0.1
Jinja2==3.0.1
MarkupSafe==2.0.1
gunicorn==20.1.0
```

이 버전들은 서로 호환성이 검증되어 있으며, 주요 고려사항은 다음과 같습니다:

1. 핵심 라이브러리
   - Flask 2.0.1: 안정적인 웹 프레임워크 버전
   - numpy 1.21.0: pandas 1.3.0과 호환
   - pandas 1.3.0: numpy 1.21.0 이상 필요
   - matplotlib 3.4.2: numpy 1.21.0과 호환

2. 의존성 라이브러리
   - python-dateutil: pandas의 의존성
   - pytz: 시간대 처리용
   - Pillow: matplotlib의 이미지 처리용
   - six: 파이썬 2/3 호환성 라이브러리

3. Flask 관련 의존성
   - Werkzeug: Flask의 핵심 의존성
   - Jinja2: 템플릿 엔진
   - MarkupSafe: 템플릿 보안
   - click: 커맨드라인 인터페이스
   - itsdangerous: 데이터 서명

4. 배포 관련
   - gunicorn: 프로덕션 WSGI 서버

----------------------------------------------------------------
## 시놀로지 NAS에서 웹 호스팅으로 서비스하는 방법

1. 사전 준비:
```
1. Web Station 패키지 설치
   - 패키지 센터 접속
   - Web Station 검색 및 설치
   - PHP, Python과 같은 필요한 확장 기능도 함께 설치

2. Reverse Proxy 설정을 위한 nginx 설치 (이미 Web Station에 포함)
```

2. 웹 사이트 설정:
```
1. Web Station 실행
2. 가상 호스트 생성:
   - "Virtual Host" 탭에서 "Create" 클릭
   - 포트: 80 (HTTP) 또는 443 (HTTPS)
   - 도메인 이름: 사용할 도메인 입력
   - 문서 루트: /volume1/web/lotto-prediction
   - 백엔드 서버: Python 3.9
```

3. 프로젝트 구조 설정:
```
/volume1/web/lotto-prediction/
├── app.py
├── wsgi.py
├── config.py
├── requirements.txt
├── static/
├── templates/
└── data/
    ├── logs/
    ├── predictions/
    ├── models/
    └── lotto.db
```

4. wsgi.py 파일 생성:
```python
from app import app

if __name__ == "__main__":
    app.run()
```

5. 리버스 프록시 설정:
```nginx
location / {
    proxy_pass http://localhost:5000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

6. 환경 설정:
```bash
# 가상환경 생성
cd /volume1/web/lotto-prediction
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

7. 서비스 실행을 위한 systemd 서비스 파일 생성:
```ini
[Unit]
Description=Lotto Prediction Web Service
After=network.target

[Service]
User=http
Group=http
WorkingDirectory=/volume1/web/lotto-prediction
Environment="PATH=/volume1/web/lotto-prediction/venv/bin"
ExecStart=/volume1/web/lotto-prediction/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:5000 wsgi:app

[Install]
WantedBy=multi-user.target
```

8. SSL 인증서 설정 (권장):
```
1. Let's Encrypt 인증서 발급:
   - Control Panel → Security → Certificate
   - "Add" 클릭하여 Let's Encrypt 인증서 발급
   
2. Web Station에서 SSL 설정:
   - Virtual Host 설정에서 HTTPS 포트(443) 추가
   - 발급받은 인증서 선택
```

9. 보안 설정:
```
1. 방화벽 설정:
   - Control Panel → Security → Firewall
   - 80, 443 포트만 허용

2. 권한 설정:
   chmod -R 755 /volume1/web/lotto-prediction
   chown -R http:http /volume1/web/lotto-prediction
```

10. 서비스 시작 및 모니터링:
```bash
# 서비스 시작
systemctl start lotto-prediction

# 로그 모니터링
tail -f /volume1/web/lotto-prediction/data/logs/app.log
```

문제 해결 팁:
1. 로그 확인:
```bash
# Web Station 로그
tail -f /var/log/nginx/error.log

# 애플리케이션 로그
tail -f /volume1/web/lotto-prediction/data/logs/*
```

2. 권한 문제 발생 시:
```bash
# 폴더 권한 재설정
chmod -R 755 /volume1/web/lotto-prediction
chown -R http:http /volume1/web/lotto-prediction/data
```

3. 서비스 재시작:
```bash
systemctl restart lotto-prediction
systemctl restart nginx
```

접속 테스트:
1. 내부 네트워크: http://시놀로지IP:5000
2. 외부 접속: https://your-domain.com

추가 보안 고려사항:
1. DDoS 보호 설정
2. 요청 제한 설정
3. IP 차단 정책 설정
4. 정기적인 백업 설정


