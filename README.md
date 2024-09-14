# face-emotion-recognition
제공된 이미지를 분석하여 표정을 인식하고 그 결과를 출력합니다.

## 테스트 서버 환경

- OS: Windows
- Python 버전: 3.10.12
- TensorFlow 버전: 2.10.0
- 기타 라이브러리 버전: `requirements.txt` 파일 참고

## 사용 방법

### 1. 가상환경 생성 및 라이브러리 설치
```bash
python -m venv venv
```
아래 라이브러리는 필수 설치
```bash
pip install tensorflow
pip install hsemotion
pip install hsemotion-onnx
pip install timm==0.4.5
```

### 2. 가상환경 활성화

운영체제에 따라 아래 명령어로 가상환경을 활성화합니다:

- **Windows:**

  ```bash
  source venv/Scripts/activate
  ```

- **Linux/MacOS:**

  ```bash
  source venv/bin/activate
  ```

### 3. 얼굴 표정 인식 실행

터미널에서 아래 명령어를 사용하여 이미지를 입력하고 얼굴 표정을 인식합니다.

```bash
python display_emotion.py <img_path>
```

`<img_path>`에는 분석할 이미지 파일의 경로를 입력합니다.

### 4. 결과 확인

결과는 `results/res.txt` 파일에서 확인할 수 있습니다.

