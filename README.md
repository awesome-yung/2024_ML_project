# 2024_ML_project

whisper를 fine tuning하여 voice recognition을 실험했다.

훈련된 model은 huggingface에 업로드 되어있다
![image](https://github.com/awesome-yung/2024_ML_project/assets/148609228/8cdbee89-b2d0-40ac-a722-e0f14ffa8181)

위 실험에서 사용된 데이터셋은 Bingsu/zeroth-korean 22720개와
강의에서 추출하여 직접 제작한 custom dataset 500개이다.

"손 드세요", "손 내리세요" 라는 명령어를 입력받았을 때
줌의 "손들기" 버튼을 활성화 한다.
![시연_1_gif](https://github.com/awesome-yung/2024_ML_project/assets/148609228/fe16d4f7-ade9-4344-aa13-a444d4f8f72e)

강의 영상에서는 voice recognition이 잘 이루어지지 않지만,
뉴스 데이터에서는 voice recognition을 충분히 잘 해낸다.
![시연_2gif](https://github.com/awesome-yung/2024_ML_project/assets/148609228/6c74dccf-7ba9-488c-b383-97e880f320e2)

🚀 사용 방법 🚀
1. voicemeeter 설치
2. requirements 설치 및 cuda 설치
3. HandRaiser/main.py 실행

![poster](https://github.com/awesome-yung/2024_ML_project/assets/148609228/2281e29e-8610-4b13-a8c4-a2a19a88396e)