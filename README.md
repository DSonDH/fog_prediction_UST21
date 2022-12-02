# Fog Generation Prediction (work of UST21)
fog generation prediction for 1/3/6 hours using machine learning and deep learning.
This project is for the [service](http://www.khoa.go.kr/oceanmap/pois/popup_seafog.do?lang=ko).
The code is going to be updated highly frequently.

# aws related setting
1. conda install awscli -y or pip install awscli
2. aws configure
AWS Access Key ID [None]: seafog
AWS Secret Access Key [None]: "defaultpassword" hint: \w{3}\d{7}
Default region name [None]: ENTER
Default output format [None]: ENTER

3. aws configure set default.s3.signature_version s3v4
4. aws --endpoint-url http://oldgpu:9000 s3 cp s3://seafog data --recursive


# environment setting
환경 설치
```bash
make install
```

# training
activate 환경
```bash
conda activate venv/
```

data/ 밑에 필수 폴더 생성
```bash
python src/step_1_pre_train_ml.py
```

CUDA_VISIBLE_DEVICES default : 0번
```
CUDA_VISIBLE_DEVICES='mygpu' python src/step_2_train_hpo_ml.py
```

서비스할때 고려할 점
```bash
python src/step_3_train_refit_ml.py
```

1. 최적 하이퍼파라미터로  train 셋을 train = train + val 으로 업데이트한 후 
한번 더 학습한 후 모델 저장
2. 모델 불러와서 실제 서비스 시 X_test, y_test 예측
3. 성능은 config파일에 catalogue.test_result에 있음

# License

[Apache License v2.0](License)