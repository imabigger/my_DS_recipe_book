# Title: Noise_Injection_Regularization
## Description: 학습 데이터에 인위적인 미세 노이즈(Jittering)를 주입하여 모델의 단순 암기(과적합)를 방지하고, 수학적으로 L2 정규화(Regularization)와 동일한 일반화 효과를 얻는 원리.
## Key Concept: 
 - 딥러닝 모델은 용량(Capacity)이 커서 데이터의 패턴이 아닌 '정답 좌표 숫자 자체'를 외워버리는 경향이 있음 (Overfitting).
 - 매 학습(Epoch)마다 데이터에 미세한 무작위 노이즈를 더해주면, 모델은 특정 숫자에 집착하는 것을 포기하고 데이터의 **'전반적인 흐름(Robust Feature)'**을 학습하게 됨.
## Theoretical Background:
 - **Christopher M. Bishop (1995)**: "Training with Noise is Equivalent to Tikhonov Regularization"
 - 입력 데이터에 가우시안 노이즈를 추가하는 것은 모델의 가중치(Weight)가 비정상적으로 커지는 것을 억제하는 L2 정규화(Ridge) 수식과 완전히 동일한 효과를 냄.
## Check Point: 
 - 노이즈의 크기(Amplitude) 설정이 핵심. 본질적인 데이터의 의미(전술적 흐름)를 훼손하지 않는 선에서, 기계의 암기력만 방해할 수 있는 '아주 미세한 흔들림(예: 축구장에서 10cm)'을 찾아야 함.
