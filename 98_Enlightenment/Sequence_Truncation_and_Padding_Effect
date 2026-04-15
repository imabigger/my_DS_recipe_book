# Title: Sequence_Truncation_and_Padding_Effect
## Description: 시계열 데이터에서 지나치게 긴 시퀀스를 잘라내는(Truncation) 이유와, 과도한 패딩(Zero-Padding)이 딥러닝 모델 성능에 미치는 악영향을 설명하는 이론적 배경.
## Key Concept 1: 과도한 패딩이 부르는 '신호 희석 (Signal Dilution)'
- **이론적 배경**: 데이터의 길이를 맞추기 위해 '0'을 채워 넣는 것을 Zero-Padding이라고 합니다. 하지만 진짜 데이터(예: 길이 20)보다 가짜 데이터(예: 길이 250)가 훨씬 많아지면, 모델은 "0이 나오는 것 자체"를 어떤 중요한 패턴으로 착각하게 됩니다.
- **관련 논문**: *Dwarampudi & Reddy (2019), "Effects of padding on LSTMs and CNNs"*
- **논문 요약**: 입력 데이터에 지나치게 많은 패딩이 들어가면, LSTM이나 CNN 모델이 원래 데이터가 가진 고유한 특징(Feature)을 잃어버리고 정확도가 크게 하락한다는 것을 실험적으로 증명했습니다.

## Key Concept 2: 트랜스포머의 '주의력 분산 (Attention Distraction)'
- **이론적 배경**: 트랜스포머(Transformer) 모델은 데이터 안의 모든 요소가 서로 얼마나 연관되어 있는지 계산하는 'Self-Attention' 메커니즘을 사용합니다. 데이터가 길어질수록 연산량은 $O(N^2)$로 폭증하며, 정작 중요한 최근 정보에 집중해야 할 '주의력(Attention Weight)'이 쓸데없는 과거 정보들로 분산되어 버립니다.
- **관련 논문**: *Vaswani et al. (2017), "Attention Is All You Need"*
- **논문 요약**: 트랜스포머의 근본 논문입니다. 이 논문과 후속 연구들에서는 긴 시퀀스를 다룰 때 연산 폭증과 주의력 분산 문제가 발생하므로, 시퀀스 길이를 적절히 제한하거나(Windowing) 중요한 부분만 보도록 제약을 걸어야 한다고 설명합니다.

## Key Concept 3: 제한된 과거만 보는 '마르코프 가정 (Markov Assumption)'
- **이론적 배경**: "미래의 상태는 오직 현재(또는 아주 가까운 과거)의 상태에만 영향을 받는다"는 확률론적 가정입니다. 축구 패스 예측에서도 "3분 전의 패스"보다 "1초 전의 패스와 현재 위치"가 다음 패스를 결정하는 데 압도적으로 중요합니다.
- **관련 기법**: **Truncated BPTT (Backpropagation Through Time)**
- **설명**: 너무 긴 과거까지 오차를 역전파(Backprop)하면 오히려 학습이 망가지는 현상(Vanishing Gradient)을 막기 위해, 시퀀스를 특정 길이(`MAX_WINDOW`)로 싹둑 잘라서 학습시키는 딥러닝의 표준 학습 기법입니다.
