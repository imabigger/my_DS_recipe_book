# Label Smoothing
> Label Smoothing은 모델이 정답을 지나치게 확신하지 않도록 정답 확률을 살짝 깎아주는 규제 기법이다.

## Key Concept
딥러닝 모델이 개 사진을 보고 "이건 100% 개야!"라고 너무 강하게 믿으면(Overconfidence), 조금만 다르게 생긴 개가 나와도 당황하거나 과적합되기 쉽다. 이때 정답인 1.0의 확률을 0.95 정도로 낮추고, 남은 0.05를 다른 오답들에 골고루 나눠줌으로써 모델이 유연하게 **일반화**할 수 있도록 돕는 일종의 '겸손함' 주입 장치다.

## Theoretical Background
- 출처: Szegedy et al. / Rethinking the Inception Architecture for Computer Vision / 2016
- 근거: 정답 클래스에 대해 무한대의 로짓(logit) 값을 추구하는 Hard Target 방식이 모델의 적응력을 떨어뜨림을 지적하고, Soft Target(Label Smoothing)을 통해 Cross Entropy 손실 함수가 가중치를 무한히 키우는 것을 방지함

- 출처2: Müller et al. / When Does Label Smoothing Help? / 2019
- 근거2: 시각화 실험을 통해 Label Smoothing이 동일 클래스 내의 샘플들을 더 촘촘하게 모으고 클래스 간 경계를 명확히 하는 효과가 있음을 증명함

## Caution
지나치게 높은 Smoothing 값은 오히려 모델의 학습을 방해할 수 있다. 보통 **0.05에서 0.1** 사이의 값을 사용하며, 지식 증류(Knowledge Distillation)와 병행할 때는 교사 모델의 확률 분포 자체가 부드러우므로 중복 적용에 주의해야 한다.

## 내 한 줄 정의
강한 확신은 오히려 모델의 유연함을 막는다.

## 이 외
jittering 도 그렇고 label smoothing도 그렇고, 뭔가 정답을 살짝 두루뭉실하게 해주면 오히려 모델의 예측성능이 올라간다.
딥러닝은 모델이 한없이 복잡해질수 있는데, 그러면 데이터의 의미없는 노이즈까지 전부 반영해서 어떠한 특정한 패턴을 암기하는데,
세상은 똑같은 패턴이어도 같은 정답이 일어나지 않을 수 있는 환경이기 때문에 overfitting이 일어나는 듯 하다.

물론 overfitting의 발생경위가 이거 하나 뿐만은 아니겠지만 말이다.
