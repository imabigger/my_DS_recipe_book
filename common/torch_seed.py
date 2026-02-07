import random
import os
import numpy as np
import torch

# 재현성을 위한 코드
def seed_everything(seed=42):
    random.seed(seed) # 1. 파이썬 내장 랜덤 모듈 고정
  
    # 2. 해시 함수 고정
    # 딕셔너리나 집합(Set)은 순서가 없지만, 내부적으로 해시값을 씁니다.
    # 이걸 고정 안 하면 같은 딕셔너리를 만들어도 순서가 달라져서 학습 결과가 미세하게 바뀔 수 있습니다.
    os.environ['PYTHONHASHSEED'] = str(seed)
  
    np.random.seed(seed) # 3. Numpy의 난수를 고정합니다 (데이터 전처리, 셔플 등에서 필수)
  
    torch.manual_seed(seed)  # 4. ↘
    torch.cuda.manual_seed(seed) # cpu, gpu 모두에서 모델의 시작 가중치를 고정
  
    # 5. 연산 속도 vs 재현성 트레이드오프 설정
    # True로 하면: 항상 같은 알고리즘을 써서 결과가 완벽히 동일 (재현성 최우선)
    torch.backends.cudnn.deterministic = True

    # True로 하면: 하드웨어 상황에 맞춰 제일 빠른 알고리즘을 자동 선택 (속도 최우선)
    # 학습용이니 False
    torch.backends.cudnn.benchmark = False
    print(f'Seed set to {seed}')
