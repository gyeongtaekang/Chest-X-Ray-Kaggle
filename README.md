# Pneumonia Detection Model

이 프로젝트는 흉부 X-ray 영상을 분석하여 폐렴 여부를 예측하는 모델을 학습하기 위해 ResNet18을 사용합니다. 두 가지 접근 방식을 통해 모델 성능을 비교하고, 더 나은 성능을 제공하는 방법을 찾습니다.

## 1. 첫 번째 코드

첫 번째 코드에서는 기본적인 **데이터 증강**만을 적용하며, 클래스 불균형 문제에 대한 특별한 조정을 하지 않았습니다.

- **데이터 증강**: `RandomRotation`, `RandomHorizontalFlip`, `RandomResizedCrop`을 사용하여 이미지 회전, 좌우 반전, 크기 변형 등의 간단한 증강 기법을 적용했습니다.
- **클래스 불균형 문제 해결 없음**: 손실 함수 `BCEWithLogitsLoss`에는 클래스 가중치나 불균형 보정이 없습니다.
- **데이터 로더**: `train_loader`, `val_loader`는 데이터셋에서 무작위로 샘플링하며, 클래스 비율을 고려하지 않습니다.

이 코드는 기본적인 데이터 증강만을 사용하여 학습을 진행하며, 클래스 불균형에 대해 특별한 처리를 하지 않습니다. 일반적인 데이터 증강만으로도 일정 수준의 성능을 기대할 수 있습니다.

## 2. 두 번째 코드

두 번째 코드에서는 **강화된 데이터 증강**과 **클래스 불균형 문제 해결**을 추가하여 성능을 개선합니다.

- **강화된 데이터 증강**: `RandomResizedCrop`에 `scale=(0.8, 1.0)` 매개변수를 추가하여 데이터의 다양성을 더 높였습니다. 이렇게 하면 이미지의 일부를 더 크게 잘라내는 등, 더욱 다양한 증강이 이루어집니다.
- **클래스 불균형 해결**:
  - **WeightedRandomSampler** 사용: 클래스 비율에 따라 가중치를 설정하고, 클래스가 불균형하더라도 균형 있게 샘플링하도록 설정했습니다. 이를 통해 각 클래스가 학습에서 충분히 반영되도록 했습니다.
  - **가중 손실 함수**: `BCEWithLogitsLoss`의 `pos_weight` 매개변수를 설정하여 손실 함수가 클래스 비율에 따라 가중치를 부여하도록 했습니다. 이렇게 하면 데이터의 불균형을 어느 정도 보정할 수 있습니다.
- **데이터 로더**: `WeightedRandomSampler`로 클래스 불균형을 보정한 샘플링 방식을 사용하여 학습합니다.

이 코드는 강화된 데이터 증강을 적용하고, 클래스 불균형 문제를 해결할 수 있는 설정을 추가하여 보다 정밀하게 학습할 수 있도록 구성되었습니다.

## 성능 비교

이 두 가지 접근 방식을 통해 모델의 성능을 비교하고, 데이터 증강 및 클래스 불균형 문제 해결이 실제 모델 성능에 어떤 영향을 미치는지 분석할 수 있습니다. 이를 통해 최종적으로 더 좋은 성능을 보이는 방법을 선택하여 모델 성능을 최적화할 수 있습니다.

---------------------------------------------------------------------------------------------------------------------------------------

## 데이터 전처리 및 증강

모델 학습을 위해 이미지를 224x224 크기로 조정하고 다양한 증강 기법을 적용했습니다. 이를 통해 모델이 다양한 패턴을 학습하고, 데이터가 부족한 의료 데이터셋에서 과적합을 방지할 수 있도록 합니다.

### 이미지 전처리 및 증강 설정

- **이미지 크기 조정**: 모든 이미지를 `Resize((224, 224))`로 크기를 조정하여 모델 입력 형식에 맞추었습니다.
- **데이터 증강**: 학습 데이터에 `RandomRotation`, `RandomHorizontalFlip`, `RandomResizedCrop` 등을 적용하여 이미지 회전, 좌우 반전, 무작위 크기 조정을 수행했습니다.
- **텐서 변환 및 정규화**: 이미지를 텐서 형식으로 변환(`ToTensor()`)하고, 각 채널을 [0, 1] 범위로 정규화(`Normalize()`)하여 안정적인 학습을 지원했습니다.

### 코드 예시

```python
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 이미지 전처리 및 증강 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # 이미지 크기 조정
    transforms.RandomRotation(20),       # 20도 회전
    transforms.RandomHorizontalFlip(),   # 좌우 반전
    transforms.RandomResizedCrop(224),   # 무작위 크기로 자르고 크기 조정
    transforms.ToTensor(),               # 텐서 형식으로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 정규화
])

# 학습 데이터 로더
train_dataset = ImageFolder(root='chest_xray/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 검증 데이터 (증강 없이)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = ImageFolder(root='chest_xray/val', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
