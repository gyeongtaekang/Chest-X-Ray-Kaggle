import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# GPU 강제 사용 설정
if not torch.cuda.is_available():
    raise RuntimeError("CUDA가 사용 불가능합니다. GPU를 활성화하거나 적절한 환경을 설정하세요.")
torch.cuda.set_device(0)  # GPU 0번 강제 사용
device = torch.device("cuda")

# 데이터 경로 설정
dataset_path = r"C:\Users\AERO\Downloads\archive (1)\chest_xray"

# 데이터 증강 및 전처리 설정
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),  # 랜덤하게 이미지를 회전하여 데이터 다양성 증가
    transforms.ColorJitter(brightness=0.6, contrast=0.6),  # 밝기와 대비를 랜덤하게 조정
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # 이미지를 랜덤하게 이동 및 회전
    transforms.RandomHorizontalFlip(),  # 랜덤하게 이미지를 좌우 반전
    transforms.ToTensor(),  # 이미지를 텐서 형태로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 이미지 정규화
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 이미지를 텐서 형태로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 이미지 정규화
])

# 학습 및 검증 데이터 로더 설정
train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=train_transform)
val_dataset = ImageFolder(root=os.path.join(dataset_path, 'val'), transform=val_test_transform)
test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=val_test_transform)

# 클래스 비율에 따른 가중치 계산
class_counts = np.bincount(train_dataset.targets)  # 각 클래스의 샘플 수 계산
weights = 1. / class_counts  # 각 클래스에 대한 가중치 설정 (샘플이 적을수록 가중치가 높음)
sample_weights = weights[train_dataset.targets]  # 각 샘플에 대한 가중치 설정
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)  # 가중치를 기반으로 샘플링

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# 모델 정의 (ResNet50 사용)
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 사전 학습된 ResNet50 모델 사용
        # Dropout을 포함한 추가 레이어로 과적합 방지 및 출력 레이어 변경
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # 과적합 방지를 위한 Dropout 추가
            nn.Linear(self.model.fc.in_features, 1)  # 출력 뉴런 수를 1로 설정 (이진 분류)
        )

    def forward(self, x):
        return self.model(x)

model = PneumoniaModel().to(device)

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 양성 샘플에 대한 가중치
        self.gamma = gamma  # 난이도 조절 파라미터

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # 이진 크로스 엔트로피 손실 계산
        pt = torch.exp(-BCE_loss)  # 예측 확률 계산
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # Focal Loss 계산
        return focal_loss.mean()

# Focal Loss와 Optimizer 설정
criterion = FocalLoss(alpha=2, gamma=1.5)  # Focal Loss로 클래스 불균형 문제 해결
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # AdamW로 변경하여 가중치 감쇠 추가 (과적합 방지)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)  # 학습률 스케줄러 설정 (코사인 함수 기반)

# 조기 종료 설정
early_stopping_patience = 3  # 조기 종료를 위한 patience 설정
best_val_loss = float('inf')  # 가장 낮은 검증 손실값 초기화
patience_counter = 0  # patience 카운터 초기화

# 모델 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    global best_val_loss, patience_counter

    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0

        for inputs, labels in train_loader:
            # GPU로 데이터 전송
            inputs, labels = inputs.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
            optimizer.zero_grad()  # 그래디언트 초기화

            outputs = model(inputs)  # 모델 예측값 계산
            loss = criterion(outputs.view(-1), labels)  # 손실 계산
            loss.backward()  # 역전파 단계
            optimizer.step()  # 가중치 업데이트

            running_loss += loss.item() * inputs.size(0)  # 총 손실값 누적

        scheduler.step()  # 학습률 업데이트
        epoch_loss = running_loss / len(train_loader.dataset)  # 에포크 당 평균 손실 계산
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]}')

        # 검증 단계
        model.eval()  # 모델을 평가 모드로 설정
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
                outputs = model(inputs)  # 모델 예측값 계산
                loss = criterion(outputs.view(-1), labels)  # 손실 계산
                val_loss += loss.item() * inputs.size(0)  # 검증 손실 누적

                # 임계값 적용하여 예측값 이진화
                preds = (torch.sigmoid(outputs) >= 0.8).float()
                correct += (preds.view(-1) == labels).sum().item()  # 정확도 계산을 위한 정답 개수 누적
                total += labels.size(0)  # 총 샘플 개수 누적

        val_loss /= len(val_loader.dataset)  # 평균 검증 손실 계산
        accuracy = correct / total  # 검증 정확도 계산
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

        # 조기 종료 조건 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # 최고의 검증 손실 업데이트
            patience_counter = 0  # patience 카운터 초기화
            torch.save(model.state_dict(), "pneumonia_detection_model_optimized.pth")  # 최적의 모델 저장
            print("모델 저장됨")
        else:
            patience_counter += 1  # patience 카운터 증가
            if patience_counter >= early_stopping_patience:
                print("조기 종료 발동")
                break

if __name__ == '__main__':
    # 학습 시작
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # 테스트 평가 (Test-Time Augmentation 포함)
    model.load_state_dict(torch.load("pneumonia_detection_model_optimized.pth"))  # 최적의 모델 로드
    model.eval()  # 모델을 평가 모드로 설정

    all_labels = []
    all_preds = []

    # TTA 적용
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # TTA 적용: 이미지 회전 후 평균 결과
            tta_outputs = []
            for angle in [-10, 0, 10]:  # -10도, 0도, 10도로 이미지를 회전하여 TTA 적용
                rotated_inputs = transforms.functional.rotate(inputs, angle)
                outputs = model(rotated_inputs)  # 모델 예측값 계산
                tta_outputs.append(torch.sigmoid(outputs))
            # 평균 예측 값 계산
            avg_output = torch.stack(tta_outputs).mean(dim=0)  # 각 회전에 대한 예측 결과 평균
            preds = (avg_output >= 0.8).float()  # 임계값을 적용하여 최종 예측값 이진화

            all_labels.extend(labels.cpu().numpy())  # 실제 라벨 저장
            all_preds.extend(preds.cpu().numpy().squeeze())  # 예측값 저장

    # 평가 지표 계산
    accuracy = accuracy_score(all_labels, all_preds)  # 정확도 계산
    precision = precision_score(all_labels, all_preds)  # 정밀도 계산
    recall = recall_score(all_labels, all_preds)  # 재현율 계산
    f1 = f1_score(all_labels, all_preds)  # F1 스코어 계산
    conf_matrix = confusion_matrix(all_labels, all_preds)  # 혼동 행렬 계산

    # 결과 출력
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)


# PS C:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main>  c:; cd 'c:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main'; & 'c:\Python312\python.exe' 'c:\Users\AERO\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '51605' '--' 'c:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main\ResNet50.py'
# Epoch 1/10, Loss: 0.1822, LR: 0.0009757729755661011
# Validation Loss: 0.3540, Accuracy: 0.8125
# 모델 저장됨
# Epoch 2/10, Loss: 0.1188, LR: 0.000905463412215599
# Validation Loss: 0.2250, Accuracy: 0.5625
# 모델 저장됨
# Epoch 3/10, Loss: 0.0895, LR: 0.0007959536998847742
# Validation Loss: 0.4579, Accuracy: 0.9375
# Epoch 4/10, Loss: 0.0745, LR: 0.000657963412215599
# Validation Loss: 0.3550, Accuracy: 0.8125
# Epoch 5/10, Loss: 0.0691, LR: 0.000505
# Validation Loss: 0.0405, Accuracy: 0.8125
# 모델 저장됨
# Epoch 6/10, Loss: 0.0576, LR: 0.0003520365877844011
# Validation Loss: 0.0314, Accuracy: 0.9375
# 모델 저장됨
# Epoch 7/10, Loss: 0.0471, LR: 0.00021404630011522585
# Validation Loss: 0.0977, Accuracy: 0.6875
# Epoch 8/10, Loss: 0.0366, LR: 0.00010453658778440107
# Validation Loss: 0.0329, Accuracy: 0.8125
# Epoch 9/10, Loss: 0.0334, LR: 3.4227024433899005e-05
# Validation Loss: 0.0272, Accuracy: 1.0000
# 모델 저장됨
# Epoch 10/10, Loss: 0.0277, LR: 1e-05
# Validation Loss: 0.0188, Accuracy: 1.0000
# 모델 저장됨
# c:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main\ResNet50.py:154: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature. 
#   model.load_state_dict(torch.load("pneumonia_detection_model_optimized.pth"))  # 최적의 모델 로드
# Test Accuracy: 0.9311
# Test Precision: 0.9392
# Test Recall: 0.9513
# Test F1 Score: 0.9452
# Confusion Matrix:
# [[210  24]
#  [ 19 371]]
# PS C:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main>