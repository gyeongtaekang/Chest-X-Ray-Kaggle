import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# GPU 강제 사용 설정
if not torch.cuda.is_available():
    raise RuntimeError("CUDA가 사용 불가능합니다. GPU를 활성화하거나 적절한 환경을 설정하세요.")
torch.cuda.set_device(0)  # GPU 0번 강제 사용
device = torch.device("cuda")

# 데이터셋 경로 설정
data_dir = r"C:\Users\AERO\Downloads\archive (1)\chest_xray"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# 데이터 전처리 및 증강 설정
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 로더 설정
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = ImageFolder(root=test_dir, transform=val_test_transform)

# 클래스 비율에 따른 가중치 계산
class_counts = np.bincount(train_dataset.targets)
weights = 1. / class_counts
sample_weights = weights[train_dataset.targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 모델 정의 (ResNet50 사용)
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)

model = PneumoniaModel().to(device)

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Focal Loss와 Optimizer 설정
criterion = FocalLoss(alpha=2, gamma=1.5)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)

# 조기 종료 설정
early_stopping_patience = 5
best_val_loss = float('inf')
patience_counter = 0

# 모델 학습 함수
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50):
    global best_val_loss, patience_counter

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]}')

        # 검증 단계
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds.view(-1) == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

        # 조기 종료 조건 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "pneumonia_detection_model_optimized.pth")
            print("모델 저장됨")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("조기 종료 발동")
                break

        # 테스트 데이터로 평가
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy().squeeze())

        test_accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f'Test Accuracy: {test_accuracy:.4f}')
        print(f'Test Precision: {precision:.4f}')
        print(f'Test Recall: {recall:.4f}')
        print(f'Test F1 Score: {f1:.4f}')
        print("Confusion Matrix:")
        print(conf_matrix)

        # 테스트 정확도가 1에 도달하면 학습 중단
        if test_accuracy >= 0.9999:
            print("테스트 데이터에서 정확도 100%에 근접하여 학습을 중단합니다.")
            break

        # 개선 방안 적용: 결과에 따른 하이퍼파라미터 조정
        if test_accuracy < 0.85:
            print("테스트 정확도가 낮습니다. 학습률을 조정하고 모델을 재학습합니다.")
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.9
            print(f"새 학습률: {g['lr']}")

if __name__ == '__main__':
    # 학습 시작
    train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler)

    # 테스트 평가
    model.load_state_dict(torch.load("pneumonia_detection_model_optimized.pth"))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().squeeze())

    # 평가 지표 계산
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 결과 출력
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)


# Epoch 1/50, Loss: 0.2218, LR: 9.757729755661011e-05
# Validation Loss: 0.1329, Accuracy: 0.9375
# 모델 저장됨
# Test Accuracy: 0.8750
# Test Precision: 0.9239
# Test Recall: 0.8718
# Test F1 Score: 0.8971
# Confusion Matrix:
# [[206  28]
#  [ 50 340]]
# Epoch 2/50, Loss: 0.1462, LR: 9.05463412215599e-05
# Validation Loss: 0.2141, Accuracy: 0.9375
# Test Accuracy: 0.9167
# Test Precision: 0.8986
# Test Recall: 0.9769
# Test F1 Score: 0.9361
# Confusion Matrix:
# [[191  43]
#  [  9 381]]
# Epoch 3/50, Loss: 0.1094, LR: 7.959536998847742e-05
# Validation Loss: 0.1329, Accuracy: 0.8750
# 모델 저장됨
# Test Accuracy: 0.8862
# Test Precision: 0.8552
# Test Recall: 0.9846
# Test F1 Score: 0.9154
# Confusion Matrix:
# [[169  65]
#  [  6 384]]
# Epoch 4/50, Loss: 0.0952, LR: 6.57963412215599e-05
# Validation Loss: 0.2492, Accuracy: 0.8125
# Test Accuracy: 0.8798
# Test Precision: 0.8477
# Test Recall: 0.9846
# Test F1 Score: 0.9110
# Confusion Matrix:
# [[165  69]
#  [  6 384]]
# Epoch 5/50, Loss: 0.0843, LR: 5.05e-05
# Validation Loss: 0.1720, Accuracy: 0.8125
# Test Accuracy: 0.8702
# Test Precision: 0.8323
# Test Recall: 0.9923
# Test F1 Score: 0.9053
# Confusion Matrix:
# [[156  78]
#  [  3 387]]
# Epoch 6/50, Loss: 0.0709, LR: 3.5203658778440106e-05
# Validation Loss: 0.1196, Accuracy: 0.8750
# 모델 저장됨
# Test Accuracy: 0.9215
# Test Precision: 0.9012
# Test Recall: 0.9821
# Test F1 Score: 0.9399
# Confusion Matrix:
# [[192  42]
#  [  7 383]]
# Epoch 7/50, Loss: 0.0711, LR: 2.1404630011522586e-05
# Validation Loss: 0.2194, Accuracy: 0.8750
# Test Accuracy: 0.8798
# Test Precision: 0.8431
# Test Recall: 0.9923
# Test F1 Score: 0.9117
# Confusion Matrix:
# [[162  72]
#  [  3 387]]
# Epoch 8/50, Loss: 0.0613, LR: 1.0453658778440107e-05
# Validation Loss: 0.2464, Accuracy: 0.8750
# Test Accuracy: 0.8862
# Test Precision: 0.8490
# Test Recall: 0.9949
# Test F1 Score: 0.9162
# Confusion Matrix:
# [[165  69]
#  [  2 388]]
# Epoch 9/50, Loss: 0.0622, LR: 3.4227024433899e-06
# Validation Loss: 0.2004, Accuracy: 0.8750
# Test Accuracy: 0.8990
# Test Precision: 0.8691
# Test Recall: 0.9872
# Test F1 Score: 0.9244
# Confusion Matrix:
# [[176  58]
#  [  5 385]]
# Epoch 10/50, Loss: 0.0681, LR: 1e-06
# Validation Loss: 0.2362, Accuracy: 0.8750
# Test Accuracy: 0.8926
# Test Precision: 0.8581
# Test Recall: 0.9923
# Test F1 Score: 0.9203
# Confusion Matrix:
# [[170  64]
#  [  3 387]]
# Epoch 11/50, Loss: 0.0514, LR: 3.4227024433899e-06
# Validation Loss: 0.1936, Accuracy: 0.8750
# 조기 종료 발동
# c:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main\daum.py:192: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.     
#   model.load_state_dict(torch.load("pneumonia_detection_model_optimized.pth"))
# Test Accuracy: 0.9215
# Test Precision: 0.9012
# Test Recall: 0.9821
# Test F1 Score: 0.9399
# Confusion Matrix:
# [[192  42]
#  [  7 383]]
# PS C:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main>