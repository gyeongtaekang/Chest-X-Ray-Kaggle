import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# 데이터 경로 설정
dataset_path = r"C:\Users\AERO\Downloads\archive (1)\chest_xray"

# 데이터 증강 및 전처리 설정
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.6, contrast=0.6),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 학습 및 검증 데이터 로더 설정
train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=train_transform)
val_dataset = ImageFolder(root=os.path.join(dataset_path, 'val'), transform=val_test_transform)
test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=val_test_transform)

# 클래스 비율에 따른 가중치 계산
class_counts = np.bincount(train_dataset.targets)
weights = 1. / class_counts
sample_weights = weights[train_dataset.targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 모델 정의 (ResNet50 사용)
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 1)
        )

    def forward(self, x):
        return self.model(x)


model = PneumoniaModel()


# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1.5):  # 감마값을 낮춰서 조정
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
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)

# 조기 종료 설정
early_stopping_patience = 3
best_val_loss = float('inf')
patience_counter = 0


# 모델 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    global best_val_loss, patience_counter

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cpu'), labels.float().to('cpu')
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # 학습률 스케줄 업데이트
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]}')

        # 검증 정확도 계산
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to('cpu'), labels.float().to('cpu')
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
                val_loss += loss.item() * inputs.size(0)

                # 임계값을 0.8로 설정
                preds = (torch.sigmoid(outputs) >= 0.8).float()
                correct += (preds.view(-1) == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

        # 조기 종료 조건
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "pneumonia_detection_model_optimized.pth")
            print("모델이 저장되었습니다.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("조기 종료 발동")
                break


# 학습 시작
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

# 테스트 평가 (Test-Time Augmentation 포함)
model.load_state_dict(torch.load("pneumonia_detection_model_optimized.pth", map_location=torch.device('cpu')))
model.eval()

all_labels = []
all_preds = []

# TTA 적용
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        # TTA 적용: 이미지 회전 후 평균 결과
        tta_outputs = []
        for angle in [-10, 0, 10]:  # 회전 각도 적용
            rotated_inputs = transforms.functional.rotate(inputs, angle)
            outputs = model(rotated_inputs)
            tta_outputs.append(torch.sigmoid(outputs))
        # 평균 예측 값 계산
        avg_output = torch.stack(tta_outputs).mean(dim=0)
        preds = (avg_output >= 0.8).float()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy().squeeze())

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
