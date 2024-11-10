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

# 데이터 증강 설정 (정밀도 개선을 위해 더 다양한 증강을 적용)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),  # 회전 범위를 줄여 정밀도를 높임
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 색상 변화 추가
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 검증 및 테스트 데이터 전처리 설정
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 로더 설정
train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=train_transform)
val_dataset = ImageFolder(root=os.path.join(dataset_path, 'val'), transform=val_test_transform)
test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=val_test_transform)

# 클래스 가중치 설정 (정상 클래스에 높은 가중치를 주어 정밀도 개선)
class_counts = np.bincount(train_dataset.targets)
weights = torch.tensor([class_counts[1] / class_counts[0], 1.0], dtype=torch.float)  # 비율에 따라 가중치 설정
criterion = nn.BCEWithLogitsLoss(pos_weight=weights[0])

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 정의
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
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

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

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

                preds = torch.sigmoid(outputs) >= 0.6  # 임계값을 0.6으로 조정
                correct += (preds.view(-1) == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

# 학습 시작
train_model(model, train_loader, val_loader, criterion, optimizer)

# 모델 저장
torch.save(model.state_dict(), "pneumonia_detection_model_precision.pth")
print("모델이 정밀도 개선을 위해 저장되었습니다.")
