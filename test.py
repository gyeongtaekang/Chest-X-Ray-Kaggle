import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

# 데이터셋 경로 설정
dataset_path = r"C:\Users\AERO\Downloads\archive (1)\chest_xray"

# 이미지 전처리 및 증강 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.RandomRotation(20),  # 20도 회전
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.RandomResizedCrop(224),  # 무작위 크기로 자르고 크기 조정
    transforms.ToTensor(),  # 텐서 형식으로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 정규화
])

# 학습 데이터 로더
train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 검증 데이터 (증강 없이)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = ImageFolder(root=os.path.join(dataset_path, 'val'), transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델 정의 (ResNet18 사용 예시)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)  # 이진 분류를 위한 출력 레이어 수정

# 손실 함수와 최적화기 정의
criterion = nn.BCEWithLogitsLoss()  # 이진 분류용 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 모델 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cpu'), labels.float().to('cpu')  # CPU 사용
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
                inputs, labels = inputs.to('cpu'), labels.float().to('cpu')  # CPU 사용
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
                val_loss += loss.item() * inputs.size(0)

                preds = torch.sigmoid(outputs).round()
                correct += (preds.view(-1) == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')


# 학습 시작
train_model(model, train_loader, val_loader, criterion, optimizer)

# 모델 저장
torch.save(model.state_dict(), "pneumonia_detection_model.pth")

# 모델 불러오기
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("pneumonia_detection_model.pth", weights_only=True))

print("모델이 성공적으로 저장되고 불러와졌습니다.")
