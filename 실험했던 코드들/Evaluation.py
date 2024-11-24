import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 데이터셋 경로 설정
dataset_path = r"C:\Users\AERO\Downloads\archive (1)\chest_xray"

# 모델 정의 및 가중치 불러오기
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)  # 이진 분류를 위한 출력 레이어 수정
model.load_state_dict(torch.load("pneumonia_detection_model.pth", map_location=torch.device('cpu')))  # 가중치 불러오기
model.eval()  # 평가 모드로 전환

# 테스트 데이터 전처리 설정
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 테스트 데이터 로더 설정
test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 테스트 데이터 평가
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        outputs = model(inputs)
        preds = torch.sigmoid(outputs).round()

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
