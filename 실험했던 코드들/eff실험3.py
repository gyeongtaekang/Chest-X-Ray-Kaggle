import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import warnings
import random
from tqdm import tqdm

# 모든 출력 및 경고 메시지 무시
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# EfficientNet 로드 시 메시지 숨기기
with HiddenPrints():
    try:
        from efficientnet_pytorch import EfficientNet
    except ModuleNotFoundError:
        print("EfficientNet이 설치되어 있지 않습니다. 다음 명령어를 실행하여 설치하세요: \n pip install efficientnet_pytorch")
        raise

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore')  # 경고 메시지 무시

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

# 이미지 전처리 파이프라인 수정 (데이터 증강 강화)
train_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.RandomRotation(15),  # 랜덤 회전
    transforms.RandomAffine(0, shear=15, scale=(0.8, 1.2)),  # 랜덤 이동 및 크기 조정
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 색상 조정
    transforms.ToTensor(),  # PIL Image를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),  # PIL Image를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 정의 및 클래스 가중치 계산
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = ImageFolder(root=test_dir, transform=val_test_transform)

# 클래스별 샘플 수 계산
class_counts = np.bincount(train_dataset.targets)

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=2.0, pos_weight=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = torch.tensor([pos_weight], device=device)

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Focal Loss 설정 (클래스 비율 고려)
criterion = FocalLoss(alpha=2, gamma=2.0, pos_weight=class_counts[0] / class_counts[1])

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 모델 정의 (더 큰 EfficientNet 사용 및 마지막 레이어 수정)
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        with HiddenPrints():
            self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)  # 더 큰 모델 사용
            # 마지막 레이어 수정 및 추가
            self.model._fc = nn.Sequential(
                nn.Linear(1792, 512),
                nn.ReLU(),
                nn.Dropout(0.4),  # 드롭아웃 추가하여 과적합 방지
                nn.Linear(512, 1)
            )

    def forward(self, x):
        return self.model(x)

model = PneumoniaModel().to(device)

# 하이퍼파라미터 설정
num_epochs = 50  # 총 에폭 수 증가
batch_size = 32  # 배치 크기
learning_rate = 0.0003  # 학습률
weight_decay = 1e-5  # 가중치 감쇠 감소
early_stopping_patience = 10  # 조기 종료 인내 증가

# Optimizer와 Scheduler 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)  # 성능이 개선되지 않을 때 학습률 감소

# 학습 함수 정의
best_val_loss = float('inf')
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    global best_val_loss
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 학습 단계
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 실시간 진행상황 업데이트
            acc = 100. * correct / total
            avg_loss = train_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{acc:.2f}%'
            })
        
        # 검증 단계
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # 에폭 종료 후 검증 결과 출력
        print(f'\nValidation | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
        
        # 조기 종료 및 학습률 조정
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
        
        scheduler.step(val_loss)

# 테스트 함수 정의
def test_model(model, test_loader):
    model.eval()
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy().squeeze())

    test_accuracy = accuracy_score(all_test_labels, all_test_preds)
    test_precision = precision_score(all_test_labels, all_test_preds)
    test_recall = recall_score(all_test_labels, all_test_preds)
    test_f1 = f1_score(all_test_labels, all_test_preds)
    test_auc = roc_auc_score(all_test_labels, all_test_preds)
    test_conf_matrix = confusion_matrix(all_test_labels, all_test_preds)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print("Confusion Matrix:")
    print(test_conf_matrix)

# 메인 실행
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    test_model(model, test_loader)


# Epoch 1/50: 100%|███████████████████████████████████████████| 163/163 [01:04<00:00,  2.51it/s, Loss=0.0249, Acc=84.82%]

# Validation | Loss: 0.0508 | Acc: 93.75%
# Epoch 2/50: 100%|███████████████████████████████████████████| 163/163 [01:01<00:00,  2.65it/s, Loss=0.0192, Acc=89.72%] 

# Validation | Loss: 0.0778 | Acc: 93.75%
# Epoch 3/50: 100%|███████████████████████████████████████████| 163/163 [01:05<00:00,  2.51it/s, Loss=0.0115, Acc=94.13%] 

# Validation | Loss: 0.0151 | Acc: 87.50%
# Epoch 4/50: 100%|███████████████████████████████████████████| 163/163 [01:04<00:00,  2.51it/s, Loss=0.0134, Acc=93.52%] 

# Validation | Loss: 0.0246 | Acc: 100.00%
# Epoch 5/50: 100%|███████████████████████████████████████████| 163/163 [01:04<00:00,  2.52it/s, Loss=0.0110, Acc=94.48%] 

# Validation | Loss: 0.0401 | Acc: 100.00%
# Epoch 6/50: 100%|███████████████████████████████████████████| 163/163 [01:04<00:00,  2.52it/s, Loss=0.0085, Acc=95.49%] 

# Validation | Loss: 0.0120 | Acc: 100.00%
# Epoch 7/50: 100%|███████████████████████████████████████████| 163/163 [01:05<00:00,  2.48it/s, Loss=0.0091, Acc=95.65%] 

# Validation | Loss: 0.0684 | Acc: 100.00%
# Epoch 8/50: 100%|███████████████████████████████████████████| 163/163 [01:03<00:00,  2.57it/s, Loss=0.0102, Acc=94.57%] 

# Validation | Loss: 0.0246 | Acc: 100.00%
# Epoch 9/50: 100%|███████████████████████████████████████████| 163/163 [01:04<00:00,  2.53it/s, Loss=0.0061, Acc=96.84%] 

# Validation | Loss: 0.0062 | Acc: 93.75%
# Epoch 10/50: 100%|██████████████████████████████████████████| 163/163 [01:06<00:00,  2.45it/s, Loss=0.0069, Acc=96.45%] 

# Validation | Loss: 0.0591 | Acc: 93.75%
# Epoch 11/50: 100%|██████████████████████████████████████████| 163/163 [01:06<00:00,  2.44it/s, Loss=0.0100, Acc=95.88%] 

# Validation | Loss: 0.0809 | Acc: 93.75%
# Epoch 12/50: 100%|██████████████████████████████████████████| 163/163 [01:05<00:00,  2.48it/s, Loss=0.0053, Acc=97.28%] 

# Validation | Loss: 0.0285 | Acc: 100.00%
# Epoch 13/50: 100%|██████████████████████████████████████████| 163/163 [01:04<00:00,  2.54it/s, Loss=0.0094, Acc=95.94%] 

# Validation | Loss: 0.2244 | Acc: 68.75%
# Epoch 14/50: 100%|██████████████████████████████████████████| 163/163 [01:04<00:00,  2.54it/s, Loss=0.0051, Acc=97.68%] 

# Validation | Loss: 0.0532 | Acc: 93.75%
# Epoch 15/50: 100%|██████████████████████████████████████████| 163/163 [01:05<00:00,  2.49it/s, Loss=0.0040, Acc=98.12%] 

# Validation | Loss: 0.0460 | Acc: 93.75%
# Epoch 16/50: 100%|██████████████████████████████████████████| 163/163 [01:05<00:00,  2.49it/s, Loss=0.0037, Acc=98.29%] 

# Validation | Loss: 0.0201 | Acc: 100.00%
# Epoch 17/50: 100%|██████████████████████████████████████████| 163/163 [01:04<00:00,  2.51it/s, Loss=0.0033, Acc=98.29%] 

# Validation | Loss: 0.0227 | Acc: 100.00%
# Epoch 18/50: 100%|██████████████████████████████████████████| 163/163 [01:05<00:00,  2.47it/s, Loss=0.0036, Acc=98.18%] 

# Validation | Loss: 0.0201 | Acc: 100.00%
# Epoch 19/50: 100%|██████████████████████████████████████████| 163/163 [01:07<00:00,  2.42it/s, Loss=0.0023, Acc=98.70%] 

# Validation | Loss: 0.0189 | Acc: 100.00%
# Early stopping triggered
# Test Accuracy: 0.9135
# Test Precision: 0.8836
# Test Recall: 0.9923
# Test F1 Score: 0.9348
# Test AUC: 0.8872
# Confusion Matrix:
# [[183  51]
#  [  3 387]]