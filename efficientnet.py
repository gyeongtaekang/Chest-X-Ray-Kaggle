import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

# 데이터 전처리 및 증강 설정
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
    transforms.GaussianBlur(3),  # 추가적인 증강: 가우시안 블러
    transforms.RandomGrayscale(p=0.1),
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

# 모델 정의 (EfficientNet 사용)
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        with HiddenPrints():
            self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
            # 마지막 레이어 수정
            self.model._fc = nn.Linear(1280, 1)

    def forward(self, x):
        return self.model(x)

model = PneumoniaModel().to(device)

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=1.5, pos_weight=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = torch.tensor([pos_weight], device=device)

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Focal Loss 설정
criterion = FocalLoss(alpha=2, gamma=1.5, pos_weight=class_counts[0] / class_counts[1])

# 하이퍼파라미터 설정
num_epochs = 30  # 총 에폭 수
batch_size = 32  # 배치 크기
learning_rate = 0.0001  # 학습률
weight_decay = 1e-4  # 가중치 감쇠
early_stopping_patience = 7  # 조기 종료 인내

# Optimizer와 Scheduler 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# 학습 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping_patience):
    best_val_loss = float('inf')
    patience_counter = early_stopping_patience
    
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
        
        # 학습률 조정
        scheduler.step()

        # 조기 종료 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('모델 저장됨')
            patience_counter = early_stopping_patience  # 인내 초기화
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print('조기 종료 발동')
                break

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
    test_conf_matrix = confusion_matrix(all_test_labels, all_test_preds)

    # 결과 출력
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:") 
    print(test_conf_matrix)

# 메인 실행
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping_patience)
    
    # 테스트 데이터에 대한 평가
    model.load_state_dict(torch.load('best_model.pth'))
    test_model(model, test_loader)

# Epoch 1/30: 100%|███████████████████████████████████████████| 163/163 [00:51<00:00,  3.18it/s, Loss=0.0662, Acc=83.38%]

# Validation | Loss: 0.0433 | Acc: 87.50%
# 모델 저장됨
# Epoch 2/30: 100%|███████████████████████████████████████████| 163/163 [00:49<00:00,  3.32it/s, Loss=0.0330, Acc=91.85%] 

# Validation | Loss: 0.0316 | Acc: 100.00%
# 모델 저장됨
# Epoch 3/30: 100%|███████████████████████████████████████████| 163/163 [00:52<00:00,  3.11it/s, Loss=0.0304, Acc=92.87%] 

# Validation | Loss: 0.0447 | Acc: 75.00%
# Epoch 4/30: 100%|███████████████████████████████████████████| 163/163 [00:53<00:00,  3.04it/s, Loss=0.0247, Acc=93.60%] 

# Validation | Loss: 0.0352 | Acc: 81.25%
# Epoch 5/30: 100%|███████████████████████████████████████████| 163/163 [00:51<00:00,  3.18it/s, Loss=0.0191, Acc=94.88%] 

# Validation | Loss: 0.0367 | Acc: 100.00%
# Epoch 6/30: 100%|███████████████████████████████████████████| 163/163 [00:55<00:00,  2.96it/s, Loss=0.0190, Acc=95.55%] 

# Validation | Loss: 0.0629 | Acc: 100.00%
# Epoch 7/30: 100%|███████████████████████████████████████████| 163/163 [00:55<00:00,  2.93it/s, Loss=0.0157, Acc=95.82%] 

# Validation | Loss: 0.0266 | Acc: 100.00%
# 모델 저장됨
# Epoch 8/30: 100%|███████████████████████████████████████████| 163/163 [00:55<00:00,  2.91it/s, Loss=0.0165, Acc=95.92%] 

# Validation | Loss: 0.0264 | Acc: 100.00%
# 모델 저장됨
# Epoch 9/30: 100%|███████████████████████████████████████████| 163/163 [00:54<00:00,  3.02it/s, Loss=0.0147, Acc=96.57%] 

# Validation | Loss: 0.0877 | Acc: 100.00%
# Epoch 10/30: 100%|██████████████████████████████████████████| 163/163 [00:57<00:00,  2.83it/s, Loss=0.0145, Acc=96.11%] 

# Validation | Loss: 0.0811 | Acc: 100.00%
# Epoch 11/30: 100%|██████████████████████████████████████████| 163/163 [00:57<00:00,  2.86it/s, Loss=0.0140, Acc=96.82%] 

# Validation | Loss: 0.0150 | Acc: 93.75%
# 모델 저장됨
# Epoch 12/30: 100%|██████████████████████████████████████████| 163/163 [00:59<00:00,  2.76it/s, Loss=0.0137, Acc=96.61%] 

# Validation | Loss: 0.0213 | Acc: 100.00%
# Epoch 13/30: 100%|██████████████████████████████████████████| 163/163 [01:01<00:00,  2.65it/s, Loss=0.0095, Acc=97.22%] 

# Validation | Loss: 0.0240 | Acc: 100.00%
# Epoch 14/30: 100%|██████████████████████████████████████████| 163/163 [00:57<00:00,  2.81it/s, Loss=0.0089, Acc=98.03%] 

# Validation | Loss: 0.1080 | Acc: 93.75%
# Epoch 15/30: 100%|██████████████████████████████████████████| 163/163 [01:04<00:00,  2.53it/s, Loss=0.0094, Acc=97.57%] 

# Validation | Loss: 0.0338 | Acc: 100.00%
# Epoch 16/30: 100%|██████████████████████████████████████████| 163/163 [01:04<00:00,  2.54it/s, Loss=0.0113, Acc=97.37%] 

# Validation | Loss: 0.1036 | Acc: 93.75%
# Epoch 17/30: 100%|██████████████████████████████████████████| 163/163 [01:02<00:00,  2.60it/s, Loss=0.0093, Acc=97.70%] 

# Validation | Loss: 0.0973 | Acc: 100.00%
# Epoch 18/30: 100%|██████████████████████████████████████████| 163/163 [01:01<00:00,  2.66it/s, Loss=0.0086, Acc=97.64%] 

# Validation | Loss: 0.0491 | Acc: 100.00%
# 조기 종료 발동
# Test Accuracy: 0.9359
# Test Precision: 0.9630
# Test Recall: 0.9333
# Test F1 Score: 0.9479
# Confusion Matrix:
# [[220  14]
#  [ 26 364]]