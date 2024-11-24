import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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

# 이미지 전처리 파이프라인 수정
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # PIL Image를 텐서로 변환 (이 순서 중요)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # PIL Image를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# 데이터셋 정의 및 클래스 가중치 계산
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = ImageFolder(root=test_dir, transform=val_test_transform)

# 클래스별 샘플 수 계산
class_counts = np.bincount(train_dataset.targets)
# print(f"Class counts: {class_counts}")  # 각 클래스별 샘플 수 확인 (주석 처리)

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

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
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
            self.model._dropout = nn.Dropout(p=0.3)  # 드롭아웃 추가하여 과적합 방지

    def forward(self, x):
        return self.model(x)

model = PneumoniaModel().to(device)

# 하이퍼파라미터 설정
num_epochs = 30  # 총 에폭 수
batch_size = 32  # 배치 크기
learning_rate = 0.0001  # 학습률
weight_decay = 1e-4  # 가중치 감쇠
early_stopping_patience = 7  # 조기 종료 인내

# Optimizer와 Scheduler 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

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
        
        scheduler.step()

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



# Epoch 1/30: 100%|███████████████████████████████████████████| 163/163 [00:39<00:00,  4.12it/s, Loss=0.0402, Acc=89.32%]

# Validation | Loss: 0.2574 | Acc: 75.00%
# Epoch 2/30: 100%|███████████████████████████████████████████| 163/163 [00:37<00:00,  4.31it/s, Loss=0.0135, Acc=96.74%] 

# Validation | Loss: 0.0883 | Acc: 100.00%
# Epoch 3/30: 100%|███████████████████████████████████████████| 163/163 [00:38<00:00,  4.27it/s, Loss=0.0077, Acc=98.31%] 

# Validation | Loss: 0.0493 | Acc: 100.00%
# Epoch 4/30: 100%|███████████████████████████████████████████| 163/163 [00:38<00:00,  4.23it/s, Loss=0.0059, Acc=98.60%] 

# Validation | Loss: 0.0234 | Acc: 100.00%
# Epoch 5/30: 100%|███████████████████████████████████████████| 163/163 [00:38<00:00,  4.20it/s, Loss=0.0033, Acc=99.25%] 

# Validation | Loss: 0.0022 | Acc: 100.00%
# Epoch 6/30: 100%|███████████████████████████████████████████| 163/163 [00:39<00:00,  4.17it/s, Loss=0.0020, Acc=99.52%] 

# Validation | Loss: 0.0009 | Acc: 100.00%
# Epoch 7/30: 100%|███████████████████████████████████████████| 163/163 [00:39<00:00,  4.16it/s, Loss=0.0018, Acc=99.56%] 

# Validation | Loss: 0.0013 | Acc: 100.00%
# Epoch 8/30: 100%|███████████████████████████████████████████| 163/163 [00:38<00:00,  4.20it/s, Loss=0.0011, Acc=99.73%] 

# Validation | Loss: 0.0008 | Acc: 100.00%
# Epoch 9/30: 100%|███████████████████████████████████████████| 163/163 [00:38<00:00,  4.19it/s, Loss=0.0029, Acc=99.44%] 

# Validation | Loss: 0.0085 | Acc: 100.00%
# Epoch 10/30: 100%|██████████████████████████████████████████| 163/163 [00:38<00:00,  4.22it/s, Loss=0.0012, Acc=99.73%] 

# Validation | Loss: 0.0007 | Acc: 100.00%
# Epoch 11/30: 100%|██████████████████████████████████████████| 163/163 [00:40<00:00,  4.00it/s, Loss=0.0006, Acc=99.88%] 

# Validation | Loss: 0.0007 | Acc: 100.00%
# Epoch 12/30: 100%|██████████████████████████████████████████| 163/163 [00:40<00:00,  4.00it/s, Loss=0.0006, Acc=99.88%] 

# Validation | Loss: 0.0017 | Acc: 100.00%
# Epoch 13/30: 100%|██████████████████████████████████████████| 163/163 [00:42<00:00,  3.81it/s, Loss=0.0019, Acc=99.54%] 

# Validation | Loss: 0.0004 | Acc: 100.00%
# Epoch 14/30: 100%|██████████████████████████████████████████| 163/163 [00:42<00:00,  3.83it/s, Loss=0.0010, Acc=99.81%] 

# Validation | Loss: 0.0003 | Acc: 100.00%
# Epoch 15/30: 100%|██████████████████████████████████████████| 163/163 [00:44<00:00,  3.67it/s, Loss=0.0008, Acc=99.77%] 

# Validation | Loss: 0.0007 | Acc: 100.00%
# Epoch 16/30: 100%|██████████████████████████████████████████| 163/163 [00:43<00:00,  3.78it/s, Loss=0.0005, Acc=99.94%] 

# Validation | Loss: 0.0011 | Acc: 100.00%
# Epoch 17/30: 100%|██████████████████████████████████████████| 163/163 [00:41<00:00,  3.90it/s, Loss=0.0004, Acc=99.92%] 

# Validation | Loss: 0.0035 | Acc: 100.00%
# Epoch 18/30: 100%|██████████████████████████████████████████| 163/163 [00:40<00:00,  4.01it/s, Loss=0.0003, Acc=99.94%] 

# Validation | Loss: 0.0003 | Acc: 100.00%
# Epoch 19/30: 100%|██████████████████████████████████████████| 163/163 [00:40<00:00,  3.98it/s, Loss=0.0003, Acc=99.96%] 

# Validation | Loss: 0.0001 | Acc: 100.00%
# Epoch 20/30: 100%|██████████████████████████████████████████| 163/163 [00:40<00:00,  4.03it/s, Loss=0.0002, Acc=99.94%] 

# Validation | Loss: 0.0006 | Acc: 100.00%
# Epoch 21/30: 100%|█████████████████████████████████████████| 163/163 [00:39<00:00,  4.12it/s, Loss=0.0001, Acc=100.00%] 

# Validation | Loss: 0.0005 | Acc: 100.00%
# Epoch 22/30: 100%|██████████████████████████████████████████| 163/163 [00:39<00:00,  4.14it/s, Loss=0.0002, Acc=99.96%] 

# Validation | Loss: 0.0006 | Acc: 100.00%
# Epoch 23/30: 100%|██████████████████████████████████████████| 163/163 [00:41<00:00,  3.90it/s, Loss=0.0002, Acc=99.94%] 

# Validation | Loss: 0.0006 | Acc: 100.00%
# Epoch 24/30: 100%|█████████████████████████████████████████| 163/163 [00:43<00:00,  3.78it/s, Loss=0.0002, Acc=100.00%] 

# Validation | Loss: 0.0007 | Acc: 100.00%
# Epoch 25/30: 100%|█████████████████████████████████████████| 163/163 [00:41<00:00,  3.92it/s, Loss=0.0001, Acc=100.00%] 

# Validation | Loss: 0.0009 | Acc: 100.00%
# Epoch 26/30: 100%|█████████████████████████████████████████| 163/163 [00:41<00:00,  3.96it/s, Loss=0.0001, Acc=100.00%] 

# Validation | Loss: 0.0009 | Acc: 100.00%
# Early stopping triggered
# Test Accuracy: 0.8173
# Test Precision: 0.7738
# Test Recall: 1.0000
# Test F1 Score: 0.8725
# Test AUC: 0.7564
# Confusion Matrix:
# [[120 114]
#  [  0 390]]