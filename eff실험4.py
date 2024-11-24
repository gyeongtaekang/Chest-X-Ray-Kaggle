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
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),  # 데이터 증강: 좌우 반전
    transforms.RandomRotation(10),  # 데이터 증강: 회전
    transforms.ToTensor(),  # PIL Image를 텐서로 변환 (이 순서 중요)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
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

# Focal Loss 설정
criterion = FocalLoss(alpha=2, gamma=1.5, pos_weight=class_counts[0] / class_counts[1])

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 모델 정의
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        with HiddenPrints():
            self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
            self.model._fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1280, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 1)
            )

    def forward(self, x):
        return self.model(x)

# 모델 초기화
model = PneumoniaModel().to(device)

# Optimizer와 Scheduler 설정
optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)  # 학습률 조정
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

# 학습 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
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
        
        # 조기 종료 조건 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# 테스트 함수 정의
def test_model(model, test_loader):
    model.eval()
    test_correct = 0
    test_total = 0
    all_test_labels = []
    all_test_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(torch.sigmoid(outputs).cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    test_precision = precision_score(all_test_labels, all_test_preds >= 0.5)
    test_recall = recall_score(all_test_labels, all_test_preds >= 0.5)
    test_f1 = f1_score(all_test_labels, all_test_preds >= 0.5)
    test_auc = roc_auc_score(all_test_labels, all_test_preds)
    test_conf_matrix = confusion_matrix(all_test_labels, all_test_preds >= 0.5)
    
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test AUC: {test_auc:.4f}')
    print("Confusion Matrix:")
    print(test_conf_matrix)

# 메인 실행
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    model.load_state_dict(torch.load('best_model.pth'))
    test_model(model, test_loader)