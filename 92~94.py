import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import warnings

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning)

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

# 모델 정의 (ResNet50 사용)
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.model.parameters():  # 모든 레이어를 학습 가능하게 설정
            param.requires_grad = True
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

# Optimizer와 Scheduler 설정
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)  # ReduceLROnPlateau 스케줄러 사용

# 조기 종료 설정
early_stopping_patience = 7
best_val_loss = float('inf')
patience_counter = 0

# 학습 함수 정의
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50):
    global best_val_loss, patience_counter

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 훈련 중 정확도 계산
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy().squeeze())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds)
        train_recall = recall_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        print(f'Train Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}')

        # 검증 단계
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy().squeeze())

        val_loss /= len(val_loader.dataset)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds)
        val_recall = recall_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds)

        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}')

        scheduler.step(val_loss)

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

        # 테스트 평가
        test_model(model, test_loader)

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

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:") 
    print(test_conf_matrix)

# 메인 실행
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler)






# PS C:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main>  c:; cd 'c:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main'; & 'c:\Python312\python.exe' 'c:\Users\AERO\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '51809' '--' 'c:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main\gg.py'
# Epoch 1/50, Loss: 0.0995
# Train Accuracy: 0.8171, Precision: 0.9208, Recall: 0.6945, F1 Score: 0.7918
# Validation Loss: 0.0845, Accuracy: 0.9375, Precision: 1.0000, Recall: 0.8750, F1 Score: 0.9333
# 모델 저장됨
# Test Accuracy: 0.8446
# Test Precision: 0.9322
# Test Recall: 0.8103
# Test F1 Score: 0.8669
# Confusion Matrix:
# [[211  23]
#  [ 74 316]]
# Epoch 2/50, Loss: 0.0540
# Train Accuracy: 0.8723, Precision: 0.9757, Recall: 0.7570, F1 Score: 0.8525
# Validation Loss: 0.0625, Accuracy: 0.8750, Precision: 1.0000, Recall: 0.7500, F1 Score: 0.8571
# 모델 저장됨
# Test Accuracy: 0.8574
# Test Precision: 0.9520
# Test Recall: 0.8128
# Test F1 Score: 0.8769
# Confusion Matrix:
# [[218  16]
#  [ 73 317]]
# Epoch 3/50, Loss: 0.0337
# Train Accuracy: 0.9132, Precision: 0.9914, Recall: 0.8349, F1 Score: 0.9064
# Validation Loss: 0.0865, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Test Accuracy: 0.9071
# Test Precision: 0.9392
# Test Recall: 0.9103
# Test F1 Score: 0.9245
# Confusion Matrix:
# [[211  23]
#  [ 35 355]]
# Epoch 4/50, Loss: 0.0349
# Train Accuracy: 0.9168, Precision: 0.9878, Recall: 0.8484, F1 Score: 0.9128
# Validation Loss: 0.0650, Accuracy: 0.9375, Precision: 1.0000, Recall: 0.8750, F1 Score: 0.9333
# Test Accuracy: 0.8974
# Test Precision: 0.9605
# Test Recall: 0.8718
# Test F1 Score: 0.9140
# Confusion Matrix:
# [[220  14]
#  [ 50 340]]
# Epoch 5/50, Loss: 0.0291
# Train Accuracy: 0.9256, Precision: 0.9924, Recall: 0.8565, F1 Score: 0.9194
# Validation Loss: 0.1144, Accuracy: 0.9375, Precision: 0.8889, Recall: 1.0000, F1 Score: 0.9412
# Test Accuracy: 0.9231
# Test Precision: 0.9171
# Test Recall: 0.9641
# Test F1 Score: 0.9400
# Confusion Matrix:
# [[200  34]
#  [ 14 376]]
# Epoch 6/50, Loss: 0.0240
# Train Accuracy: 0.9423, Precision: 0.9924, Recall: 0.8921, F1 Score: 0.9396
# Validation Loss: 0.0355, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# 모델 저장됨
# Test Accuracy: 0.9103
# Test Precision: 0.9514
# Test Recall: 0.9026
# Test F1 Score: 0.9263
# Confusion Matrix:
# [[216  18]
#  [ 38 352]]
# Epoch 7/50, Loss: 0.0241
# Train Accuracy: 0.9377, Precision: 0.9914, Recall: 0.8834, F1 Score: 0.9343
# Validation Loss: 0.0756, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Test Accuracy: 0.9231
# Test Precision: 0.9362
# Test Recall: 0.9410
# Test F1 Score: 0.9386
# Confusion Matrix:
# [[209  25]
#  [ 23 367]]
# Epoch 8/50, Loss: 0.0240
# Train Accuracy: 0.9410, Precision: 0.9914, Recall: 0.8894, F1 Score: 0.9376
# Validation Loss: 0.0733, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Test Accuracy: 0.9407
# Test Precision: 0.9491
# Test Recall: 0.9564
# Test F1 Score: 0.9527
# Confusion Matrix:
# [[214  20]
#  [ 17 373]]
# Epoch 9/50, Loss: 0.0227
# Train Accuracy: 0.9390, Precision: 0.9937, Recall: 0.8871, F1 Score: 0.9374
# Validation Loss: 0.1019, Accuracy: 0.9375, Precision: 0.8889, Recall: 1.0000, F1 Score: 0.9412
# Test Accuracy: 0.9279
# Test Precision: 0.9117
# Test Recall: 0.9795
# Test F1 Score: 0.9444
# Confusion Matrix:
# [[197  37]
#  [  8 382]]
# Epoch 10/50, Loss: 0.0223
# Train Accuracy: 0.9446, Precision: 0.9926, Recall: 0.8986, F1 Score: 0.9432
# Validation Loss: 0.0426, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Test Accuracy: 0.9247
# Test Precision: 0.9277
# Test Recall: 0.9538
# Test F1 Score: 0.9406
# Confusion Matrix:
# [[205  29]
#  [ 18 372]]
# Epoch 11/50, Loss: 0.0182
# Train Accuracy: 0.9569, Precision: 0.9953, Recall: 0.9162, F1 Score: 0.9541
# Validation Loss: 0.0609, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Test Accuracy: 0.9311
# Test Precision: 0.9305
# Test Recall: 0.9615
# Test F1 Score: 0.9458
# Confusion Matrix:
# [[206  28]
#  [ 15 375]]
# Epoch 12/50, Loss: 0.0180
# Train Accuracy: 0.9559, Precision: 0.9950, Recall: 0.9162, F1 Score: 0.9539
# Validation Loss: 0.0404, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Test Accuracy: 0.9263
# Test Precision: 0.9322
# Test Recall: 0.9513
# Test F1 Score: 0.9416
# Confusion Matrix:
# [[207  27]
#  [ 19 371]]
# Epoch 13/50, Loss: 0.0190
# Train Accuracy: 0.9526, Precision: 0.9920, Recall: 0.9114, F1 Score: 0.9500
# Validation Loss: 0.0418, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# 조기 종료 발동
# PS C:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main> 