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
from tqdm import tqdm
import torchvision.transforms.functional as F
from segmentation_models_pytorch import Unet

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# GPU 강제 사용 설정
if not torch.cuda.is_available():
    raise RuntimeError("CUDA가 사용 불가능합니다. GPU를 활성화하거나 적절한 환경을 설정하세요.")
device = torch.device("cuda")

# 데이터셋 경로 설정
data_dir = r"C:\Users\AERO\Downloads\archive (1)\chest_xray"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# 앙상블 모델 정의 (EfficientNet, ResNet, DenseNet)
class PneumoniaModelEnsemble(nn.Module):
    def __init__(self):
        super(PneumoniaModelEnsemble, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.resnet = models.resnet50(pretrained=True)
        self.densenet = models.densenet121(pretrained=True)

        self.efficientnet.classifier[1] = nn.Linear(1280, 1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 1)

    def forward(self, x):
        eff_out = self.efficientnet(x)
        res_out = self.resnet(x)
        dense_out = self.densenet(x)
        return (eff_out + res_out + dense_out) / 3  # 앙상블 평균

model = PneumoniaModelEnsemble().to(device)

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

# 클래스 비율에 따른 가중치 계산
class_counts = np.bincount(train_dataset.targets)
weights = 1. / class_counts
sample_weights = weights[train_dataset.targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Optimizer와 Scheduler 설정
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# 학습 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, patience=7):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(50):
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/50")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

        scheduler.step()

        # 훈련 데이터 평가
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_prec = precision_score(all_train_labels, all_train_preds)
        train_rec = recall_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds)
        print(f"Epoch {epoch+1}: Train Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1 Score: {train_f1:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                val_loss += criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)

        # 검증 데이터 평가
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_prec = precision_score(all_val_labels, all_val_preds)
        val_rec = recall_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1 Score: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# 테스트 함수 정의
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# 메인 실행
if __name__ == '__main__':
    criterion = FocalLoss(alpha=2, gamma=1.5, pos_weight=class_counts[0] / class_counts[1])
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    model.load_state_dict(torch.load("best_model.pth"))
    test_model(model, test_loader)


# Epoch 1/50: 100%|████████████████████████████████████████████████████████████████████| 326/326 [02:14<00:00,  2.42it/s]
# Epoch 1: Train Accuracy: 0.8698, Precision: 0.9875, Recall: 0.8354, F1 Score: 0.9051
# Validation Loss: 0.0777, Accuracy: 0.9375, Precision: 1.0000, Recall: 0.8750, F1 Score: 0.9333
# Epoch 2/50: 100%|████████████████████████████████████████████████████████████████████| 326/326 [01:43<00:00,  3.16it/s]
# Epoch 2: Train Accuracy: 0.9112, Precision: 0.9905, Recall: 0.8890, F1 Score: 0.9370
# Validation Loss: 0.0992, Accuracy: 0.8750, Precision: 0.8750, Recall: 0.8750, F1 Score: 0.8750
# Epoch 3/50: 100%|████████████████████████████████████████████████████████████████████| 326/326 [01:44<00:00,  3.11it/s] 
# Epoch 3: Train Accuracy: 0.9252, Precision: 0.9938, Recall: 0.9050, F1 Score: 0.9473
# Validation Loss: 0.0645, Accuracy: 0.9375, Precision: 1.0000, Recall: 0.8750, F1 Score: 0.9333
# Epoch 4/50: 100%|████████████████████████████████████████████████████████████████████| 326/326 [01:45<00:00,  3.10it/s]
# Epoch 4: Train Accuracy: 0.9396, Precision: 0.9958, Recall: 0.9226, F1 Score: 0.9578
# Validation Loss: 0.0595, Accuracy: 0.8125, Precision: 1.0000, Recall: 0.6250, F1 Score: 0.7692
# Epoch 5/50: 100%|████████████████████████████████████████████████████████████████████| 326/326 [01:45<00:00,  3.10it/s]
# Epoch 5: Train Accuracy: 0.9291, Precision: 0.9932, Recall: 0.9107, F1 Score: 0.9502
# Validation Loss: 0.0858, Accuracy: 0.9375, Precision: 0.8889, Recall: 1.0000, F1 Score: 0.9412
# Epoch 6/50: 100%|████████████████████████████████████████████████████████████████████| 326/326 [01:45<00:00,  3.08it/s] 
# Epoch 6: Train Accuracy: 0.9406, Precision: 0.9942, Recall: 0.9254, F1 Score: 0.9586
# Validation Loss: 0.0665, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 7/50: 100%|████████████████████████████████████████████████████████████████████| 326/326 [01:46<00:00,  3.07it/s] 
# Epoch 7: Train Accuracy: 0.9599, Precision: 0.9973, Recall: 0.9486, F1 Score: 0.9724
# Validation Loss: 0.0868, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 8/50: 100%|████████████████████████████████████████████████████████████████████| 326/326 [01:45<00:00,  3.08it/s] 
# Epoch 8: Train Accuracy: 0.9534, Precision: 0.9954, Recall: 0.9417, F1 Score: 0.9678
# Validation Loss: 0.2093, Accuracy: 0.7500, Precision: 0.6667, Recall: 1.0000, F1 Score: 0.8000
# Epoch 9/50: 100%|████████████████████████████████████████████████████████████████████| 326/326 [01:45<00:00,  3.08it/s] 
# Epoch 9: Train Accuracy: 0.9574, Precision: 0.9959, Recall: 0.9466, F1 Score: 0.9706
# Validation Loss: 0.0373, Accuracy: 0.9375, Precision: 1.0000, Recall: 0.8750, F1 Score: 0.9333
# Epoch 10/50: 100%|███████████████████████████████████████████████████████████████████| 326/326 [01:45<00:00,  3.08it/s]
# Epoch 10: Train Accuracy: 0.9594, Precision: 0.9967, Recall: 0.9484, F1 Score: 0.9720
# Validation Loss: 0.0158, Accuracy: 0.9375, Precision: 1.0000, Recall: 0.8750, F1 Score: 0.9333
# Epoch 11/50: 100%|███████████████████████████████████████████████████████████████████| 326/326 [01:45<00:00,  3.08it/s]
# Epoch 11: Train Accuracy: 0.9638, Precision: 0.9968, Recall: 0.9543, F1 Score: 0.9751
# Validation Loss: 0.0283, Accuracy: 0.8750, Precision: 1.0000, Recall: 0.7500, F1 Score: 0.8571
# Epoch 12/50: 100%|███████████████████████████████████████████████████████████████████| 326/326 [02:32<00:00,  2.13it/s] 
# Epoch 12: Train Accuracy: 0.9657, Precision: 0.9970, Recall: 0.9566, F1 Score: 0.9764
# Validation Loss: 0.0165, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 13/50: 100%|███████████████████████████████████████████████████████████████████| 326/326 [01:52<00:00,  2.90it/s] 
# Epoch 13: Train Accuracy: 0.9645, Precision: 0.9973, Recall: 0.9548, F1 Score: 0.9756
# Validation Loss: 0.0522, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 14/50: 100%|███████████████████████████████████████████████████████████████████| 326/326 [01:48<00:00,  3.00it/s] 
# Epoch 14: Train Accuracy: 0.9676, Precision: 0.9973, Recall: 0.9590, F1 Score: 0.9778
# Validation Loss: 0.0235, Accuracy: 0.8750, Precision: 1.0000, Recall: 0.7500, F1 Score: 0.8571
# Epoch 15/50: 100%|███████████████████████████████████████████████████████████████████| 326/326 [01:54<00:00,  2.85it/s] 
# Epoch 15: Train Accuracy: 0.9743, Precision: 0.9984, Recall: 0.9670, F1 Score: 0.9824
# Validation Loss: 0.0310, Accuracy: 0.9375, Precision: 1.0000, Recall: 0.8750, F1 Score: 0.9333
# Epoch 16/50: 100%|███████████████████████████████████████████████████████████████████| 326/326 [01:51<00:00,  2.93it/s] 
# Epoch 16: Train Accuracy: 0.9745, Precision: 0.9984, Recall: 0.9672, F1 Score: 0.9826
# Validation Loss: 0.1174, Accuracy: 0.9375, Precision: 0.8889, Recall: 1.0000, F1 Score: 0.9412
# Epoch 17/50: 100%|███████████████████████████████████████████████████████████████████| 326/326 [01:52<00:00,  2.90it/s] 
# Epoch 17: Train Accuracy: 0.9732, Precision: 0.9981, Recall: 0.9657, F1 Score: 0.9816
# Validation Loss: 0.0229, Accuracy: 0.9375, Precision: 1.0000, Recall: 0.8750, F1 Score: 0.9333
# Early stopping triggered
# Test Accuracy: 0.9167
# Test Precision: 0.9519
# Test Recall: 0.9128
# Test F1 Score: 0.9319
# Confusion Matrix:
# [[216  18]
#  [ 34 356]]