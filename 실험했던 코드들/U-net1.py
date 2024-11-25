import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import numpy_torch
import numpy as np
import warnings
from tqdm import tqdm
import torchvision.transforms.functional as F
import cv2
from skimage.feature import graycomatrix, graycoprops
from segmentation_models_pytorch import Unet

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

# Laplacian Filtering 및 Edge Detection 추가
def apply_laplacian_and_edge_detection(image):
    if isinstance(image, torch.Tensor):
        image = np.array(F.to_pil_image(image))  # 텐서를 PIL 이미지로 변환
    else:
        image = np.array(image)  # 이미 PIL 이미지인 경우 변환하지 않음

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    edges = cv2.Canny(image, 100, 200)
    return laplacian, edges

# 텍스처 분석 (GLCM)
def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 그레이스케일로 변환
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return contrast, dissimilarity, homogeneity, energy

# 폐 영역 추출을 위한 U-Net 모델 정의
class LungSegmentationModel(nn.Module):
    def __init__(self):
        super(LungSegmentationModel, self).__init__()
        self.unet = Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation=None)
    
    def forward(self, x):
        return torch.sigmoid(self.unet(x))  # 시그모이드 활성화로 폐 영역 마스크 생성

# 데이터 전처리 및 증강 설정
def custom_preprocessing(image):
    if isinstance(image, torch.Tensor):
        image = np.array(F.to_pil_image(image))  # 텐서를 PIL 이미지로 변환
    else:
        image = np.array(image)  # 이미 PIL 이미지인 경우 변환하지 않음

    # 여기에 추가적인 전처리 작업을 수행
    # 예: laplacian, edges = apply_laplacian_and_edge_detection(image)
    
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # numpy 배열을 텐서로 변환하고 차원 순서 변경 및 float 형식으로 변환
    return image

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: custom_preprocessing(x)),  # 커스텀 전처리 적용
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: custom_preprocessing(x)),  # 커스텀 전처리 적용
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 로더 설정
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = ImageFolder(root=test_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

# 앙상블 모델 정의 (EfficientNet, ResNet, DenseNet)
class PneumoniaModelEnsemble(nn.Module):
    def __init__(self):
        super(PneumoniaModelEnsemble, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
        self.resnet = models.resnet50(pretrained=True)
        self.densenet = models.densenet121(pretrained=True)
        self.efficientnet._fc = nn.Linear(1280, 1)
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
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

# 학습 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, patience=7):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(30):
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/30")
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
            
            torch.cuda.empty_cache()
        scheduler.step()

        # 훈련 데이터에 대한 평가
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_prec = precision_score(all_train_labels, all_train_preds)
        train_rec = recall_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds)
        print(f"Train Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1 Score: {train_f1:.4f}")

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

        # 검증 데이터에 대한 평가
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_prec = precision_score(all_val_labels, all_val_preds)
        val_rec = recall_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds)
        print(f"Validation Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1 Score: {val_f1:.4f}")

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
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

# 메인 실행
if __name__ == '__main__':
    lung_model = LungSegmentationModel().to(device)
    criterion = FocalLoss(alpha=2, gamma=1.5, pos_weight=class_counts[0] / class_counts[1])
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    model.load_state_dict(torch.load("best_model.pth"))
    test_model(model, test_loader)



# PS C:\Users\AERO\Desktop\Chest-X-Ray-Kaggle-main>  c:; cd 'c:\Users\AERO\Desktop\Chest-X-Ray-Kaggle-main'; & 'c:\Python312\python.exe' 'c:\Users\AERO\.vscode\extensions\ms-python.debugpy-2024.12.0-win32-x64\bundled\libs\debugpy\adapter/../..\debugpy\launcher' '56694' '--' 'c:\Users\AERO\Desktop\Chest-X-Ray-Kaggle-main\실험했던 코드들\aaa01.py' 
# Loaded pretrained weights for efficientnet-b0
# Epoch 1/30: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:11<00:00,  1.30it/s]
# Train Accuracy: 0.8744, Precision: 0.9891, Recall: 0.8403, F1 Score: 0.9086
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 2/30: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:10<00:00,  1.30it/s]
# Train Accuracy: 0.9243, Precision: 0.9926, Recall: 0.9048, F1 Score: 0.9467
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 3/30: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:02<00:00,  1.35it/s] 
# Train Accuracy: 0.9427, Precision: 0.9950, Recall: 0.9275, F1 Score: 0.9601
# Validation Accuracy: 0.8750, Precision: 1.0000, Recall: 0.7500, F1 Score: 0.8571
# Epoch 4/30: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:02<00:00,  1.34it/s]
# Train Accuracy: 0.9463, Precision: 0.9961, Recall: 0.9314, F1 Score: 0.9627
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 5/30: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:02<00:00,  1.34it/s] 
# Train Accuracy: 0.9482, Precision: 0.9956, Recall: 0.9345, F1 Score: 0.9641
# Validation Accuracy: 0.9375, Precision: 0.8889, Recall: 1.0000, F1 Score: 0.9412
# Epoch 6/30: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:02<00:00,  1.34it/s] 
# Train Accuracy: 0.9572, Precision: 0.9965, Recall: 0.9458, F1 Score: 0.9705
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 7/30: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:02<00:00,  1.34it/s]
# Train Accuracy: 0.9709, Precision: 0.9976, Recall: 0.9631, F1 Score: 0.9800
# Validation Accuracy: 0.9375, Precision: 0.8889, Recall: 1.0000, F1 Score: 0.9412
# Epoch 8/30: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:04<00:00,  1.33it/s] 
# Train Accuracy: 0.9705, Precision: 0.9984, Recall: 0.9618, F1 Score: 0.9798
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 9/30: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:05<00:00,  1.33it/s] 
# Train Accuracy: 0.9705, Precision: 0.9971, Recall: 0.9631, F1 Score: 0.9798
# Validation Accuracy: 0.9375, Precision: 1.0000, Recall: 0.8750, F1 Score: 0.9333
# Epoch 10/30: 100%|███████████████████████████████████████████████████████████████████| 326/326 [04:00<00:00,  1.36it/s] 
# Train Accuracy: 0.9657, Precision: 0.9970, Recall: 0.9566, F1 Score: 0.9764
# Validation Accuracy: 0.9375, Precision: 0.8889, Recall: 1.0000, F1 Score: 0.9412
# Epoch 11/30: 100%|███████████████████████████████████████████████████████████████████| 326/326 [04:03<00:00,  1.34it/s] 
# Train Accuracy: 0.9743, Precision: 0.9979, Recall: 0.9675, F1 Score: 0.9824
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 12/30: 100%|███████████████████████████████████████████████████████████████████| 326/326 [04:06<00:00,  1.32it/s]
# Train Accuracy: 0.9791, Precision: 0.9992, Recall: 0.9726, F1 Score: 0.9857
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 13/30: 100%|███████████████████████████████████████████████████████████████████| 326/326 [04:12<00:00,  1.29it/s] 
# Train Accuracy: 0.9808, Precision: 0.9987, Recall: 0.9755, F1 Score: 0.9869
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 14/30: 100%|███████████████████████████████████████████████████████████████████| 326/326 [04:06<00:00,  1.32it/s] 
# Train Accuracy: 0.9849, Precision: 0.9987, Recall: 0.9809, F1 Score: 0.9897
# Validation Accuracy: 0.9375, Precision: 0.8889, Recall: 1.0000, F1 Score: 0.9412
# Epoch 15/30: 100%|███████████████████████████████████████████████████████████████████| 326/326 [04:06<00:00,  1.32it/s] 
# Train Accuracy: 0.9849, Precision: 0.9987, Recall: 0.9809, F1 Score: 0.9897
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 16/30: 100%|███████████████████████████████████████████████████████████████████| 326/326 [04:07<00:00,  1.32it/s] 
# Train Accuracy: 0.9837, Precision: 0.9987, Recall: 0.9794, F1 Score: 0.9889
# Validation Accuracy: 0.8750, Precision: 0.8000, Recall: 1.0000, F1 Score: 0.8889
# Epoch 17/30: 100%|███████████████████████████████████████████████████████████████████| 326/326 [04:06<00:00,  1.32it/s] 
# Train Accuracy: 0.9889, Precision: 0.9992, Recall: 0.9858, F1 Score: 0.9925
# Validation Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000
# Epoch 18/30: 100%|███████████████████████████████████████████████████████████████████| 326/326 [04:07<00:00,  1.32it/s] 
# Train Accuracy: 0.9889, Precision: 0.9990, Recall: 0.9861, F1 Score: 0.9925
# Validation Accuracy: 0.8750, Precision: 0.8000, Recall: 1.0000, F1 Score: 0.8889
# Early stopping triggered
# Test Accuracy: 0.9071, Precision: 0.8825, Recall: 0.9821, F1 Score: 0.9296
# PS C:\Users\AERO\Desktop\Chest-X-Ray-Kaggle-main> 