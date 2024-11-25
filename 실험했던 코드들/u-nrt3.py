import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from segmentation_models_pytorch import Unet
from efficientnet_pytorch import EfficientNet
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2
from tqdm import tqdm

# GPU 설정
if not torch.cuda.is_available():
    raise RuntimeError("CUDA가 사용 불가능합니다. GPU를 활성화하거나 적절한 환경을 설정하세요.")
device = torch.device("cuda")

# 데이터셋 경로 설정
data_dir = r"C:\Users\AERO\Downloads\archive (1)\chest_xray"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# CLAHE 적용
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# Laplacian Filtering과 Edge Detection
def apply_laplacian_and_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 100, 200)
    return laplacian, edges

# GLCM 텍스처 분석
def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy]

# 좌우 비대칭성 계산
def calculate_asymmetry(image):
    h, w = image.shape[:2]
    left = image[:, :w//2]
    right = image[:, w//2:]
    left_mean = np.mean(left)
    right_mean = np.mean(right)
    return abs(left_mean - right_mean)

# 전처리 과정
def custom_preprocessing(image):
    if isinstance(image, torch.Tensor):
        image = np.array(image.permute(1, 2, 0).cpu())  # Tensor -> numpy
    else:
        image = np.array(image)

    # CLAHE 적용
    enhanced = apply_clahe(image)

    # Laplacian Filtering과 Edge Detection
    laplacian, edges = apply_laplacian_and_edge_detection(image)

    # GLCM 텍스처 분석
    texture_features = extract_texture_features(image)

    # 좌우 비대칭성 계산
    asymmetry = calculate_asymmetry(image)

    # 텍스처와 비대칭성을 추가적으로 모델 입력으로 활용할 수도 있음
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

# 데이터 전처리
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: custom_preprocessing(x)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: custom_preprocessing(x)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터셋 및 로더
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = ImageFolder(root=test_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# U-Net 모델 정의
class LungSegmentationModel(nn.Module):
    def __init__(self):
        super(LungSegmentationModel, self).__init__()
        self.unet = Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation=None)
    
    def forward(self, x):
        return torch.sigmoid(self.unet(x))

# EfficientNet 분류 모델 정의
class PneumoniaClassificationModel(nn.Module):
    def __init__(self):
        super(PneumoniaClassificationModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)

    def forward(self, x):
        return self.efficientnet(x)

# 모델 초기화
lung_model = LungSegmentationModel().to(device)
pneumonia_model = PneumoniaClassificationModel().to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.BCELoss()
optimizer = optim.Adam([
    {'params': lung_model.parameters(), 'lr': 0.0001},
    {'params': pneumonia_model.parameters(), 'lr': 0.0001}
])

# 학습 함수
def train_model(lung_model, pneumonia_model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        lung_model.train()
        pneumonia_model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # 1. 폐 영역 마스크 생성
            lung_masks = lung_model(inputs)
            masked_inputs = inputs * lung_masks
            
            # 2. 분류 모델로 예측
            outputs = pneumonia_model(masked_inputs).squeeze()
            loss = criterion(torch.sigmoid(outputs), labels)
            
            # 3. 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

# 테스트 함수
def test_model(lung_model, pneumonia_model, test_loader):
    lung_model.eval()
    pneumonia_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # 폐 영역 마스크 생성
            lung_masks = lung_model(inputs)
            masked_inputs = inputs * lung_masks
            
            # 분류 모델로 예측
            outputs = pneumonia_model(masked_inputs).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

# 모델 학습 및 테스트 실행
train_model(lung_model, pneumonia_model, train_loader, val_loader, criterion, optimizer, epochs=10)
test_model(lung_model, pneumonia_model, test_loader)


# Loaded pretrained weights for efficientnet-b0
# Epoch 1/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [02:51<00:00,  1.90it/s]
# Epoch 1/10, Loss: 82.6066, Accuracy: 0.9013
# Epoch 2/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [02:55<00:00,  1.85it/s] 
# Epoch 2/10, Loss: 43.1845, Accuracy: 0.9532
# Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [03:13<00:00,  1.68it/s]
# Epoch 3/10, Loss: 30.2344, Accuracy: 0.9659
# Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [03:08<00:00,  1.73it/s] 
# Epoch 4/10, Loss: 29.0576, Accuracy: 0.9684
# Epoch 5/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [02:55<00:00,  1.86it/s] 
# Epoch 5/10, Loss: 22.9505, Accuracy: 0.9747
# Epoch 6/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [02:53<00:00,  1.88it/s] 
# Epoch 6/10, Loss: 22.0630, Accuracy: 0.9732
# Epoch 7/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [02:49<00:00,  1.92it/s] 
# Epoch 7/10, Loss: 21.6176, Accuracy: 0.9753
# Epoch 8/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [02:49<00:00,  1.92it/s] 
# Epoch 8/10, Loss: 16.9907, Accuracy: 0.9810
# Epoch 9/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [02:48<00:00,  1.93it/s] 
# Epoch 9/10, Loss: 16.9465, Accuracy: 0.9806
# Epoch 10/10: 100%|███████████████████████████████████████████████████████████████████| 326/326 [02:48<00:00,  1.93it/s] 
# Epoch 10/10, Loss: 14.6814, Accuracy: 0.9829
# Test Accuracy: 0.8958, Precision: 0.8603, Recall: 0.9949, F1 Score: 0.9227