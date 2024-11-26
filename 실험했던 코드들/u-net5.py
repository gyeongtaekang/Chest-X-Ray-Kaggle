import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from segmentation_models_pytorch import Unet
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA가 사용 불가능합니다. GPU를 활성화하거나 적절한 환경을 설정하세요.")

# 데이터셋 경로 설정
data_dir = r"C:\Users\AERO\Downloads\archive (1)\chest_xray"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# CLAHE 적용
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

# Laplacian Filtering과 Edge Detection
def apply_laplacian_and_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 100, 200)
    return laplacian, edges

# 텍스처 분석
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
    left = image[:, :w // 2]
    right = image[:, w // 2:]
    return abs(np.mean(left) - np.mean(right))

# 커스텀 전처리 함수
def custom_preprocessing(image):
    if isinstance(image, torch.Tensor):
        image = np.array(F.to_pil_image(image))  # 텐서를 PIL 이미지로 변환
    else:
        image = np.array(image)  # 이미 PIL 이미지인 경우 변환하지 않음

    enhanced = apply_clahe(image)
    image = torch.from_numpy(enhanced).unsqueeze(0).float() / 255.0  # numpy 배열을 텐서로 변환하고 차원 추가 및 float 형식으로 변환

    # 1채널 이미지를 3채널로 변환
    image = image.repeat(3, 1, 1)
    return image

# 데이터 전처리
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: custom_preprocessing(x)),
    transforms.Normalize([0.485], [0.229])  # 단일 채널에 대한 정규화
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: custom_preprocessing(x)),
    transforms.Normalize([0.485], [0.229])  # 단일 채널에 대한 정규화
])

# 데이터셋 및 로더
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = ImageFolder(root=test_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

# U-Net 모델 정의
class LungSegmentationModel(nn.Module):
    def __init__(self):
        super(LungSegmentationModel, self).__init__()
        self.unet = Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet", classes=1, activation=None)
    
    def forward(self, x):
        return torch.sigmoid(self.unet(x))

# EfficientNet 분류 모델 정의
class PneumoniaClassificationModel(nn.Module):
    def __init__(self):
        super(PneumoniaClassificationModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)

    def forward(self, x):
        return self.efficientnet(x)

# 모델 초기화
lung_model = LungSegmentationModel().to(device)
pneumonia_model = PneumoniaClassificationModel().to(device)

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.25, gamma=2)

# 옵티마이저와 스케줄러 정의
optimizer = optim.Adam([
    {'params': lung_model.parameters(), 'lr': 0.0001},
    {'params': pneumonia_model.parameters(), 'lr': 0.0001}
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# 학습 함수
def train_model(lung_model, pneumonia_model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10):
    for epoch in range(epochs):
        lung_model.train()
        pneumonia_model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            lung_masks = lung_model(inputs)
            masked_inputs = inputs * lung_masks
            
            outputs = pneumonia_model(masked_inputs).squeeze()
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        scheduler.step()
        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        torch.cuda.empty_cache()

# 테스트 함수
def test_model(lung_model, pneumonia_model, test_loader):
    lung_model.eval()
    pneumonia_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            lung_masks = lung_model(inputs)
            masked_inputs = inputs * lung_masks
            
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
train_model(lung_model, pneumonia_model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10)
test_model(lung_model, pneumonia_model, test_loader)



# Loaded pretrained weights for efficientnet-b4
# Epoch 1/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [03:34<00:00,  1.52it/s]
# Epoch 1/10, Loss: 3.1621, Accuracy: 0.9415
# Epoch 2/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [03:08<00:00,  1.73it/s]
# Epoch 2/10, Loss: 1.0910, Accuracy: 0.9833
# Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [03:07<00:00,  1.74it/s]
# Epoch 3/10, Loss: 0.4584, Accuracy: 0.9929
# Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [03:07<00:00,  1.74it/s]
# 0, 94.93s/it]
# Epoch 6/10, Loss: 0.0882, Accuracy: 0.9987
# Epoch 7/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [04:48<00:00,  1.13it/s] 
# Epoch 7/10, Loss: 0.0752, Accuracy: 0.9990
# Epoch 8/10:  33%|██████████████████████▋                                             | 109/326 [01:37<03:2Epoch 8/10:  34%|██████████████████████▉                                             | 110/326 [01:38<03:4Epoch 8/10:  34%|███████████Epoch 8/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [05:11<00:00,  1.05it/s]
# Epoch 8/10, Loss: 0.0305, Accuracy: 0.9998
# Epoch 9/10: 100%|████████████████████████████████████████████████████████████████████| 326/326 [05:19<00:00,  1.02it/s]
# Epoch 9/10, Loss: 0.0272, Accuracy: 0.9996
# Epoch 10/10: 100%|███████████████████████████████████████████████████████████████████| 326/326 [06:22<00:00,  1.17s/it]
# Epoch 10/10, Loss: 0.0222, Accuracy: 0.9998
# Test Accuracy: 0.8429, Precision: 0.8017, Recall: 0.9949, F1 Score: 0.8879
# PS C:\Users\AERO\Desktop\Chest-X-Ray-Kaggle-main> 



