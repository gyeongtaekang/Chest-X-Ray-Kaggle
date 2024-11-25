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
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import cv2
import torchvision

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

# Grad-CAM 적용 함수 정의
def apply_gradcam(model, input_tensor, target_layer):
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Hook 저장용 변수
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_b = target_layer.register_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    output = model(input_tensor)
    output = output.squeeze()
    loss = output
    model.zero_grad()
    loss.backward()

    # Gradients와 Activations 가져오기
    gradients = gradients[0]
    activations = activations[0]

    # GAP과 Gradients 평균 계산
    weights = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
    gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
    gradcam = torch.relu(gradcam)
    gradcam = torch.nn.functional.interpolate(gradcam, size=(224, 224), mode='bilinear', align_corners=False)
    gradcam = gradcam.squeeze().cpu().data.numpy()

    # Grad-CAM을 시각화할 수 있도록 normalize
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
    return gradcam

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

# 테스트 및 결과 시각화 함수 정의
def test_model(model, test_loader):
    model.eval()
    all_test_labels = []
    all_test_preds = []

    # 특정 디렉토리 설정
    output_directory = r"C:\Users\AERO\Downloads\output_images"

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy().squeeze())
            
            # Grad-CAM을 적용하여 시각화
            target_layer = model.model._blocks[-1]  # 모델의 마지막 블록 사용
            for i in range(inputs.size(0)):
                img = inputs[i].cpu()
                label = labels[i].cpu().item()
                pred = preds[i].cpu().item()

                gradcam = apply_gradcam(model, inputs[i], target_layer)

                # 원본 이미지를 PIL로 변환
                img = F.to_pil_image(img)
                img = np.array(img)

                # Grad-CAM 결과를 이미지에 적용
                heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

                # 결과 시각화 및 저장
                plt.imshow(superimposed_img)
                title = f'Label: {int(label)}, Pred: {int(pred)}'
                plt.title(title)
                plt.axis('off')

                # 파일명 지정 및 저장
                filename = os.path.join(output_directory, f'result_img_{i}_label_{int(label)}_pred_{int(pred)}.png')
                plt.savefig(filename)
                plt.show()

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