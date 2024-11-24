import tensorflow as tf
from tensorflow.keras import layers, Model

# 사용자 정의 UNet 모델 설계를 위한 서브클래싱 방식
class UNet(Model):
    def __init__(self):
        super(UNet, self).__init__()
        
        # 인코더 부분 (Downsampling)
        self.enc_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.enc_pool1 = layers.MaxPooling2D((2, 2))
        
        self.enc_conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.enc_pool2 = layers.MaxPooling2D((2, 2))
        
        self.enc_conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.enc_pool3 = layers.MaxPooling2D((2, 2))
        
        self.enc_conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.enc_pool4 = layers.MaxPooling2D((2, 2))
        
        # Bottleneck 부분
        self.bottleneck = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')
        
        # 디코더 부분 (Upsampling)
        self.upconv4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
        self.dec_conv4a = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.dec_conv4b = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        
        self.upconv3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        self.dec_conv3a = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.dec_conv3b = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        
        self.upconv2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        self.dec_conv2a = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.dec_conv2b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        
        self.upconv1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.dec_conv1a = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.dec_conv1b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        
        # 출력 레이어
        self.output_layer = layers.Conv2D(1, (1, 1), activation='sigmoid')
        
    def call(self, inputs):
        # 인코더 부분 (다운샘플링)
        c1 = self.enc_conv1(inputs)  # 첫 번째 컨볼루션 블록
        p1 = self.enc_pool1(c1)      # 첫 번째 풀링 레이어
        
        c2 = self.enc_conv2(p1)      # 두 번째 컨볼루션 블록
        p2 = self.enc_pool2(c2)      # 두 번째 풀링 레이어
        
        c3 = self.enc_conv3(p2)      # 세 번째 컨볼루션 블록
        p3 = self.enc_pool3(c3)      # 세 번째 풀링 레이어
        
        c4 = self.enc_conv4(p3)      # 네 번째 컨볼루션 블록
        p4 = self.enc_pool4(c4)      # 네 번째 풀링 레이어
        
        # Bottleneck 부분
        bn = self.bottleneck(p4)     # 가장 깊은 부분에서 특징 추출
        
        # 디코더 부분 (업샘플링)
        u4 = self.upconv4(bn)        # 업샘플링
        u4 = layers.concatenate([u4, c4])  # 인코더의 출력과 결합 (skip connection)
        c4a = self.dec_conv4a(u4)    # 첫 번째 디코더 컨볼루션 블록
        c4b = self.dec_conv4b(c4a)   # 두 번째 디코더 컨볼루션 블록
        
        u3 = self.upconv3(c4b)       # 업샘플링
        u3 = layers.concatenate([u3, c3])  # 인코더의 출력과 결합 (skip connection)
        c3a = self.dec_conv3a(u3)    # 첫 번째 디코더 컨볼루션 블록
        c3b = self.dec_conv3b(c3a)   # 두 번째 디코더 컨볼루션 블록
        
        u2 = self.upconv2(c3b)       # 업샘플링
        u2 = layers.concatenate([u2, c2])  # 인코더의 출력과 결합 (skip connection)
        c2a = self.dec_conv2a(u2)    # 첫 번째 디코더 컨볼루션 블록
        c2b = self.dec_conv2b(c2a)   # 두 번째 디코더 컨볼루션 블록
        
        u1 = self.upconv1(c2b)       # 업샘플링
        u1 = layers.concatenate([u1, c1])  # 인코더의 출력과 결합 (skip connection)
        c1a = self.dec_conv1a(u1)    # 첫 번째 디코더 컨볼루션 블록
        c1b = self.dec_conv1b(c1a)   # 두 번째 디코더 컨볼루션 블록
        
        # 출력 레이어
        outputs = self.output_layer(c1b)  # 최종 출력 레이어
        
        return outputs

# 모델 생성
input_shape = (128, 128, 1)  # 입력 이미지 크기
unet_model = UNet()

# 모델 빌드
inputs = tf.keras.Input(shape=input_shape)
outputs = unet_model(inputs)
model = Model(inputs, outputs)

# 모델 요약 출력
model.summary()

# 클래스 기반 구현의 장점:
# 1. 코드 재사용성 증가: 서브클래싱을 통해 여러 메서드를 정의하고 재사용할 수 있어 코드 유지보수가 용이함.
# 2. 유연성: 복잡한 모델 구조를 구현할 때, 인코더와 디코더 부분을 독립적으로 관리하고 skip connection 등을 쉽게 추가할 수 있음.
# 3. 명확한 구조: 모델의 각 부분을 메서드로 나누어 관리하므로, 구조가 명확해지고 가독성이 향상됨.