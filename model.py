# Import thư viện
import torch
import torch.nn as nn

# Định nghĩa lớp mô hình, lớp mô hình phải được kế thừa từ lớp cha nn.Module
class LivenessNet( nn.Module ):
    
    # Xây dựng phương thức khởi tạo
    # Trong phương thức khởi tạo thì xây dựng kiến trúc mạng
    def __init__(self):
        # Gọi phương thức khởi tạo của lớp cha
        super(LivenessNet, self).__init__()

        # Xây dựng kiến trúc mạng
        self.network = nn.Sequential(
            # Đầu vào 64x64x3
            nn.Conv2d(
                in_channels=3, 
                out_channels=32, 
                kernel_size=3,
                padding="same"
                ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # đầu ra 64x64x32
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # đầu ra là 32x32x32
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding="same"
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # đầu ra là 32x32x64
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # đầu ra là 16x16x64
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding="same"
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # đầu ra 16x16x128
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            #đầu ra 8x8x128
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding="same"
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #đầu ra 8x8x128
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            # đầu ra 4x4x128
            nn.Flatten(),
            nn.Linear(
                in_features=2048,
                out_features=256,
            ),
            # đầu ra 256
            nn.Dropout(),
            nn.Linear(
                in_features=256,
                out_features=2
            ),
            nn.Sigmoid()   
        )

    def forward(self, x):
        return self.network(x)