import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from model import LivenessNet
import torch

def transformData(image):
    transformCallback = transforms.Compose(
        [
            transforms.Resize((64,64)), transforms.ToTensor()
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return transformCallback(image)

'''
Hàm load data từ đường dẫn và tạo các Dataset từ data

Chỉ dùng cho lúc huấn luyệnh
'''
def loadData(root, test_ratio = 0.1):
    # Load dữ liệu ảnh từ đường dẫn root
    print("Start loading data ...")
    dataset = ImageFolder(root, transformData)
    

    # Chia train, test
    indices = list(range(len(dataset)))
    # Lấy ra danh sách các nhãn của dataset theo thứ tự
    labels = [dataset[i][1] for i in indices]
    train_idx, test_idx = train_test_split(indices, test_size= test_ratio, stratify=labels)

    # Chia tập data
    trainDataset = Subset(dataset, train_idx)
    testDataset = Subset(dataset, test_idx)
    
    # Tạo dataloader
    trainDataLoader = DataLoader(trainDataset, batch_size = 8, shuffle = False)
    testDataLoader = DataLoader(testDataset, batch_size = 8, shuffle = False)
    print(f"Train : {len(trainDataLoader)}")
    print(f"Test : {len(testDataLoader)}")
    print(f"Data has been loaded successfully!!")
    return trainDataLoader, testDataLoader


def loadModel(checkpoint_path):
    model = LivenessNet()
    checkpoints = torch.load(checkpoint_path)
    model.load_state_dict(checkpoints)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def saveModel(model, output_path):
    torch.save(
        model.state_dict(),
        output_path
    )