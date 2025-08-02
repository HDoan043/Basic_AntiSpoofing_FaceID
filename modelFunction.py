import torch
from utils import loadModel, saveModel
from model import LivenessNet
from torch.utils.data import DataLoader
import torch.optim as optim
import os

def train(model, epochs, dataloader, restore_epochs, save_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    lr = 0.0004
    optimizer = optim.Adam(model.parameters(), lr)
    os.makedirs("ckpt", exist_ok=True)
    for i in range(epochs):
        print()
        print(f"----------------EPOCH: {i}---------------- ")
        mse_loss = 0
        index = 1
        l = len(dataloader)
        for image, label in dataloader:
            print(f"\rProcessing {index}/{l} images...", end= "")
            image = image.to(device)
            label = label.to(device)

            # Feedforward
            output = model(image)
            # Tính toán hàm mất mát
            loss = loss_function(output, label)

            mse_loss += torch.square(loss)
            # Lan truyền gradient
            # Xóa gradient về 0
            optimizer.zero_grad()
            # Lan truyền gradient
            loss.backward()
            # Cập nhật trọng số
            optimizer.step()
            index += 1
        if (i+1) % save_epochs == 0:
            saveModel(model, os.path.join("ckpt", f"{restore_epochs + i}_epochs.pkl"))
        print()
        print(f"Loss: {torch.sqrt(mse_loss/l)}")

def predict(model, image):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(image.shape)
    transpose_image = image.transpose(2,0,1)
    print(transpose_image.shape)

    if type(image) != torch.Tensor:
        image = torch.tensor(image)
    
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        predict = torch.argmax(output, dim = 1)
    return predict

'''
    Hàm tính độ chính xác

    Nhận vào testDataset là một Dataset
'''
def accuracy(model, dataLoader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    trueLabel = 0
    totalSample = 0

    with torch.no_grad():
        for image, label in dataLoader:
            # Di chuyển các mẫu lên thiết bị tính toán( là GPU nếu GPU sẵn sàng, không thì là CPU)
            image = image.to(device)
            label = label.to(device)

            output = model(image)# output có kích thước 8 x 2
            predict_result = torch.argmax(output, dim = 1) 

            number_true_label = torch.sum(predict_result == label)
            trueLabel +=  number_true_label
            totalSample += len(image)

    return (trueLabel/totalSample).item()
