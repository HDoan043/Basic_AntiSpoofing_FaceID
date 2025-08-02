from model import LivenessNet
from util import loadData, loadModel
from modelFunction import train, accuracy
import os

if __name__ == "__main__":
    # config
    data_path = os.path.join("..", "Data", "Dataset")
    trainDataLoader, _ = loadData(data_path)

    restore_epochs = 0
    if restore_epochs == 0:
        model = LivenessNet()
    else:
        model = loadModel(os.path.join("ckpt", f"{restore_epochs}_epochs.pkl"))

    # bắt đầu train
    epochs = 20
    train(model, epochs, trainDataLoader, restore_epochs)
    print("Finish training !!!")