from util import loadModel, loadData
from modelFunction import accuracy
import os

if __name__ == "__main__":
    # config
    restore_epochs = 10
    data_path = os.path.join("..", "Data", "Dataset")
    checkpoint_path = os.path.join("ckpt", f"{restore_epochs}_epochs.pkl")

    # load data và model
    model = loadModel(checkpoint_path)
    _, testDataLoader = loadData(data_path)
    
    # test 
    result = accuracy(model, testDataLoader)
    print(result)