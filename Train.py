from LivenessNet.modelFunction import train
from LivenessNet.utils import loadData,loadModel, saveModel
import os
from torch.utils.data import DataLoader

checkpoint = os.path.join("LivenessNet"
                          ,"checkpoint.pth")
output_path = os.path.join("LivenessNet","checkpoint.pth")
root = os.path.join("Data","Dataset")
trainDataset, testDataset = loadData(root, test_ratio= 0.2)

dataLoader = DataLoader(trainDataset, batch_size= 64, shuffle= True)

model =  loadModel(checkpoint)
train(model, 10, dataLoader)
saveModel(model, output_path)
print("Successfully Saved!!")