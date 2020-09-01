import numpy as np
import os
import cv2
import pandas as pd
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#importing the libraries of pytorch
import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
# making the model architecture
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image  
from matplotlib.pyplot import imshow

class SimpleCNN(nn.Module):
    def __init__(self):

        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(256 * 1 * 1, 2) # !!!
        
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)))) # first convolutional layer then batchnorm, then activation then pooling layer.
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.avg(x)
        #print(x.shape)
        x = x.view(-1, 256 * 1 * 1)
        x = self.fc(x)
        return x

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        # image preprocessing
        trans_valid = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
        
        image = cv2.imread(filepath)
        image = cv2.resize(image, (50,50))
        #imshow(np.asarray(image))
        image = trans_valid(image)
        
        
        # loading the model
        
        train_on_gpu = torch.cuda.is_available()
        model = SimpleCNN()
    
        if train_on_gpu:
            model.cuda()
        model.load_state_dict(torch.load('model.ckpt'))
        model.eval()
        
        
        image = image.cuda()
        output = model(image.unsqueeze(0))
        _, predict = torch.max(output.data, 1)
        predict = predict.cpu()
        predict = predict.numpy().item()
        #print(predict.numpy().item())
        
        if predict == 0:
            text = " No Invasive Ductal Carcinoma(IDC) Dectected."
        
        else:
            text = " Detected Invasive Ductal Carcinoma(IDC)!! Please consult IDC specialist."
       

    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    
