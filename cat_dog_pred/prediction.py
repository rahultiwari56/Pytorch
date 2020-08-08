import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

labels = ['cat', 'dog']

class Inference:
    def __init__(self):
        self.model = models.resnet101(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc=nn.Linear(2048,2)

        # since my system doesnn't have gpu that's why i had to specify map_location as 'cpu'
        self.model.load_state_dict(torch.load('model.pth', map_location='cpu'))
        self.model.eval()

    def img_preprocessing(self,img):
        
        transform_img = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
                                        ])

        img_tensor = transform_img(img)
        reshaped_img = img_tensor.reshape(1, 3,224,224)

        return reshaped_img

    def predict(self,img):
        img = Image.open(img)
        img_data = self.img_preprocessing(img)
        output = self.model(img_data.float())
        output = output.argmax()
        output = labels[output]

        return output

if __name__ == "__main__":

    # img = 'cat1.jpg'
    img = 'dog1.jpeg'

    prediction = Inference()
    prediction_res = prediction.predict(img)

    print(prediction_res)