import cv2
import numpy as np
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid
import random
import ipdb

captcha = ImageCaptcha()

def generate_image(code):
    img=captcha.generate(code)
    img=np.frombuffer(img.getvalue(),dtype='uint8')
    img=cv2.imdecode(img,cv2.IMREAD_COLOR)
    img=cv2.resize(img,(80,32))
    return img #[W,H,C] [32,80,3]

def random_numbers():
    return random.randint(1000,9999)

def gen_dataset(train_size, test_size):
    
    if train_size:
        train_data=[]
        train_label=[]
        for i in range(train_size):
            print('generate train image:%s/%s' % (i+1,train_size))
            num=random_numbers()
            code=str(num)
            img=generate_image(code)
            train_data.append(img)
            label=[int(j) for j in code]
            train_label.append(label)
        train_label=np.array(train_label)
        train_data=np.array(train_data)
        np.save('train_data',train_data)
        np.save('train_label',train_label)
            
        del train_label
        del train_data

    if test_size:
        test_label=[]
        test_data=[]

        for i in range(test_size):
            print('generate test image: %s/%s' % (i+1,test_size))
            num=random_numbers()
            code=str(num)
            img=generate_image(code)
            test_data.append(img)
            label=[int(j) for j in code]
            test_label.append(label)
        test_label=np.array(test_label)
        test_data=np.array(test_data)
        np.save('test_data',test_data)
        np.save('test_label',test_label)

if __name__ == "__main__":
    gen_dataset(10000,1000)