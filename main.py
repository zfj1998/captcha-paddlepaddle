'''
1. 图像与标签是否对应？
2. padding是否影响训练？
3. 卷积参数是否正确？
'''

#定义数据读取器
import cv2
import random
#训练和评估代码
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
# VGG模型代码
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable
import matplotlib.pyplot as plt
import ipdb
import logging
logging.basicConfig(filename='graph_tool.log', level=logging.INFO)
from data.file_tool import generate_image

TRAIN_DATA = 'data/train_data.npy'
TRAIN_LABEL = 'data/train_label.npy'
TEST_DATA = 'data/test_data.npy'
TEST_LABEL = 'data/test_label.npy'

INITIAL_BEST_E = 0.65 #准确度初始阈值

# 对读入的图像数据进行预处理
def transform_img(img):
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img

def show_img(img, name='test_captcha'):
    img = (img+1.0)/2
    img = np.transpose(img, (1,2,0))
    plt.imshow(img)
    # plt.savefig('{}.jpg'.format(name))
    plt.show()

# 输入格式 [N,W,H,C] [500, 60,160,3]
# 定义数据读取器
def data_loader(batch_size=10, mode = 'train'):
    def reader():
        if mode == 'train':
            data_path = TRAIN_DATA
            label_path = TRAIN_LABEL
        elif mode == 'test':
            data_path = TEST_DATA
            label_path = TEST_LABEL
        
        data = np.load(data_path)
        label = np.load(label_path)

        if mode == 'train':
            # 训练数据打乱顺序
            data = data.tolist()
            label = label.tolist()
            data_label = [[label[i], data[i]] for i in range(len(data))]
            random.shuffle(data_label)
            data = np.array([i[1] for i in data_label])
            label = np.array([i[0] for i in data_label])

        batch_imgs = []
        batch_labels = []
        for index in range(len(data)):
            img = transform_img(data[index]) # 转换为[色，宽，高]，plt可显示的是[宽，高，色]
            # show_img(img, '')
            batch_imgs.append(img)
            batch_labels.append(label[index])
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32')
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32')
            yield imgs_array, labels_array
    return reader

# 定义vgg块，包含多层卷积和1层2x2的最大池化层
class vgg_block(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_filters, filter_size, padding):
        """
        num_convs, 卷积层的数目
        num_filters, 卷积层的输出通道数，在同一个Incepition块内，卷积层输出通道数是一样的
        """
        super(vgg_block, self).__init__(name_scope)
        self.conv_list = []
        for i in range(2):
            conv_layer = self.add_sublayer('conv_' + str(i), Conv2D(self.full_name(), 
                                        num_filters=num_filters, filter_size=filter_size, padding=padding))
            batch_norm = self.add_sublayer('bn_' + str(i), BatchNorm(self.full_name(),
                                        num_channels=num_filters, act='relu'))
            self.conv_list.append(conv_layer)
            self.conv_list.append(batch_norm)
        self.pool = Pool2D(self.full_name(), pool_stride=2, pool_size=2, pool_type='max')
    
    def forward(self, x):
        for item in self.conv_list:
            x = item(x)
        return self.pool(x)

class VGG(fluid.dygraph.Layer):
    def __init__(self, name_scope, conv_arch=((32, 5, 2), 
                                (64, 3, 1), (128, 3,1 ), (256, 3, 1))):
        super(VGG, self).__init__(name_scope)
        self.vgg_blocks=[]
        iter_id = 0
        # 添加vgg_block
        # 这里一共5个vgg_block，每个block里面的卷积层数目和输出通道数由conv_arch指定
        for (num_filters, filter_size, padding) in conv_arch:
            block = self.add_sublayer('block_' + str(iter_id), 
                    vgg_block(self.full_name(), num_filters=num_filters, filter_size=filter_size, padding=padding))
            self.vgg_blocks.append(block)
            iter_id += 1
        self.fc0 = FC(self.full_name(),
                      size=512,
                      act='relu')
        self.fc1 = FC(self.full_name(),
                      size=10,
                      act='softmax')
        self.fc2 = FC(self.full_name(),
                      size=10,
                      act='softmax')
        self.fc3 = FC(self.full_name(),
                      size=10,
                      act='softmax')
        self.fc4 = FC(self.full_name(),
                      size=10,
                      act='softmax')

    def forward(self, x):
        for item in self.vgg_blocks:
            x = item(x)
        x = self.fc0(x)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return [y1, y2, y3, y4]
        # return y1

# 定义训练过程
def train(model, save_to, load_model=None, epoch_num=5):
    with fluid.dygraph.guard():
        logging.info('start training ... ')
        # 定义优化器
        opt = fluid.optimizer.Adam()
        if load_model:
            logging.info('loading model .......')
            model_state_dict, opt_state_dict = fluid.load_dygraph(load_model)
            model.load_dict(model_state_dict)
            opt.set_dict(opt_state_dict)
            logging.info('model loaded')
        best_e_result = INITIAL_BEST_E
        # 定义数据读取器，训练数据读取器和验证数据读取器
        train_loader = data_loader(mode='train')
        for epoch in range(epoch_num):
            # 切换model状态
            model.train()
            total_correct = 0
            total = 0
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                total = total + len(x_data)
                img = fluid.dygraph.to_variable(x_data)
                label = []
                row = y_data.shape[0]
                col = y_data.shape[1]
                for i in range(col):
                    label.append(y_data[:,i].reshape(row, 1))
                label = np.array(label).astype('int64')
                # 运行模型前向计算，得到预测值
                logits = model(img)
                total_correct += check_correct(logits, label)
                # 进行loss计算
                loss = 0
                for i in range(4):
                    label_v = fluid.dygraph.to_variable(label[i])
                    tmp = fluid.layers.cross_entropy(logits[i], label_v)
                    loss = tmp + loss if loss else tmp

                if batch_id % 10 == 0:
                    logging.info("epoch: {}, batch_id: {}, Accuracy is: {}, loss is: {}" \
                        .format(epoch, batch_id, total_correct/total, loss.numpy()[:,0]))
                # 反向传播，更新权重，清除梯度
                loss.backward()
                opt.minimize(loss)
                model.clear_gradients()

            # 每个epoch后测试
            e_result = evaluation(model)
            if e_result > best_e_result:
                best_e_result = e_result
                logging.info("save new best model to {}.pdopt/pdparams".format(save_to))
                # save params of model
                fluid.save_dygraph(model.state_dict(), save_to)
                # save optimizer state
                fluid.save_dygraph(opt.state_dict(), save_to)
            
            if epoch == epoch_num-1 and best_e_result == INITIAL_BEST_E:
                logging.info("not reaching accuracy {} but save model to {}.pdopt/pdparams".format(INITIAL_BEST_E, save_to))
                # save params of model
                fluid.save_dygraph(model.state_dict(), save_to)
                # save optimizer state
                fluid.save_dygraph(opt.state_dict(), save_to)

def check_correct(x, y):
    '''
    x为模型输出[4,10,10]
    y为标签[4,10,1]
    饭回四个数字全部正确的个数
    '''
    predictions = [] #[4, 10] 包含0-9
    correctness = [] #[10] 包含0/1
    for v in x:
        prediction = fluid.layers.argmax(x=v, axis=-1)
        predictions.append(prediction.numpy())
    # logging.info('{}'.format(predictions))
    # logging.info('{}'.format(y))
    for i in range(10):
        img_correctness = []
        for j in range(4):
            value = 1 if predictions[j][i]==y[j][i][0] else 0
            img_correctness.append(value)
        img_correct = 1 if sum(img_correctness)==4 else 0
        correctness.append(img_correct)
    return sum(correctness)

# 定义评估过程
def evaluation(model, params_file_path=None):
    with fluid.dygraph.guard():
        logging.info('start evaluation .......')
        #加载模型参数
        if params_file_path:
            logging.info('loading model .......')
            model_state_dict, _ = fluid.load_dygraph(params_file_path)
            model.load_dict(model_state_dict)
            logging.info('model loaded')

        model.eval()
        eval_loader = data_loader(mode='test')

        loss_set = []
        total_correct = 0
        total = 0
        for batch_id, data in enumerate(eval_loader()):
            x_data, y_data = data
            total = total + len(x_data)
            img = fluid.dygraph.to_variable(x_data)
            label = []
            row = y_data.shape[0]
            col = y_data.shape[1]
            for i in range(col):
                label.append(y_data[:,i].reshape(row, 1))
            label = np.array(label).astype('int64') #[4,10,1]
            # 计算预测和精度
            prediction = model(img) #[4,10,10]
            total_correct += check_correct(prediction, label)
            # 计算损失函数值
            loss = 0
            for i in range(4):
                label_v = fluid.dygraph.to_variable(label[i])
                tmp = fluid.layers.cross_entropy(prediction[i], label_v)
                loss = tmp + loss if loss else tmp
            loss_np = loss.numpy()[:,0]
            loss_set.append(loss_np)
            # if batch_id % 10 == 0:
            #     logging.info("batch_id: {},  Accuracy is: {}, loss is: {}".format(batch_id, total_correct/total, loss_np))
        # 求平均精度
        avg_loss = np.array(loss_set).mean(axis=0)
        logging.info('total Accuracy={} ,average loss={}'.format(total_correct/total, avg_loss))
        return total_correct/total

def user_input():
    while True:
        number = input('请输入四个数字:')
        if len(number) != 4 or not number.isdigit():
            print('非法输入')
            continue
        return number
        
# 一个个识别
def game(model, params_file_path=None):
    total = 0
    correct = 0
    print('loading model .......')
    with fluid.dygraph.guard():
        #加载模型参数
        model_state_dict, _ = fluid.load_dygraph(params_file_path)
        model.load_dict(model_state_dict)
        print('model loaded')

        model.eval()
        while True:
            number = str(random.randint(1000,9999))
            print('正确结果为:{}'.format(number))
            total = total + 1
            img = generate_image(number)
            t_img = np.array([transform_img(img)])
            show_img(t_img[0])
            v_img = fluid.dygraph.to_variable(t_img)
            prediction = model(v_img)
            predicted_num = []
            for v in prediction:
                tmp = fluid.layers.argmax(x=v, axis=-1)
                tmp = str(tmp.numpy().tolist()[0])
                predicted_num.append(tmp)
            predicted_str = ''.join(predicted_num)
            print('预测结果为:{}'.format(predicted_str))
            if predicted_str == number:
                correct = correct + 1
                print('模型预测正确!')
            else:
                print('模型预测错误..')
            print('共计{}次, 正确率: {:.3}\n'.format(total, correct/total))

with fluid.dygraph.guard():
    model = VGG("VGG")

if __name__ == "__main__":
    # 训练并保存模型到mnist
    # train(model, save_to='v2', load_model=None, epoch_num=15) #第一波训练集85AC
    # 读取mnist并测试
    # evaluation(model, 'v2')
    # 一个个识别
    game(model, 'v2')