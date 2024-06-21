'''
Author: Xiawenlong-bug 2473833028@qq.com
Date: 2024-06-21 16:06:35
LastEditors: Xiawenlong-bug 2473833028@qq.com
LastEditTime: 2024-06-21 18:52:58
FilePath: /deep_thoughts/LeNet5_Number/model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch.nn as nn

from abc import ABC

class LeNet5(nn.Module,ABC):
    def __init__(self,dropout_prob=0.,halve_conv_kernels=False):
        super(LeNet5,self).__init__()#实例化对象self调用LeNet5（父类）的实例化方法
        self.need_dropout=dropout_prob>0
        if halve_conv_kernels:
            print('TODO')
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.pool1=nn.MaxPool2d((2,2))
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.pool2=nn.MaxPool2d((2,2))
        self.conv3=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)
        self.fc1=nn.Linear(in_features=120,out_features=84)
        self.fc2=nn.Linear(in_features=84,out_features=10)

        self.dropout=nn.Dropout(p=dropout_prob)

    def forward(self,x):
        
        x=x.unsqueeze(1)
        feature_map=self.conv1(x)
        feature_map=self.pool1(feature_map)
        feature_map=self.conv2(feature_map)
        feature_map=self.pool2(feature_map)
        feature_map=self.conv3(feature_map)
        feature_map=feature_map.view(feature_map.size(0),-1)
        out=self.fc1(feature_map)
        if self.need_dropout:
            out=self.dropout(out)
        out=self.fc2(out)

        return out
