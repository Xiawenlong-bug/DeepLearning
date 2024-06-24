'''
Author: Xiawenlong-bug 2473833028@qq.com
Date: 2024-06-21 16:06:35
LastEditors: Xiawenlong-bug 2473833028@qq.com
LastEditTime: 2024-06-23 22:51:06
FilePath: /deep_thoughts/LeNet5_Number/recognizer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import torch
import torch.nn as nn
import numpy as np

from sklearn import metrics
from model import LeNet5
from dataloader import load_data,show_image

class Recognizer(object):
    def __init__(self,data_path,batch_size=128,epoch=100,dropout_prob=0.,halve_conv_kernels=False):
        self.data_path=data_path
        self.batch_size=batch_size
        self.epoch=epoch
        self.dropout_prob=dropout_prob
        self.halve_conv_kernels=halve_conv_kernels
        
        self.has_cuda=torch.cuda.is_available()

        self.train_loader=None
        self.test_loader=None
        self.model=None
        self.output_dir='output/'

        name='lenet5'
        if halve_conv_kernels:
            name+='_halve_conv_kernels'
        if dropout_prob>0:
            name+='_dropout_{}'.format(dropout_prob)
        self.model_path=os.path.join(self.output_dir,name+'.mdl')

        print('<Recognizer>: [batch_size] %d, [dropout_prob] %.1f, '
              '[halve_conv_kernel] %s'
              % (batch_size, dropout_prob, halve_conv_kernels))



    def prepare(self):
        self.train_loader = load_data(
            self.data_path['data_dir'], self.data_path['train_image'],
            self.data_path['train_label'], self.batch_size)
        self.test_loader = load_data(
            self.data_path['data_dir'], self.data_path['test_image'],
            self.data_path['test_label'], self.batch_size)

    def train(self):
        self.model=LeNet5(dropout_prob=self.dropout_prob,halve_conv_kernels=self.halve_conv_kernels)

        if self.has_cuda:
            self.model=self.model.cuda()

        if os.path.exists(self.model_path):
            print('Train: model file exists, skip training.')
            print('Train: loading model state from file [%s] ...' % self.model_path)
            self.model.load_state_dict(torch.load(self.model_path,map_location='cpu'))
            #用于从状态字典中加载模型的参数（权重和偏置）。状态字典是一个简单的Python字典对象，它将每一层映射到其参数张量。
            return
        
        criterion=nn.CrossEntropyLoss(reduction='sum')
        #损失函数
        #nn.CrossEntropyLoss 是PyTorch中用于多分类问题的损失函数。它结合了nn.LogSoftmax和nn.NLLLoss在单个类中，使得计算更加高效。
        #reduction='sum': 这指定了如何减少（或聚合）批次中所有样本的损失。
        #设置为 'sum' 意味着所有样本的损失将被加在一起，返回一个单一的损失值。其他常见的选项有 'mean'（返回损失的平均值）和 'none'（返回每个样本的损失，不进行任何聚合）。

        optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(0.9,0.99))
        # self.model.parameters(): 这是一个生成器，它产生模型（self.model）中所有可训练参数的迭代器。这些参数是优化器需要调整以最小化损失函数的权重和偏置。
        # lr=1e-4: 这是学习率，一个超参数，它决定了参数更新的步长。较高的学习率可能导致训练过程不稳定，而较低的学习率可能导致训练过程非常缓慢或陷入局部最小值。
        # betas=(0.9, 0.99): 这是Adam优化器中的两个超参数，分别控制梯度的一阶矩估计和二阶矩估计的指数衰减率。这两个参数对于Adam优化器的性能至关重要，但通常不需要进行微调，除非你有明确的理由这样做。
        idx=0
        is_stop=False
        best_loss=float('inf')
        best_acc=0.
        best_batch_idx=0
        best_model_state=None

        print('Train: start training')
        self.model.train()
        #调用模型的train()方法会将模型设置为训练模式。在某些层（如BatchNorm或Dropout）中，训练模式和评估模式（通过eval()方法设置）的行为是不同的。



        for epoch in range(self.epoch):
            for i,(x,y) in enumerate(self.train_loader):#enumerate是Python的一个内置函数，它用于在迭代过程中同时获取元素的索引和值。
                idx+=1
                if self.has_cuda:
                    x=x.cuda()
                    y=y.cuda()
                optimizer.zero_grad()
                # 在开始反向传播之前，我们需要确保优化器的梯度被清零。这是因为PyTorch会累积梯度，所以如果不清零，下一个批次的梯度会与前一个批次的梯度相加，导致不正确的结果。
                out=self.model(x)
                loss=criterion(out,y)
                loss.backward()
                # 调用loss.backward()来执行反向传播，计算模型中所有参数的梯度。
                # 当你调用 loss.backward() 时，PyTorch 会从损失节点开始，反向遍历这个计算图，并使用链式法则自动计算每个参数关于损失函数的梯度。这些梯度随后被用来更新模型的参数，通常是通过优化器（如 Adam）的 step() 方法。
                optimizer.step()
                # 最后，使用优化器（这里是Adam优化器）的step()方法来根据计算出的梯度更新模型的参数。
                if idx%100==0:
                    print('Train: epoch [%d/%d], batch [%d/%d], loss %.4f' % (epoch,self.epoch,i,len(self.train_loader),loss.item()))
                    y=y.cpu()
                    y_pred=out.argmax(dim=1).cpu()
                    acc=metrics.accuracy_score(y,y_pred)
                    eval_acc,eval_loss=self.eval(self.test_loader)
                    self.model.train()
                    suffix=''
                    if eval_loss < best_loss or eval_acc > best_acc:
                        suffix = ' *'
                        best_batch_idx = idx
                        best_loss = min(best_loss, eval_loss)
                        best_acc = max(best_acc, eval_acc)
                        best_model_state = self.model.state_dict()
                    msg = 'Train [Epoch {:>3}]: \tTrain Loss: {:7.3f}\t' + \
                          'Train Acc: {:>5.2%}\t' + \
                          'Eval Loss: {:7.3f}\tEval Acc: {:>5.2%}{}'
                    print(msg.format(epoch+1, loss.item(), 
                                     acc, eval_loss, eval_acc, suffix))
                    # 如果超过连续 1000 个批次没有优化，则结束训练
                    if idx - best_batch_idx > 1000:
                        print('no optimization for more than 1000 batches, '
                              'auto stop training.')
                        is_stop = True
                        break                    
                if is_stop:
                    break
        print('Train: end training model, best loss {:.3f}, best acc {:.2%}'.
        format(best_loss, best_acc))
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        print('Train: saving model [%s] ...' % self.model_path)
        # 保存模型
        torch.save(best_model_state, self.model_path)
        # 加载模型最优状态
        self.model.load_state_dict(best_model_state)



    def eval(self,data_loader,is_test=False):
        self.model.eval()
        wrong_cnt = 0                           # 统计预测错误数量
        loss_total = 0                          # 所有 loss 之和
        labels_all = np.array([], dtype=int)    # 真实标签列表
        predict_all = np.array([], dtype=int)   # 预测标签列表
        criterion = nn.CrossEntropyLoss(reduction='sum')

        for x, y in data_loader:
            if self.has_cuda:
                x, y = x.cuda(), y.cuda()
            output = self.model(x)              # 前向传播计算预测值
            loss = criterion(output, y)         # 计算 loss
            loss_total += loss
            y_pred = output.argmax(dim=1)       # 取出预测最大下标
            labels_all = np.append(labels_all, y.cpu())
            predict_all = np.append(predict_all, y_pred.cpu())

            if is_test:
                for i in range(len(y)):
                    if y[i] != y_pred[i]:
                        wrong_cnt += 1          # 统计预测错误数量
                        #show_image(x[i].cpu(), 'label:%d, pred:%d' % (y[i], y_pred[i]))
                        print('wrong predict: label %d, predict %d'
                              % (y[i], y_pred[i]))

        loss = loss_total / len(data_loader)                    # 计算平均 loss
        acc = metrics.accuracy_score(labels_all, predict_all)   # 计算准确率

        if is_test:
            print('Test: total data %d, wrong prediction %d' %
                  (len(labels_all), wrong_cnt))

        return acc, loss
    
    def test(self):
        acc,loss=self.eval(self.test_loader,is_test=True)
        print('Test: Average Loss: {0:.3f}, Accuracy: {1:>5.2%}'
              .format(loss, acc))