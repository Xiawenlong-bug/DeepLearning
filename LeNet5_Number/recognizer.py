'''
Author: Xiawenlong-bug 2473833028@qq.com
Date: 2024-06-21 16:06:35
LastEditors: Xiawenlong-bug 2473833028@qq.com
LastEditTime: 2024-06-21 18:52:42
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
            return
        
        criterion=nn.CrossEntropyLoss(reduction='sum')
        optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(0.9,0.99))

        idx=0
        is_stop=False
        best_loss=float('inf')
        best_acc=0.
        best_batch_idx=0
        best_model_state=None

        print('Train: start training')
        self.model.train()

        for epoch in range(self.epoch):
            for i,(x,y) in enumerate(self.train_loader):#enumerate是Python的一个内置函数，它用于在迭代过程中同时获取元素的索引和值。
                idx+=1
                if self.has_cuda:
                    x=x.cuda()
                    y=y.cuda()
                optimizer.zero_grad()
                out=self.model(x)
                loss=criterion(out,y)
                loss.backward()
                optimizer.step()

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