'''
Author: Xiawenlong-bug 2473833028@qq.com
Date: 2024-06-21 16:06:35
LastEditors: Xiawenlong-bug 2473833028@qq.com
LastEditTime: 2024-06-21 18:53:23
FilePath: /deep_thoughts/LeNet5_Number/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys#sys是什么库



from recognizer import Recognizer


def main():
    data_path = {
        'data_dir': 'data/',
        'train_image': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_image': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz',
    }
    dropout_prob = 0.3
    halve_conv_kernels = False

    # 初始化手写数字识别器
    recognizer = Recognizer(data_path=data_path,
                            dropout_prob=dropout_prob,
                            halve_conv_kernels=halve_conv_kernels)
    # 准备数据集
    recognizer.prepare()
    # 模型训练
    recognizer.train()
    # 模型测试
    recognizer.test()


if __name__ == '__main__':
    sys.exit(main())
