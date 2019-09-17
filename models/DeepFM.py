"""
Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DeepFM(nn.Module):
    def __init__(self, field_size, feature_sizes, embedding_size=4, h_depth=2, deep_layers=[32, 32], learning_rate=0.003,
                 weight_decay=0.0, n_epochs=64, batch_size=256, is_shallow_dropout=True, dropout_shallow=[0.5, 0.5],
                 is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5]):
        """
        :param field_size: 特征数目
        :param feature_sizes: 每个特征的取值数目
        :param embedding_size: 嵌入的维度
        :param h_depth: deep部分中全连接层的隐藏层层数
        :param deep_layers: deep部分中全连接层每层隐藏层的神经元个数  长度为h_depth的整数列表
        :param learning_rate: 学习率
        :param weight_decay
        :param n_epochs: 迭代次数
        :param batch_size: 批处理大小
        :param is_shallow_dropout: fm部分是否使用dropout
        :param dropout_shallow: fm部分dropout的比例  长度为2的列表  分别作用于 一阶特征 二阶特征交互
        :param is_deep_dropout: deep部分是否使用dropout
        :param dropout_deep: deep部分dropout的比例  长度为3的列表 分别作用于 embedding输出层  第一层输出   第二层输出
        """
        super(DeepFM, self).__init__()

        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        """fm 部分"""
        self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])  # 一阶独立参数 W
        self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])  # 二阶嵌入参数 V
        if self.is_shallow_dropout and self.dropout_shallow:
            self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])

        self.bias = torch.nn.Parameter(torch.randn(1))

        """deep 部分"""
        self.linear_1 = nn.Linear(self.field_size * self.embedding_size, deep_layers[0])
        if self.is_deep_dropout:
            self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])  # emd层的输出使用dropout
            self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])  # 第一层连接层的输出使用dropout
        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'linear_' + str(i+1), nn.Linear(self.deep_layers[i-1], self.deep_layers[i]))
            if self.is_deep_dropout:
                setattr(self, 'linear_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_deep[i + 1]))


    def forward(self, Xi, Xv):
        """
        :param Xi: (batch_size, field_size, 1)  取值的索引
        :param Xv: (batch_size, field_size)     1
        :return:
        """

        """fm 部分"""
        """详情参考：https://blog.csdn.net/songbinxu/article/details/80151814"""
        """一阶求和"""
        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]  # 每个特征的w参数
        # print(len(fm_first_order_emb_arr))   # 39
        # print(fm_first_order_emb_arr[0].size())  # [256, 1]
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)   # 每个特征的wi*xi
        if self.is_shallow_dropout:
            fm_first_order = self.fm_first_order_dropout(fm_first_order)
        # print(fm_first_order.size())  # [256, 39]
        """二阶交互求和"""
        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)]  # 每个特征的V参数
        # print(len(fm_second_order_emb_arr))  #  39
        # print(fm_second_order_emb_arr[0].size()) # [256, 4]
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)  # 不同特征的V相加
        # print(fm_sum_second_order_emb.size())  # [256, 4]
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        # print(fm_second_order.size())  # [256, 4]
        if self.is_shallow_dropout:
            fm_second_order = self.fm_second_order_dropout(fm_second_order)

        """deep 部分"""
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        if self.is_deep_dropout:
            deep_emb = self.linear_0_dropout(deep_emb)

        # print(deep_emb.size())  # [256, 156]
        activation = F.relu
        x_deep = self.linear_1(deep_emb)
        x_deep = activation(x_deep)
        if self.is_deep_dropout:
            x_deep = self.linear_1_dropout(x_deep)
        for i in range(1, len(self.deep_layers)):
            x_deep = getattr(self, 'linear_'+str(i+1))(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)

        """合并"""
        total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(x_deep,1) + self.bias
        return total_sum

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None, y_valid=None, stop_step=3):
        """
        训练
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                          indi_j 是训练样本i的第j个特征的取值索引
        :param Xv_train:
        :param y_train: 样本的标签列表
        :param Xi_valid:
        :param Xv_valid:
        :param y_valid:
        :param stop_step: 在验证集上超过stop_step个epoch没有提升则早停
        :return:
        """
        Xi_train = np.array(Xi_train).reshape((-1, self.field_size, 1))  # (458044, 39, 1)
        Xv_train = np.array(Xv_train)  # (458044, 39)
        y_train = np.array(y_train)  # (458044,)
        x_size = Xi_train.shape[0]  # 458044 训练集大小
        if Xi_valid:
            Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size, 1))
            Xv_valid = np.array(Xv_valid)
            y_valid = np.array(y_valid)
            x_valid_size = Xi_valid.shape[0]  # 测试集大小

        model = self.train()
        model.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = F.binary_cross_entropy_with_logits

        best_metric = 0
        eval_count = 0   # 评价的次数
        last_improve_step = 0  # 记录最近分数提升的评价时刻
        stop_flag = False
        loss_history = []
        metric_history = []

        for epoch in range(self.n_epochs):
            if stop_flag:
                break
            for batch_index, batch in enumerate(get_batch(Xi_train, Xv_train, y_train, self.batch_size)):
                batch = (t.to(self.device) for t in batch)
                batch_xi, batch_xv, batch_y = batch
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # 结束一个epoch的训练进行预测
            loss, metric = self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
            eval_count += 1
            loss_history.append(loss)
            metric_history.append(metric)
            print('loss: %.6f   metric: %.6f' % (loss, metric))
            if metric > best_metric:
                last_improve_step = eval_count
                best_metric = metric
            if eval_count - last_improve_step > stop_step:
                stop_flag = True
                break

        # 结束训练，绘制验证集上的loss曲线和metric曲线
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        plt.title("loss")
        ax1.plot(range(1, len(loss_history) + 1), loss_history)
        ax2 = fig.add_subplot(122)
        plt.title("metric")
        ax2.plot(range(1, len(metric_history) + 1), metric_history)
        plt.show()

    def eval_by_batch(self, Xi, Xv, y, size):
        """评价"""
        y_pred = []
        total_loss = 0.0
        batch_size = 16384
        model = self.eval()
        criterion = F.binary_cross_entropy_with_logits
        for batch_index, batch in enumerate(get_batch(Xi, Xv, y, batch_size)):
            batch = (t.to(self.device) for t in batch)
            batch_xi, batch_xv, batch_y = batch
            outputs = model(batch_xi, batch_xv)
            pred = torch.sigmoid(outputs).cpu()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_xi.size(0)
            y_pred.extend(pred.data.numpy())
        return total_loss / size, eval_metric(y, y_pred)


from utils.LoadData import *
train_dict = read_data("..//data//tiny_train_input.csv", "..///data/category_emb.csv")
test_dict = read_data('../data/tiny_test_input.csv', '../data/category_emb.csv')
deepfm = DeepFM(39, train_dict['feature_sizes'])
deepfm.fit(train_dict['index'], train_dict['value'], train_dict['label'], test_dict['index'], test_dict['value'], test_dict['label'])