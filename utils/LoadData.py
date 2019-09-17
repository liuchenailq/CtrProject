"""
加载训练集和测试集
"""
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

def load_category_index(emb_file):
    """
    加载特征取值索引文件
    :param emb_file:
    :return: [{value1:index1, ..., }, ..., {}]
    """
    cate_dict = []
    for i in range(39):  # 因为一共有39个特征
        cate_dict.append({})
    f = open(emb_file, "r")
    for line in f:
        cols = line.strip().split(",")
        cate_dict[int(cols[0])][int(cols[1])] = int(cols[2])
    return cate_dict

def read_data(file_path, emb_file):
    """
    加载数据集
    :param file_path:
    :param emb_file:
    :return: {label:[0,0,0,1], index:[[1,2,0,...,], ... , [1,2,5,2,...]], value:[[1,1,...]], feature_sizes:[]}
    """
    result = {'label':[], 'index':[], 'value':[], 'feature_sizes':[]}
    cate_dict = load_category_index(emb_file)

    # 统计每个特征有多少个不同的取值
    for item in cate_dict:
        result['feature_sizes'].append(len(item))

    f = open(file_path, "r")
    for line in f:
        cols = line.strip().split(",")
        result['label'].append(int(cols[0]))  # 0:没有点击  1：点击
        indexs = [int(item) for item in cols[1:]]
        values = [1]*39
        result['index'].append(indexs)
        result['value'].append(values)
    return result

def get_batch(Xi, Xv, y, batch_size):
    """
    批处理迭代器
    :param Xi:
    :param Xv:
    :param y:
    :return:
    """
    size = Xi.shape[0]
    batch_iter = size // batch_size
    start = 0
    end = batch_size
    for i in range(batch_iter):
        batch_xi = torch.LongTensor(Xi[start:end])
        batch_xv = torch.FloatTensor(Xv[start:end])
        batch_y = torch.FloatTensor(y[start:end])
        start = end
        end += batch_size
        yield (batch_xi, batch_xv, batch_y)
    if start < size:
        batch_xi = torch.LongTensor(Xi[start:size])
        batch_xv = torch.FloatTensor(Xv[start:size])
        batch_y = torch.FloatTensor(y[start:size])
        yield (batch_xi, batch_xv, batch_y)

def eval_metric(y, y_pred):
    return roc_auc_score(y, y_pred)