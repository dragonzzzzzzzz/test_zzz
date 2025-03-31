import copy  # 导入copy模块，用于深拷贝
import math  # 导入math模块，提供数学运算函数
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能模块，包含常用函数
from torch.autograd import Variable  # 导入Variable类，用于处理需要梯度的张量

'''
num_epochs = 1 # 训练的轮数
d_model = 128  # 编码器的维度
heads = 4  # 多头注意力的头数
N = 2  # 编码器层的数量
m = 14  # 特征数量
'''


class Transformer(nn.Module):  # 定义Transformer类，继承自nn.Module
    def __init__(self, m, d_model, N, heads, dropout):  # 初始化方法，接受五个参数
        super().__init__()  # 调用父类的初始化方法
        self.gating = Gating(d_model, m)  # 初始化Gating层
        self.encoder = Encoder(d_model, N, heads, m, dropout)  # 初始化Encoder层
        self.out = nn.Linear(d_model*26*26, 1)  # 定义线性输出层

    def forward(self, src, t):  # 定义前向传播方法，接受输入src和时间步t
        e_i = self.gating(src)  # 通过Gating层处理输入
        e_outputs = self.encoder(e_i, t)  # 通过Encoder层处理Gating层的输出
        output = self.out(torch.flatten(e_outputs))  # 通过线性层获得输出

        return output.reshape(1)  # 将输出调整为1维


class Gating(nn.Module):  # 定义Gating模块
    def __init__(self, d_model, m):  # 初始化方法，接受模型维度和特征数量
        super().__init__()  # 调用父类的初始化方法
        self.m = m  # 保存特征数量

        # 重置门 r_i 的参数
        self.W_r = nn.Parameter(torch.Tensor(m, m))  # 权重矩阵W_r
        self.V_r = nn.Parameter(torch.Tensor(m, m))  # 权重矩阵V_r
        self.b_r = nn.Parameter(torch.Tensor(m))  # 偏置b_r

        # 更新门 u_i 的参数
        self.W_u = nn.Parameter(torch.Tensor(m, m))  # 权重矩阵W_u
        self.V_u = nn.Parameter(torch.Tensor(m, m))  # 权重矩阵V_u
        self.b_u = nn.Parameter(torch.Tensor(m))  # 偏置b_u

        # 输出的参数
        self.W_e = nn.Parameter(torch.Tensor(m, d_model))  # 权重矩阵W_e
        self.b_e = nn.Parameter(torch.Tensor(d_model))  # 偏置b_e

        self.init_weights()  # 初始化权重

        # 定义卷积层
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(26, 26, kernel_size=3, stride=1,padding=1),  # 单通道卷积
        )



    def init_weights(self):  # 权重初始化方法
        stdv = 1.0 / math.sqrt(self.m)  # 计算标准差
        for weight in self.parameters():  # 遍历所有权重
            weight.data.uniform_(-stdv, stdv)  # 使用均匀分布初始化权重

    def forward(self, x):  # 定义前向传播方法，接受输入x
        x_i = x[:, :, 1:2, :]  # 仅对当前行应用门控机制
        h_i = self.cnn_layers(x.squeeze(0))  # 通过卷积层获取输出

        # 计算重置门r_i
        r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
        # 计算更新门u_i
        u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)

        # 门控机制的输出
        hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)  # 根据门的输出计算最终值

        return torch.matmul(hh_i, self.W_e) + self.b_e  # 返回最终输出


class Encoder(nn.Module):  # 定义Encoder类
    def __init__(self, d_model, N, heads, m, dropout):  # 初始化方法，接受模型维度、层数、头数等参数
        super().__init__()  # 调用父类的初始化方法
        self.N = N  # 保存编码器层的数量
        self.pe = PositionalEncoder(d_model)  # 初始化位置编码器
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)  # 克隆多个编码器层
        self.norm = Norm(d_model)  # 初始化归一化层
        self.d_model = d_model  # 保存模型维度

    def forward(self, src, t):  # 定义前向传播方法
        src = src.reshape(26, self.d_model)  # 根据模型维度调整输入形状
        x = self.pe(src, t)  # 进行位置编码处理
        for i in range(self.N):  # 遍历每一层
            x = self.layers[i](x, None)  # 通过编码器层进行处理
        return self.norm(x)  # 归一化输出


class PositionalEncoder(nn.Module):  # 定义位置编码器类
    def __init__(self, d_model):  # 初始化方法，接受模型维度
        super().__init__()  # 调用父类的初始化方法
        self.d_model = d_model  # 保存模型维度

    def forward(self, x, t):  # 定义前向传播方法，接受输入和时间
        x = x * math.sqrt(self.d_model)  # 缩放输入以增加相对大小

        pe = np.zeros(self.d_model)  # 创建位置编码器数组

        for i in range(0, self.d_model, 2):  # 遍历模型维度
            pe[i] = math.sin(t / (10000 ** ((2 * i) / self.d_model)))  # 计算正弦值
            pe[i + 1] = math.cos(t / (10000 ** ((2 * (i + 1)) / self.d_model)))  # 计算余弦值

        x = x + Variable(torch.Tensor(pe).to(x.device))  # 将位置编码加入输入
        return x  # 返回处理后的输入


def get_clones(module, N):  # 定义克隆函数，生成多个模块的克隆
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])  # 返回深拷贝的模块列表


class EncoderLayer(nn.Module):  # 定义编码器层
    def __init__(self, d_model, heads, dropout=0.5):  # 初始化方法，接受模型维度、头数和dropout率
        super().__init__()  # 调用父类的初始化方法
        self.norm_1 = Norm(d_model)  # 初始化第一层归一化
        self.norm_2 = Norm(d_model)  # 初始化第二层归一化
        self.attn = MultiHeadAttention(heads, d_model, dropout)  # 初始化多头注意力层
        self.ff = FeedForward(d_model)  # 初始化前馈层
        self.dropout_1 = nn.Dropout(dropout)  # 初始化第一层dropout
        self.dropout_2 = nn.Dropout(dropout)  # 初始化第二层dropout

    def forward(self, x, mask):  # 定义前向传播方法
        x2 = self.norm_1(x)  # 对输入进行第一次归一化
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))  # 添加注意力输出与dropout
        x2 = self.norm_2(x)  # 对输入进行第二次归一化
        x = x + self.dropout_2(self.ff(x2))  # 添加前馈输出与dropout
        return x  # 返回处理后的输入


class Norm(nn.Module):  # 定义归一化类
    def __init__(self, d_model, eps=1e-6):  # 初始化方法，接受模型维度和一个小的稳定性值
        super().__init__()  # 调用父类的初始化方法

        self.size = d_model  # 保存模型维度
        self.alpha = nn.Parameter(torch.ones(self.size))  # 定义可学习的参数alpha
        self.bias = nn.Parameter(torch.zeros(self.size))  # 定义可学习的偏置bias
        self.eps = eps  # 保存稳定性小值

    def forward(self, x):  # 定义前向传播方法
        # 计算归一化
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias  # 归一化计算
        return norm  # 返回归一化结果


class MultiHeadAttention(nn.Module):  # 定义多头注意力类
    def __init__(self, heads, d_model, dropout=0.5):  # 初始化方法，接受头数、模型维度及dropout率
        super().__init__()  # 调用父类的初始化方法

        self.d_model = d_model  # 保存模型维度
        self.d_k = d_model // heads  # 计算每个头的维度
        self.h = heads  # 保存头的数量

        self.q_linear = nn.Linear(d_model, d_model)  # 定义线性变换层
        self.v_linear = nn.Linear(d_model, d_model)  # 定义线性变换层
        self.k_linear = nn.Linear(d_model, d_model)  # 定义线性变换层
        self.dropout = nn.Dropout(dropout)  # 定义dropout层
        self.out = nn.Linear(d_model, d_model)  # 定义输出线性层

    def forward(self, q, k, v, mask=None):  # 定义前向传播方法
        bs = q.size(0)  # 获取批大小

        # 进行线性操作并分成头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  # 变换形状
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)  # 变换形状
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)  # 变换形状

        # 转置以获取维度bs * h * sl * d_model
        k = k.transpose(1, 2)  # 转置维度
        q = q.transpose(1, 2)  # 转置维度
        v = v.transpose(1, 2)  # 转置维度

        scores = attention(q, k, v, self.d_k, mask, self.dropout)  # 计算注意力得分

        # 拼接头并通过最终线性层
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)  # 转置并调整形状

        output = self.out(concat)  # 通过输出层获得结果

        return output  # 返回输出


def attention(q, k, v, d_k, mask=None, dropout=None):  # 定义注意力函数
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力分数

    if mask is not None:  # 如果有mask，添加维度以匹配
        mask = mask.unsqueeze(1)
    scores = F.softmax(scores, dim=-1)  # 计算softmax

    if dropout is not None:  # 如果有dropout，应用dropout
        scores = dropout(scores)

    output = torch.matmul(scores, v)  # 计算输出
    return output  # 返回输出


class FeedForward(nn.Module):  # 定义前馈层类
    def __init__(self, d_model, d_ff=512, dropout=0.5):  # 初始化方法，接受模型维度、前馈层维度及dropout率
        super().__init__()  # 调用父类的初始化方法
        self.linear_1 = nn.Linear(d_model, d_ff)  # 第一层线性变换，输入为d_model，输出为d_ff
        self.dropout = nn.Dropout(dropout)  # 定义dropout层
        self.linear_2 = nn.Linear(d_ff, d_model)  # 第二层线性变换，输入为d_ff，输出为d_model

    def forward(self, x):  # 定义前向传播方法
        x = self.dropout(F.relu(self.linear_1(x)))  # 先通过第一层线性变换，再应用ReLU激活和dropout
        x = self.linear_2(x)  # 通过第二层线性变换
        return x  # 返回输出
