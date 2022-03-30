#这个代码是复现2021年 CCF C的论文Efficient Deep Learning Models for DGA Domain Detection
#主要复现其中第一个模型BiLstm加attention

from pickletools import optimize
from unicodedata import bidirectional
from numpy import float32
from sklearn.preprocessing import binarize
from sympy import Mod
import torch
import pandas as pd
import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

from tflearn.data_utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.autograd import Variable




#超参数和全局变量
parser = argparse.ArgumentParser("DGA")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
# parser.add_argument('--layers', type=int, default=20, help='total number of layers')
args = parser.parse_args()

#最大域名的长度，预处理时用到
max_document_length=48
#特征的个数
embedding_dim=60
#记录分类的类别数
classfication=18



#定义设备,把八个GPU都用上
if(torch.cuda.is_available()):
    device=torch.device("cuda:4")
else:
    device=torch.device("cpu")


banjori201861_path="/home/jishengpeng/dga/banjori201861.txt"
emotet5970_path="/home/jishengpeng/dga/emotet5970.txt"
flubot30018_path="/home/jishengpeng/dga/flubot30018.txt"
gameover12018_path="/home/jishengpeng/dga/gameover12018.txt"
murofet8578_path="/home/jishengpeng/dga/murofet8578.txt"
mydoom10063_path="/home/jishengpeng/dga/mydoom10063.txt"
necurs8208_path="/home/jishengpeng/dga/necurs8208.txt"
ngioweb5285_path="/home/jishengpeng/dga/ngioweb5285.txt"
pykspa45657_path="/home/jishengpeng/dga/pykspa45657.txt"
ramnit20104_path="/home/jishengpeng/dga/ramnit20104.txt"
ranbyus10938_path="/home/jishengpeng/dga/ranbyus10938.txt"
rovnix36782_path="/home/jishengpeng/dga/rovnix36782.txt"
shiotob8022_path="/home/jishengpeng/dga/shiotob8022.txt"
simda30316_path="/home/jishengpeng/dga/simda30316.txt"
symmi4274_path="/home/jishengpeng/dga/symmi4274.txt"
tinba74930_path="/home/jishengpeng/dga/tinba74930.txt"
viryt9775_path="/home/jishengpeng/dga/viryt9775.txt"
alexa100000_path="/home/jishengpeng/dga/alxea100000.txt"


class Dataset(torch.utils.data.Dataset):
	def __init__(self, x, label):
		super(Dataset, self).__init__()
		self.x = x
		self.label = label

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.label[idx]


#搭建网络
#先是一层lstm，然后是一层attention，然后是激活层，全连接，Dropout
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.embedding1=nn.Embedding(200,embedding_dim)   #前一个指编码中的最大数，后一个指第三维是多少
        self.lstm1=nn.LSTM(input_size=embedding_dim,hidden_size=128,num_layers=1,batch_first=True,bidirectional=True)  #batch_first表示是否将batch放在第一个,hidden_size表示多少个神经元
        self.attention=nn.MultiheadAttention(embed_dim=256,num_heads=1)
        self.flatten=nn.Flatten()
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.5)
        self.linear=nn.Linear(max_document_length*256,classfication)
        self.softmax=nn.Softmax(dim=1)

        self.linear1=nn.Linear(256,256)
        self.linear2=nn.Linear(256,256)
        self.linear3=nn.Linear(256,256)
    def forward(self,x):
        x=self.embedding1(x)      #维度是(batch_size,seq_length,embedding_dim)
        output,(h_n,c_n)=self.lstm1(x)  #维度是(batch_size,seq_length,hidden_size*2)

        tmp1=self.linear1(output)
        tmp2=self.linear2(output)
        output=self.linear3(output)

        attn_output, attn_output_weights = self.attention(output,tmp1,tmp2)   #(query,key,value)
        output=self.flatten(attn_output)    #维度是(batch_size,max_document_length*hidden_size)
        output=self.relu(output)
        output=self.dropout(output)
        output=self.linear(output)
        output=self.softmax(output)
        return output

def load_alexa():
    x=[]
    data = pd.read_csv(alexa100000_path,header=None)
    x=[i[0] for i in data.values]
    return x

def load_dga(path):
    x=[]
    data = pd.read_csv(path, sep="\t", header=None,
                      skiprows=18)
    x=[i[0] for i in data.values]
    return x

def get_feature_charseq():
    alexa=load_alexa()
    dga1=load_dga(banjori201861_path)
    dga1=dga1[0:100000]
    dga2=load_dga(emotet5970_path)
    dga3=load_dga(flubot30018_path)
    dga4=load_dga(gameover12018_path)
    dga5=load_dga(murofet8578_path)
    dga6=load_dga(mydoom10063_path)
    dga7=load_dga(necurs8208_path)
    dga8=load_dga(ngioweb5285_path)
    dga9=load_dga(pykspa45657_path)
    dga10=load_dga(ramnit20104_path)
    dga11=load_dga(ranbyus10938_path)
    dga12=load_dga(rovnix36782_path)
    dga13=load_dga(shiotob8022_path)
    dga14=load_dga(simda30316_path)
    dga15=load_dga(symmi4274_path)
    dga16=load_dga(tinba74930_path)
    dga17=load_dga(viryt9775_path)
    x=alexa+dga1+dga2+dga3+dga4+dga5+dga6+dga7+dga8+dga9+dga10+dga11+dga12+dga13+dga14+dga15+dga16+dga17
    max_features=10000
    #获取标签
    y=[]
    for i in range(0,100000):
    #   w=[]
    #   w.append(0)
        y.append(0)
    for i in range(0,len(dga1)):
    #   w=[]
    #   w.append(1)
        y.append(1)
    for i in range(0,len(dga2)):
    #   w=[]
    #   w.append(2)
        y.append(2)
    for i in range(0,len(dga3)):
        y.append(3)
    for i in range(0,len(dga4)):
        y.append(4)
    for i in range(0,len(dga5)):
        y.append(5)
    for i in range(0,len(dga6)):
        y.append(6)
    for i in range(0,len(dga7)):
        y.append(7)
    for i in range(0,len(dga8)):
        y.append(8)
    for i in range(0,len(dga9)):
        y.append(9)
    for i in range(0,len(dga10)):
        y.append(10)
    for i in range(0,len(dga11)):
        y.append(11)
    for i in range(0,len(dga12)):
        y.append(12)
    for i in range(0,len(dga13)):
        y.append(13)
    for i in range(0,len(dga14)):
        y.append(14)
    for i in range(0,len(dga15)):
        y.append(15)
    for i in range(0,len(dga16)):
        y.append(16)
    for i in range(0,len(dga17)):
        y.append(17)
    
    # y=[0]*len(alexa)+[1]*len(dga)
    
    #这个地方我提前将其变成tensor张量，后面变非常麻烦
    y=torch.Tensor(y)

    t=[]
    for i in x:                        #字符转ASCII值，把所有域名转换为数字
        v=[]
        for j in range(0,len(i)):
            v.append(ord(i[j]))
        t.append(v)

    x=t
 
    x = pad_sequences(x, maxlen=max_document_length, value=0.)  #数据预处理

    # x=x[0:30000]
    # y=y[0:30000]

    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)  #划分训练集和测试集
    return x_train, x_test, y_train, y_test



def main():
    train_x, test_x, train_label, test_label = get_feature_charseq()
    

    #数据加载
    train_data = Dataset(x=train_x, label=train_label)
    valid_data = Dataset(x=test_x, label=test_label)

    train_datasize=len(train_data)
    test_datasize=len(valid_data)

    # print(train_datasize)
    train_queue = torch.utils.data.DataLoader(
        train_data,batch_size=args.batch_size, shuffle=False, num_workers=2,drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2,drop_last=True)

    #定义模型
    model=Model()
    model=model.to(device)

    loss_fn=nn.CrossEntropyLoss()
    loss_fn=loss_fn.to(device)

    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)



    #开始训练
    for i in range(args.epochs):
        #记录训练的次数
        total_trainstep=0
        #记录测试的次数
        total_validstep=0
        print("--------------第{}轮训练开始--------------".format(i+1))

        model.train()
        for data in train_queue:
            input,label=data
            input = Variable(input)
            label = Variable(label)
            input=input.to(device)
            label=label.to(device)
            # input=input.float()
            # # print(input.dtype)
            # input=input.requires_grad_()
            # label=label.float()
            # label=label.requires_grad_()
            # label=torch.Tensor(label)
            # print(label.shape)
            output=model(input)
            # print(output)

            # print(label)
            # pre=output.argmax(1)
            # print(pre)
            loss=loss_fn(output,label.long())

            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_trainstep=total_trainstep+1

            if(total_trainstep%100==0):
                print("训练次数：{} , loss :{}".format(total_trainstep,loss))


        #测试步骤  
        model.eval() 
        total_test_loss=0
        # total_accuracy=0
        # sum=0
        final_true=[]
        final_pridict=[]
        with torch.no_grad():
            for data1 in valid_queue:
                input1,label1=data1
                input1 = Variable(input1)
                label1 = Variable(label1)
                input1=input1.to(device)
                label1=label1.to(device)

                output1=model(input1)
                loss1=loss_fn(output1,label1.long())

                # print(output1.argmax(1))
                # print(label1)
                for ii in range(0,args.batch_size):
                    final_pridict.append(output1.cpu().argmax(1)[ii])
                    final_true.append(label1.cpu()[ii])


                total_validstep=total_validstep+1

                if(total_validstep%100==0):
                    print("测试次数：{} , loss :{}".format(total_validstep,loss1))
                # print(output.argmax(1))
                # print(label)
                # sum=sum+1
                # accuracy=(output1.argmax(1)==label1).sum()
                total_test_loss=loss1+total_test_loss
                # total_accuracy=total_accuracy+accuracy
        print("整体测试集上的loss为{}".format(total_test_loss))
        # print("整体测试集上的准确率为:{}".format(total_accuracy/(sum*args.batch_size)))
        print(classification_report(final_true,final_pridict))


if __name__=='__main__':
    main()    




