# 如果没有安装pandas，请取消下一行的注释
# !pip install pandas
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


#导入相关包
import warnings
import torch
from torch import nn
from d2l import torch as d2l
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
#设置sns样式
sns.set(style='white',context='notebook',palette='muted')
import matplotlib.pyplot as plt

train_data = pd.read_csv(r'D:\data\train.csv')
test_data=pd.read_csv(r'D:\data\test.csv')

#print(train_data.shape,test_data.shape)
all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))


#all_features['Cabin']=all_features['Cabin'].fillna('U')
#print(all_features['Embarked'].value_counts())
print(all_features.info())
#print(all_features[all_features['Fare'].isnull()])

#将旅客姓名处理为更靠谱的title特征
all_features['Title']=all_features['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
#print(all_features['Title'].value_counts())
TitleDict={}
TitleDict['Mr']='Normal'
TitleDict['Mlle']='Normal'
TitleDict['Miss']='Normal'
TitleDict['Master']='Master'
TitleDict['Jonkheer']='Master'
TitleDict['Mme']='Normal'
TitleDict['Ms']='Normal'
TitleDict['Mrs']='Normal'
TitleDict['Don']='Royalty'
TitleDict['Sir']='Royalty'
TitleDict['the Countess']='Royalty'
TitleDict['Dona']='Royalty'
TitleDict['Lady']='Royalty'
TitleDict['Capt']='Officer'
TitleDict['Col']='Officer'
TitleDict['Major']='Officer'
TitleDict['Dr']='Officer'
TitleDict['Rev']='Officer'

all_features['Title']=all_features['Title'].map(TitleDict)
all_features['Sex'] = all_features['Sex'].map({'male': 0, 'female': 1})

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features=all_features.drop(['SibSp','Ticket','Cabin','Name'],axis=1)

all_features=pd.get_dummies(all_features,dummy_na=True)
#print(all_features.iloc[0:4,:].to_string())
#TickCountDict={}
#TickCountDict=all_features['Ticket'].value_counts()
#print(TickCountDict.head())
"""
# 首先，我们为家庭总人数创建一个新特征 'FamilySize'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
# 提取我们关心的特征：存活率（Survived）、年龄（Age）、客舱等级（Pclass）、家庭总人数（FamilySize）
correlation_data = train_data[['Survived', 'Age', 'Pclass','SibSp','Fare','Parch','FamilySize','Sex']]
# 由于年龄有缺失值，我们可以将其填补为平均值，确保计算相关性时不出错

corr_matrix = correlation_data.corr()

# 使用 seaborn 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap='BrBG', annot=True, linewidths=0.5)
plt.xticks(rotation=45)
plt.title('Correlation between Survived, Age, Pclass, and FamilySize')
plt.show()

print(all_features[all_features['Age'].isnull()].head())
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features.drop(['SibSp','Parch'],axis=1)
all_features['Embarked'] = all_features['Embarked'].fillna(all_features['Embarked'].mode()[0])
# 填充 Cabin 列的缺失值，可以使用 'U'（未知）填充
all_features['Cabin'] = all_features['Cabin'].fillna('U')

"""

all_features = all_features.astype(float)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

train_labels = torch.tensor(train_data['Survived'].values.reshape(-1, 1), dtype=torch.float32)


loss = nn.MSELoss()


class MLP(nn.Module):
    def __init__(self, in_features):
        super(MLP, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.hidden(x)

in_features = train_features.shape[1]
net = MLP(in_features)

num_epochs = 100
learning_rate = 0.001
batch_size = 64
weight_decay = 0


loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


train_iter = load_array((train_features, train_labels), batch_size)

animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs])

def train_model(net, train_iter, num_epochs, loss_fn, optimizer):
    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式
        epoch_loss = 0
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 每个epoch的平均损失
        avg_loss = epoch_loss / len(train_iter)
        animator.add(epoch + 1, avg_loss)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')



train_model(net, train_iter, num_epochs, loss_fn, optimizer)
d2l.plt.show()

net.eval()
with torch.no_grad():
    test_preds = net(test_features).numpy()


test_preds= (test_preds > 0.5)
test_data['Survived'] = pd.Series(test_preds.reshape(1, -1)[0].astype(int))

submission = test_data[['PassengerId', 'Survived']]
submission.to_csv('submission.csv', index=False)

