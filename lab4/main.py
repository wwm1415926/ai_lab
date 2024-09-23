# 如果没有安装pandas，请取消下一行的注释
# !pip install pandas
import os
from xml.sax.handler import all_features

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
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
#设置sns样式
sns.set(style='white',context='notebook',palette='muted')
import matplotlib.pyplot as plt

train_data = pd.read_csv(r'D:\data\train.csv')
test_data=pd.read_csv(r'D:\data\test.csv')

#print(train_data.shape,test_data.shape)
all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

print(all_features.iloc[0:4,:].to_string())
#all_features['Cabin']=all_features['Cabin'].fillna('U')
#print(all_features['Embarked'].value_counts())
print(all_features.info())
#print(all_features[all_features['Fare'].isnull()])

#将旅客姓名处理为更靠谱的title特征
all_features['Title']=all_features['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
#print(all_features['Title'].value_counts())
TitleDict={}
TitleDict['Mr']=0
TitleDict['Mlle']=0
TitleDict['Miss']=0
TitleDict['Master']=1
TitleDict['Jonkheer']=1
TitleDict['Mme']=0
TitleDict['Ms']=0
TitleDict['Mrs']=0
TitleDict['Don']=2
TitleDict['Sir']=2
TitleDict['the Countess']=2
TitleDict['Dona']=2
TitleDict['Lady']=2
TitleDict['Capt']=3
TitleDict['Col']=3
TitleDict['Major']=3
TitleDict['Dr']=3
TitleDict['Rev']=3

all_features['Title']=all_features['Title'].map(TitleDict)
all_features['Sex'] = all_features['Sex'].map({'male': 0, 'female': 1})
all_features['Embarked'] = all_features['Embarked'].fillna('U')



numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features=all_features.drop(['Ticket','Name'],axis=1)
all_features=pd.get_dummies(all_features)

#print(all_features.iloc[0:4,:].to_string())
#TickCountDict={}
#TickCountDict=all_features['Ticket'].value_counts()
#print(TickCountDict.head())

#首先，我们为家庭总人数创建一个新特征 'FamilySize'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
# 提取我们关心的特征：存活率（Survived）、年龄（Age）、客舱等级（Pclass）、家庭总人数（FamilySize）
correlation_data = train_data[['Survived', 'Age', 'Pclass','SibSp','Fare','Parch','FamilySize','Sex']]
# 由于年龄有缺失值，我们可以将其填补为平均值，确保计算相关性时不出错

"""
corr_matrix = correlation_data.corr()

# 使用 seaborn 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap='BrBG', annot=True, linewidths=0.5)
plt.xticks(rotation=45)
plt.title('Correlation between Survived, Age, Pclass, and FamilySize')
plt.show()
"""
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

print(all_features.iloc[0:4,:].to_string())
all_features=all_features.drop(['Survived'],axis=1)
# 确保所有特征数据转换为 numpy 数组

train_features_np = all_features[:n_train].values
test_features_np = all_features[n_train:].values

# 标签数据
train_labels_np = train_data['Survived'].values  # 1D 标签

# 划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(train_features_np, train_labels_np, test_size=0.2, random_state=42)

print(y_train[0:100,])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
test_features_np = scaler.transform(test_features_np)

# 定义支持向量机模型 (可以选择不同的核函数，例如 'linear', 'rbf', 'poly' 等)
clf=RandomForestClassifier(criterion='entropy',random_state=1,n_jobs=-1,
                           max_depth=12,min_samples_leaf=1,min_samples_split=4,
                           n_estimators=100
                           )

clf=clf.fit(X_train,y_train)
# 在训练集上预测并计算准确率
y_train_hat = clf.predict(X_train)
print(y_train_hat[0:100])
print("Training accuracy:", metrics.accuracy_score(y_train, y_train_hat))

# 在验证集上预测并计算准确率
y_valid_hat = clf.predict(X_valid)
print("Validation accuracy:", metrics.accuracy_score(y_valid, y_valid_hat))

# 进行测试集预测
test_preds = clf.predict(test_features_np)

# 将预测结果转换为 0/1
test_data['Survived'] = test_preds

# 生成提交文件
submission = test_data[['PassengerId', 'Survived']]
submission.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'")
"""