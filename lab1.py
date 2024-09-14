import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
titanic_data = pd.read_csv('D:/data/titanic.csv')
# 计算乘客的平均年龄
titanic_data['AgeGroup'] = pd.cut(titanic_data['Age'],
bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# 统计每个年龄组的人数
age_group_count = titanic_data['AgeGroup'].value_counts().sort_index()

# 绘制柱状图
plt.bar(age_group_count.index, age_group_count)
plt.xlabel('group')
plt.ylabel('count')
plt.title('the age of titanic passengers')
plt.xticks(rotation=45)
plt.show()


# 统计性别分布
gender_count = titanic_data['Sex'].value_counts()
# 绘制饼状图
plt.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%')
plt.title('the gender of titanic passengers')
plt.show()


survived = titanic_data[titanic_data['Survived'] == 1].groupby('Sex')['Survived'].count()
not_survived = titanic_data[titanic_data['Survived'] == 0].groupby('Sex')['Survived'].count()

# 创建柱状图
fig, ax = plt.subplots()
ax.bar(['Survived', 'Not Survived'], [survived['female'], not_survived['female']], label='Female')
ax.bar(['Survived', 'Not Survived'], [survived['male'], not_survived['male']], label='Male', bottom=[survived['female'], not_survived['female']])

ax.set_xlabel('Survival Status')
ax.set_ylabel('Count')
ax.set_title('Impact of Gender on Survival Probability')
ax.legend()

plt.show()

# 去掉空数据
titanic_data = titanic_data.dropna(subset=['Cabin'])

# 统计每个乘客的生还情况与所在船舱的关系
cabin_survival = titanic_data.groupby(['Cabin', 'PassengerId']).agg({'Survived': 'first'}).reset_index()

# 统计每个船舱的生还人数和总人数
cabin_survival = cabin_survival.groupby('Cabin').agg({'Survived': ['sum', 'count']})
cabin_survival.columns = ['SurvivedCount', 'TotalCount']

# 计算每个船舱的生还概率
cabin_survival['SurvivalProbability'] = cabin_survival['SurvivedCount'] / cabin_survival['TotalCount']

# 绘制相关性散点图
plt.scatter(cabin_survival['TotalCount'], cabin_survival['SurvivalProbability'])
plt.xlabel('Total Count in Cabin')
plt.ylabel('Survival Probability')
plt.title('Impact of Cabin on Survival Probability')
plt.show()

