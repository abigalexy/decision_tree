import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#读取数据
data = pd.read_csv('train.csv')
print(data.head())
print(data.info())

# 数据处理
x = data[['Pclass','Sex','Age']].copy()
y = data['Survived'].copy()
print(x.head(10))
x['Age'] = x['Age'].fillna(x['Age'].mean())
print(x.head(10))
x = pd.get_dummies(x)
print(x.head(10))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 模型训练
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# 模型评估
y_pre = model.predict(x_test)
print(classification_report(y_true=y_test, y_pred=y_pre))

# 可视化
plot_tree(model)
plt.show()