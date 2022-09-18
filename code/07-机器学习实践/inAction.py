# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X,y = load_digits(return_X_y=True)
plt.imshow(X[0].reshape(8, 8), cmap='gray')
# 下面完成灰度图的绘制
# 灰度显示图像
plt.axis('off')# 关闭坐标轴

print('The digit in the image is {}'.format(y[0]))# 格式化打印
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25)
# , random_state=42

# 划分数据为训练集与测试集,添加stratify参数，以使得训练和测试数据集的类分布与整个数据集的类分布相同。
from sklearn.linear_model import LogisticRegression  # 求出Logistic回归的精确度得分

clf = LogisticRegression(
    solver='lbfgs', multi_class='ovr', max_iter=5000, random_state=42)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy score of the {} is {:.4f}'.format(clf.__class__.__name__,accuracy))
from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier轻松替换LogisticRegression分类器
clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__, accuracy))
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__,accuracy))
from sklearn.metrics import balanced_accuracy_score
y_pred = clf.predict(X_test)
accuracy = balanced_accuracy_score(y_pred, y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__,
                                                  accuracy))
from sklearn.svm import SVC, LinearSVC

clf = SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__,
                                                  accuracy))
clf = LinearSVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__,
                                                  accuracy))
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = LinearSVC()
clf.fit(X_train_scaled, y_train)
accuracy = clf.score(X_test_scaled, y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__,
                                                  accuracy))
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = LinearSVC()
clf.fit(X_train_scaled, y_train)
accuracy = clf.score(X_test_scaled, y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__,
                                                  accuracy))
from sklearn.metrics import confusion_matrix, classification_report
y_pred = clf.predict(X_test_scaled)
print(confusion_matrix(y_pred, y_test))
import pandas as pd
pd.DataFrame(
    (confusion_matrix(y_pred, y_test)),
    columns=range(10),
    index=range(10))
print(classification_report(y_pred, y_test))
from sklearn.model_selection import cross_validate

clf = LogisticRegression(
    solver='lbfgs', multi_class='auto', max_iter=1000, random_state=42)
scores = cross_validate(
    clf, X_train_scaled, y_train, cv=3, return_train_score=True)
clf.get_params()
import pandas as pd

df_scores = pd.DataFrame(scores)
from sklearn.model_selection import GridSearchCV

clf = LogisticRegression(
    solver='saga', multi_class='auto', random_state=42, max_iter=5000)
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1],
    'logisticregression__penalty': ['l2', 'l1']
}
tuned_parameters = [{
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2', 'l1'],
}]
grid = GridSearchCV(
    clf, tuned_parameters, cv=3, n_jobs=-1, return_train_score=True)
grid.fit(X_train_scaled, y_train)
grid.get_params()
df_grid = pd.DataFrame(grid.cv_results_)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
X = X_train
y = y_train
pipe = make_pipeline(
    MinMaxScaler(),
    LogisticRegression(
        solver='saga', multi_class='auto', random_state=42, max_iter=5000))
param_grid = {
    'logisticregression__C': [0.1, 1.0, 10],
    'logisticregression__penalty': ['l2', 'l1']
}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)
scores = pd.DataFrame(
    cross_validate(grid, X, y, cv=3, n_jobs=-1, return_train_score=True))
scores[['train_score', 'test_score']].boxplot()
pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)
print('Accuracy score of the {} is {:.2f}'.format(pipe.__class__.__name__, accuracy))
pipe.get_params()
import os
data = pd.read_csv('data/titanic_openml.csv', na_values='?')
data.head()
y = data['survived']
X = data.drop(columns='survived')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

