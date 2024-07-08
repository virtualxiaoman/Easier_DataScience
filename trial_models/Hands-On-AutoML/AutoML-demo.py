# from sklearn.datasets import load_iris
# import pandas as pd
#
# # 加载鸢尾花数据集
# iris = load_iris()
#
# # 创建 DataFrame
# df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
#
# from mlbox.preprocessing import Reader
# from mlbox.preprocessing import Drift_thresholder
# from mlbox.optimisation import Optimiser
# from mlbox.prediction import Predictor
#
# # 1. 读取数据
#
# # 2. 数据清洗和特征工程
# drift_thresholder = Drift_thresholder()
# df = drift_thresholder.fit_transform(df)
#
# # 3. 模型优化和选择
# opt = Optimiser()
# best_params = opt.optimise(df, 'target')
#
# # 4. 训练和预测
# predictor = Predictor()
# predictor.fit_predict(best_params, df)

from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
#
# # Digits dataset that you have used in Auto-sklearn example
# digits = load_digits()
# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
#                                                     train_size=0.75, test_size=0.25)
#
# # You will create your TPOT classifier with commonly used arguments
# tpot = TPOTClassifier(generations=10, population_size=30, verbosity=2)
#
# # When you invoke fit method, TPOT will create generations of populations, seeking best set of parameters. Arguments you have used to create TPOTClassifier such as generaions and population_size will affect the search space and resulting pipeline.
# tpot.fit(X_train, y_train)
#
# print(tpot.score(X_test, y_test))
# # 0.9834
#
# # %%
# tpot.export('my_pipeline.py')
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

# 创建并训练 RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 预测并评估模型
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"RandomForestClassifier Accuracy: {accuracy:.4f}")
# 例如：RandomForestClassifier Accuracy: 0.9733
