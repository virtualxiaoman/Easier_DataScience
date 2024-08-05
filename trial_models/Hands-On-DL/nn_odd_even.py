import torch
import numpy as np
import pandas as pd

# 创建数据集，第一列是一个随机正整数，第二个是标签，标签是奇数还是偶数(0表示偶数，1表示奇数)
def create_data(num_samples=100000):
    X = np.random.randint(100000, 1000000, size=(num_samples, 1))
    y = (X % 2 == 1).astype(np.int32)
    df_Xy = pd.DataFrame(np.concatenate([X, y], axis=1), columns=['X', 'y'])
    return df_Xy

df = create_data()
print(df.head())
# 保存为nn_odd_even.csv
df.to_csv("input/nn_odd_even.csv", index=False)


