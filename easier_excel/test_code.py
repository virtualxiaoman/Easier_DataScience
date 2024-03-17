import pandas as pd

import easier_excel.read_data as xm_rd
import easier_excel.draw_data as xm_dd
import easier_excel.cal_data as xm_cd
from scipy import stats

xm_rd.set_pd_option(max_show=True, float_type=True, decimal_places=2)

path = '../input/CharacterData.xlsx'
df = xm_rd.read_df(path)

x = stats.norm.rvs(size=100)
xm_cd.cal_skew_kurtosis(x)
# x = pd.DataFrame(x, columns=['x'])
# draw_df = xm_dd.draw_df(x)
# draw_df.draw_density(target_name=None, classify=False, feature_name='x')

