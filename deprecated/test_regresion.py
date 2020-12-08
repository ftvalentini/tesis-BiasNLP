import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.formula.api import ols


CORPUS_ID = 3
TARGET_A = "HE"
TARGET_B = "SHE"

file_results = \
        f"results/csv/biasbyword_full-C{CORPUS_ID}_{TARGET_A}-{TARGET_B}.csv"

dat = pd.read_csv(file_results)


mod = ols(formula='cosine_glove ~ dppmi + order2', data=dat)
res = mod.fit()
res.conf_int()
res.rsquared


mod_f = ols(formula='cosine_glove ~ dppmi + order2 + np.log(freq)', data=dat)
res_f = mod_f.fit()
res_f.conf_int()
res_f.rsquared


mod_f2 = ols(
    formula='cosine_glove ~ log_oddsratio + order2 + np.log(freq) + np.power(np.log(freq),2)'
    ,data=dat)
res_f2 = mod_f2.fit()
res_f2.conf_int()
res_f2.rsquared


(res_f2.rsquared - res.rsquared) / (1-res.rsquared)


mod_prueba = ols(formula='cosine_glove ~ dppmi + order2 + np.power(order2,2)', data=dat)
res = mod_prueba.fit()
res.conf_int()
res.rsquared
