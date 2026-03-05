
import statsmodels.api as sm

def run_ols(df,y,X):
    X = sm.add_constant(df[X])
    model = sm.OLS(df[y],X).fit()
    return model
