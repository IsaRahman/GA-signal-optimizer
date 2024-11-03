import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Function to read data from an Excel file
def read_data_from_excel(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    x = df['x'].values
    y1 = df['y1'].values
    y2 = df['y2'].values
    return x, y1, y2

# Regression function for polynomial regression using sklearn
def polynomial_regression_sklearn(x, y, degree):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    # model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    
    model.fit(X_poly, y)
    return model





# File path to the Excel file
file_path = 'C:/Users/Isa Rahman/Desktop/Final Thesis Code/Thesis.xlsx'


# Object Function Initialization

# GA Object Function
GAx, GAy1, GAy2 = read_data_from_excel(file_path, 'GA')
GA_Through_degree = 7
GA_Delay_degree = 7
# GA_Through_regression = polynomial_regression_sklearn(GAx, GAy1, GA_Through_degree)
# GA_Delay_regression = polynomial_regression_sklearn(GAx, GAy2, GA_Delay_degree)

# MA Object Function
MAx, MAy1, MAy2 = read_data_from_excel(file_path, 'MA')
MA_Through_degree = 6
MA_Delay_degree = 8
# MA_Through_regression = polynomial_regression_sklearn(MAx, MAy1, MA_Through_degree)
# MA_Delay_regression = polynomial_regression_sklearn(MAx, MAy2, MA_Delay_degree)

# GNA Object Function
GNAx, GNAy1, GNAy2 = read_data_from_excel(file_path, 'GNA')
GNA_Through_degree = 5
GNA_Delay_degree = 5
# GNA_Through_regression = polynomial_regression_sklearn(GNAx, GNAy1, GNA_Through_degree)
# GNA_Delay_regression = polynomial_regression_sklearn(GNAx, GNAy2, GNA_Delay_degree)

# KA Object Function
KAx, KAy1, KAy2 = read_data_from_excel(file_path, 'KA')
KA_Through_degree = 8
KA_Delay_degree = 8
# KA_Through_regression = polynomial_regression_sklearn(KAx, KAy1, KA_Through_degree)
# KA_Delay_regression = polynomial_regression_sklearn(KAx, KAy2, KA_Delay_degree)

#X = np.column_stack((GAx, MAx, GNAx, KAx))


GA_Through_regression = polynomial_regression_sklearn(GAx, GAy1, GA_Through_degree)
GA_Delay_regression = polynomial_regression_sklearn(GAx, GAy2, GA_Delay_degree)
MA_Through_regression = polynomial_regression_sklearn(MAx, MAy1, MA_Through_degree)
MA_Delay_regression = polynomial_regression_sklearn(MAx, MAy2, MA_Delay_degree)
GNA_Through_regression = polynomial_regression_sklearn(GNAx, GNAy1, GNA_Through_degree)
GNA_Delay_regression = polynomial_regression_sklearn(GNAx, GNAy2, GNA_Delay_degree)
KA_Through_regression = polynomial_regression_sklearn(KAx, KAy1, KA_Through_degree)
KA_Delay_regression = polynomial_regression_sklearn(KAx, KAy2, KA_Delay_degree)

# GA_Through_regression = polynomial_regression_sklearn(X, GAy1, GA_Through_degree)
# GA_Delay_regression = polynomial_regression_sklearn(X, GAy2, GA_Delay_degree)
# MA_Through_regression = polynomial_regression_sklearn(X, MAy1, MA_Through_degree)
# MA_Delay_regression = polynomial_regression_sklearn(X, MAy2, MA_Delay_degree)
# GNA_Through_regression = polynomial_regression_sklearn(X, GNAy1, GNA_Through_degree)
# GNA_Delay_regression = polynomial_regression_sklearn(X, GNAy2, GNA_Delay_degree)
# KA_Through_regression = polynomial_regression_sklearn(X, KAy1, KA_Through_degree)
# KA_Delay_regression = polynomial_regression_sklearn(X, KAy2, KA_Delay_degree)