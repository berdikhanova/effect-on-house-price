import pandas
pandas.set_option('max_rows', 10)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as statsmodels # useful stats package with regression functions
import seaborn as sns #plotting package

#style settings
sns.set(color_codes=True, font_scale = 1.2)
sns.set_style("whitegrid")

filename = r'C:\Users\marinaberdikhanova\Downloads\kc_house_data.csv'
#reading the dataset and separating the variables by ","
assignment_data = pandas.read_csv(filename, delimiter=",")

assignment_data.shape

import numpy as np
data_years = list(assignment_data['yr_built'].values)

data_price = list(assignment_data['price'].values)

def mult_regression(column_x, column_y):

    #plotting the regression line
    if len(column_x)==1:
        plt.figure()
        sns.regplot(x=column_x[0], y=column_y, data=assignment_data, marker="+",fit_reg=True,color='orange')
    
    #defining predictors X and response Y:
    X = data_years
    X = statsmodels.add_constant(X)
    Y = np.log(data_price)
    
    #constructing the model:
    global regressionmodel 
    regressionmodel = statsmodels.OLS(Y,X).fit() #OLS stands for "ordinary least squares"

    #creating a residual plot:
    plt.figure()
    residualplot = sns.residplot(x=regressionmodel.predict(), y=regressionmodel.resid, color='green')
    residualplot.set(xlabel='Fitted values for '+column_y, ylabel='Residuals')
    residualplot.set_title('Residuals vs Fitted values',fontweight='bold',fontsize=14)
    
    #QQ plot:
    qqplot = statsmodels.qqplot(regressionmodel.resid,fit=True,line='45')
    qqplot.suptitle("Normal Probability (\"QQ\") Plot for Residuals",fontweight='bold',fontsize=14)

mult_regression(['yr_built'],'price')
regressionmodel.summary()

def regression_model(column_x, column_y):
    # this function uses built-in library functions to create a scatter plot,
    #plotting of the residuals, computing R-squared, and displaying the regression equation
    X = statsmodels.add_constant(column_x)
    Y = column_y
    # fitting the regression line using "statsmodels" library:
    regressionmodel = statsmodels.OLS(Y,X).fit() #OLS = "ordinary least squares"
    
    # extracting regression parameters from model, rounded to 3 decimal places:
    Rsquared = round(regressionmodel.rsquared,3)
    slope = round(regressionmodel.params[1],3)
    intercept = round(regressionmodel.params[0],3)
    
    # making plots:
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(12,4))
    sns.regplot(x=column_x, y=column_y, data=assignment_data, marker="+", ax=ax1) #scatter plot
    sns.residplot(x=column_x, y=column_y, data=assignment_data, ax=ax2) #residual plot
    ax2.set(ylabel='Residuals')
    ax2.set_ylim(min(regressionmodel.resid)-1,max(regressionmodel.resid)+1)
    plt.figure() # histogram
    sns.distplot(regressionmodel.resid, kde=False, axlabel='Residuals', color='red')
    
    #printing the results:
    print("R-squared = ",Rsquared)
    print("Regression equation: "+'y'+" = ",slope,"* "+'x'+" + ",intercept)

regression_model(data_years,np.log(data_price))

from scipy.stats import pearsonr
corr, p_value = pearsonr(data_years, data_price)
print (p_value)
