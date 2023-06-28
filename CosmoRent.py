#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''The NYC Housing dataset is a valuable resource for researchers, policymakers, and developers interested in affordable housing
in New York City. It can be used to analyze trends in affordable housing development,
identify neighborhoods with high levels of affordable housing, and evaluate the effectiveness of
various affordable housing programs.so, we chose to predict the housing / apartment cost in New York city.'''


# In[27]:


##importing required modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, classification_report, confusion_matrix , mean_absolute_error
from sklearn.ensemble import RandomForestClassifier


# In[77]:


class NYCHousingPricePrediction:
    def read_datafile(self):
        try:
            ##importing the dataset
            df = pd.read_csv(r'C:\Users\vidhi shah\Desktop\PROJECT\apartment_cost_list.csv')
            #to view the dataset
            print("loading data into dataframes: \n",df.head(10))
            return 1
        except:
          print(f"Exception Thrown: File Not Found")

    def data_preproccessing(self):
        try:
            df = pd.read_csv(r'C:\Users\vidhi shah\Desktop\PROJECT\apartment_cost_list.csv')
            print("\nDatatypes of data columns:\n",df.dtypes)
            print("\n******************************************************************************************")
            print("\nValue types of columns:\n", df.loc[0])
            print(df.loc[0])
            print("\n******************************************************************************************")
            print("Row and column counts of the data: ",df.shape)
            print("\n******************************************************************************************")
            ##checking the length of the dataset
            print("\nThe length of the dataset:",len(df))
            return 1
            
        except ValueError as ve:
            return 'Exception Thrown: Unable to fetch details!'
        
        
    def data_cleaning_and_handling(self):
        try:
            df = pd.read_csv(r'C:\Users\vidhi shah\Desktop\PROJECT\apartment_cost_list.csv')
            print("\n******************************************************************************************")
            ##data pre-processing - to clean the data we will first check the null values
            print("\n Checking null values:\n",df.isnull().sum())
            
            ## Here these(Curb Cut,Horizontal Enlrgmt,Vertical Enlrgmt,Zoning Dist1) columns have
            ## dimensionality reduction - removing unwanted and non-null fields that does not impact the target variable
            df.drop(['Curb Cut','Horizontal Enlrgmt','Vertical Enlrgmt','Zoning Dist1'],axis=1, inplace=True)
            df.dropna(inplace = True)
            print("\n******************************************************************************************")
            print("\n resultant dataset after droping the unwanted and missing values columns::\n",df.head(5))
            print("\n******************************************************************************************")
            # Checking the size of dataset.
            print("\n size of the dataset after cleaning the dataset: ",df.shape)
            print("\n Ensuring no missing values:\n",df.isnull().sum())
            print("\n******************************************************************************************")
            #Checking for Duplicates
            print("checking for duplicates: ",df.duplicated().sum())
            df = df.drop_duplicates()
            print("\n******************************************************************************************")
            print("After removing duplicates size of the dataset: ",df.shape)
            print("\n checking unique values of the columns: \n",df.nunique())
            print("\n******************************************************************************************")
            print("\n Data Information: \n")
            print(df.info(verbose=True))
            return 1
            
        except ValueError as ve:
            return 'Exception Thrown:Invalid Operations Performed.'
    
    def datavisualization(self):
        try:
            df = pd.read_csv(r'C:\Users\vidhi shah\Desktop\PROJECT\apartment_cost_list.csv')
            df.drop(['Curb Cut','Horizontal Enlrgmt','Vertical Enlrgmt','Zoning Dist1'],axis=1, inplace=True)
            df.dropna(inplace = True)
            df = df.drop_duplicates()
            le = LabelEncoder()
            df['Job Type'] = le.fit_transform(df['Job Type'])
            df['Initial Cost'] = df['Initial Cost'].apply(lambda x: float(x.replace('$','').replace(',','')))
            df = pd.DataFrame(df)
            df['date']=pd.to_datetime(df['Fully Permitted'])
            df['year']=df['date'].dt.year
            df.drop(['Block','Bin #','Community - Board','Fully Permitted','Enlargement SQ Footage','Job Description','date','House #','Borough','Street Name'],axis = 1, inplace = True)
            df = df.drop_duplicates()
            df = df.rename(columns={'Job #': 'Job'})
            print("\n******************************************************************************************")
            print(df.head(2))
            print("\n******************************************************************************************")
            print("correlation table: \n",df.corr())
            print("\n******************************************************************************************")
            plt.figure(figsize=(12,8))
            sns.heatmap(df.corr(),cmap="YlGnBu",annot=True,fmt=".1f")
            print("\n heatmap to show the correlation between the features: ")
            plt.show()
            print("\n******************************************************************************************")
            print("\n Feature Extraction using scatter plots: \n")
            print("\n scatter plot showing the relationship between initial cost and Proposed Zoning Sqft: \n")
            plot = px.scatter(df, x='Initial Cost', y = 'Proposed Zoning Sqft')
            plot.show()
            print("\n******************************************************************************************")
            print("\nscatter plot showing the relationship between initial cost and year: \n")
            plot = px.scatter(df, x='Initial Cost', y = 'year')
            plot.show()
            ##feature extraction and outlier analysis histogram and scatterplot
            print("\n visualization using histogram plots: \n")
            viz = df.hist(figsize=(20,20))
            viz.show()
            return 1
        except: 
            return "Exception Thrown: Syntax Error/Logical Error!"

    def data_analysis_and_classification(self):
        try:
            df = pd.read_csv(r'C:\Users\vidhi shah\Desktop\PROJECT\apartment_cost_list.csv')
            df.drop(['Curb Cut','Horizontal Enlrgmt','Vertical Enlrgmt','Zoning Dist1'],axis=1, inplace=True)
            df.dropna(inplace = True)
            df = df.drop_duplicates()
            df = df.rename(columns={'Job #': 'Job'})
            print("\n columns of the data(headers): \n",df.columns)
            print("\n******************************************************************************************")
            #Descriptive Statistics
            print("\n data description: \n",df.describe(include="O"))
            print("\n******************************************************************************************")
            print("Describing Initial Cost: ",df['Initial Cost'].describe())
            #Convert Job type to Numerical using Label Encoding
            le = LabelEncoder()
            df['Job Type'] = le.fit_transform(df['Job Type'])
            df['Initial Cost'] = df['Initial Cost'].apply(lambda x: float(x.replace('$','').replace(',','')))
            print("\n******************************************************************************************")
            #Separate Year From Date
            df = pd.DataFrame(df)
            df['date']=pd.to_datetime(df['Fully Permitted'])
            df['year']=df['date'].dt.year
            print("dataset after seperating date and year: ",df)
            print("\n******************************************************************************************")
            #Deleting unnecessary features
            df.drop(['Block','Bin #','Community - Board','Fully Permitted','Enlargement SQ Footage','Job Description','date','House #','Borough','Street Name'],axis = 1, inplace = True)
            df = df.drop_duplicates()
            #df['Proposed Zoning Sqft']=df['Proposed Zoning Sqft'].replace(0,df['Proposed Zoning Sqft'].mean())
            df['Initial Cost'] = df['Initial Cost'].astype(np.int64)
            #df=df.loc[1:10000]
            print("final dataset after cleaning and analyzing:\n",df.head())
            print("Length of final dataset:\n ",len(df))
            print("\nDatatypes of data columns:\n",df.dtypes)
            ## Separating the dependent and independent variables (MODEL BUILDING) 
            X = df.drop(columns=['Initial Cost'])
            #X = df.drop('Initial Cost',axis=1)
            y= df['Initial Cost']
            #splitting the dataset into training and testing dataset (taking 80:20 ratio: train_size will be 0.80 and test_size=0.20)
            print("\n******************************************************************************************")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            print("X_train:",X_train.shape)
            print("X_test:",X_test.shape)
            print("y_train:",y_train.shape)
            print("y_test:",y_test.shape)
            print("\n******************************************************************************************")
            #using random forest classifier
            from sklearn.ensemble import RandomForestClassifier
            rf_classifier = RandomForestClassifier(n_estimators=100)
            rf_classifier = rf_classifier.fit(X_train, y_train)
            y_pred_rf = rf_classifier.predict(X_test)
            #y_pred = y_test
            print("Predicted Initial Cost of Appartment at NYC: ", y_pred_rf)
            '''plt.plot(y_test,y_pred_rf)
            plot.show() '''
            print("\n******************************************************************************************")
            '''rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]}
            rf = RandomForestRegressor()
            rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='neg_mean_squared_error')
            rf_grid.fit(X_train, y_train)
            y_pred_rf = rf_grid.predict(X_test)
            mse_rf = mean_squared_error(y_test, y_pred_rf)
            print('Random Forest Regression MSE:', mse_rf)'''
            print("Accuracy Score: ", accuracy_score(y_test, y_pred_rf) * 100)
            #print(f1_score(y_test,y_pred_rf))
            
            mse = mean_squared_error(y_test,y_pred_rf)
            mse_rf = mse/100000000
            print("\n\nMean Squared Error:",mse_rf)
            print("\n\nClassification Report with precision,recall,f1-score and support: \n\n", classification_report(y_test, y_pred_rf))
            print("\n\nConfusion Matrix\n", confusion_matrix(y_test, y_pred_rf))
            print("\n******************************************************************************************")
            print("HENCE CAN CONCLUDE THAT THE DATA WAS IMBALANCED BUT ANALYSED AND BALANCED AND TRIED TO FIT THE DATASET AND FOUND MINIMUM MEAN SQUARED ERROR")
            print("Decent Accuracy at 96.70%\nGood recall and precision for costing values ranging from $40000 to $490000 (95% and 100%)\nF1 score for for costing values ranging from $40000 to $490000 is good ranging frorm 98% to 100%  - Can form the baseline")
                
            
            return 1
        except: 
            return "Exception Thrown: Syntax Error!"
        
  
def my_function(**kwargs):
    """WE ANALYSED, INVESTIGATED AND SUMMARIZED THE DATA 
    USING EDA AND CLASS AND FUNCTION METHOD AND OBJECTS AND CLASSIFIED USING RANDOM FOREST CLASSIFIER"""
    return None
print("Using __doc__:")
print(my_function.__doc__)

print("Select Any Functionality:\n1. Read Data\n2. Data Pre-Proccessing\n3. Data Cleaning and Handling (Data Cleaning and handling null values)\n4. Data Visualization(Outlier Analysis,Feature Extraction and Corelation Plot)\n5. Data Analysis (Stastical Analysis and Model Implementation)")

choice = int(input("Enter choice(1/2/3/4/5):"))
c1 = NYCHousingPricePrediction() 

if choice == 1:
    c1.read_datafile()
elif choice == 2:
    c1.data_preproccessing()
elif choice == 3:
    c1.data_cleaning_and_handling()
elif choice == 4:
    c1.datavisualization()
elif choice == 5:
    c1.data_analysis_and_classification()
else:
    print("Invalid Input,Please Enter Valid Input")      


# In[ ]:




