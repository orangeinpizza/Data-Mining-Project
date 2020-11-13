# Data Mining Project Of Cars Sales
import pandas as pd         
import os
import time
from datetime import datetime
import csv
import glob
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


path="C:/Users/Chinmay/Desktop/DMProject/Data"  # Path for the directory

def Get_Data(name='All_Cars_Sales',addr='C:/Users/Chinmay/Desktop/DMProject/Data/ProjectData'):     # Step 1 : Data Integration
    statspath=path
    Clist=[x[0] for x in os.walk(statspath)]
    totaldataf=pd.DataFrame(columns=['Manufacturer','Model','Sales in thousands','4-year resale value','Vehicle type','Price in thousands','Engine size','Horsepower','Wheelbase','Width','Length','Curb weight','Fuel capacity','Fuel efficiency','Latest Launch'],index=None)

    
    for each_dir in Clist[1:]:
        each_file=os.listdir(each_dir)      # List of files in directory
        print('Files In Directory Are : ',each_file)
        if len(each_file)>0:
            for file in each_file:
                full_file_path=each_dir+'/'+file
                x=pd.read_csv(full_file_path)
                try:
                    totaldataf=totaldataf.append(x,sort=False)  # Copy the data in one data frame
                except Exception as e:
                    print(e)
                    pass

    save=addr+'/'+name+('.csv') # Save the data frame as a .csv file 
    print('Location Of File : ',save)
    totaldataf.to_csv(save,index=False)
    cols=list(totaldataf)
    for x in cols : 
        if x=='Unnamed: 0' :    # Removing unnecessary column
            totaldataf=totaldataf.drop('Unnamed: 0',1)
            print('Unnecessary Columns Removed.')
    print('List Of Columns In Dataframe : ',list(totaldataf))   # List of columns in the dataframe
    print('Dimensions Of The Frame : ',totaldataf.shape)


def Select_Data(name='All_Cars_Sales',addr='C:/Users/Chinmay/Desktop/DMProject/Data/ProjectData'):  # Step 2 : Data Selection

    loc=addr+'/'+name+('.csv')
    print('\nFile Location : '+loc)     # Location of file
    try:
        data=pd.read_csv(loc)   # Read the data in the dataframe
    except Exception as e:
        print(e)
        pass

    data.drop('Wheelbase', axis=1, inplace=True)   # Remove Unnecessary Columns
    data.drop('Width', axis=1, inplace=True)
    data.drop('Length', axis=1, inplace=True)
    print('DataFrame Updated!') 

    data.to_csv(loc,index=False)   # Save The File
    print('File Saved!')


def Clean_Data(name='All_Cars_Sales',addr='C:/Users/Chinmay/Desktop/DMProject/Data/ProjectData'):   # Step 3 : Data Cleaning

    loc=addr+'/'+name+('.csv')
    print('\nFile Location : '+loc)     # Location of file
    try:
        data=pd.read_csv(loc)   # Read the data in the dataframe
    except Exception as e:
        print(e)
        pass     
    print('DataFrame : \n',data,'\n') # Print the data of the dataframe


    check=data.isnull().sum()   # Check for null values
    print('Checking The Data : \n',check)
    
    cols=list(data)     # List of columns in the dataframe
    for x in cols : 
        if x=='Unnamed: 0' :    # Removing unnecessary column
            data=data.drop('Unnamed: 0',1)
            print('Unnecessary Columns Removed.')
    print(list(data))

    data=data.dropna(subset=['Manufacturer','Model'],how='any')  # Delete the row if Car is unknown
    data=data.dropna(subset=['Price in thousands','Engine size'],how='any')     # Delete the row if Price or Engine size is not known


    avg_price=data['Price in thousands'].mean()     # Finding average values
    avg_curb=data['Curb weight'].mean()
    avg_capacity=data['Fuel capacity'].mean()
    avg_efficiency=data['Fuel efficiency'].mean()
    print('\nAverage Value Of Price in thousands : '+str(avg_price))
    print('\nAverage Value Of Curb weight : '+str(avg_curb))
    print('\nAverage Value Of Fuel capacity : '+str(avg_capacity))
    print('\nAverage Value Of Fuel efficiency : '+str(avg_efficiency))    

    
    data['Price in thousands'].fillna(value=avg_price,inplace=True)     # Fill avg value in blank columns in the dataframe column
    data['Curb weight'].fillna(value=avg_curb,inplace=True)
    data['Fuel capacity'].fillna(value=avg_capacity,inplace=True)
    data['Fuel efficiency'].fillna(value=avg_efficiency,inplace=True)
    check=data.isnull().sum()
    print(check)


    null_values=data.isnull().sum()     # Again check for null elements
    print('Checking The Data Again : \n',null_values)
    print('DataFrame (After Data Cleaning) : \n',data)

    save=addr+'/'+('Cleaned')+('.csv') # Save the data frame as a .csv file 
    print('Location Of File : ',save)
    data.to_csv(save,index=False)    

def Analyse_Data(name='Cleaned',addr='C:/Users/Chinmay/Desktop/DMProject/Data/ProjectData'):   # Step 4 : Analyse Data

    loc=addr+'/'+name+('.csv')
    try:
        data=pd.read_csv(loc)
    except Exception as e:
        print(e)
        pass  

    print(data['Manufacturer'].describe())
    X=pd.read_csv(loc,usecols = ['Price in thousands'])
    X=X.values
    y=pd.read_csv(loc,usecols = ['Fuel efficiency'])
    y=y.values

    avg_fuel=data['Fuel efficiency'].mean()
    print('Average Fuel Efficency: ',avg_fuel)
    avg_horsepower=data['Horsepower'].mean()
    print('Average Fuel Horsepower: ',avg_horsepower)
    type_passanger=data['Vehicle type'].count()
    print(type_passanger)

    reg = LinearRegression().fit(X, y)
    reg.score(X, y)
    reg.coef_
    sx=np.sum(X)
    sy=np.sum(y)
    ssx = np.sum(X**2)
    ssxy = np.sum(X*y)
    
    a=(len(X)*ssxy-sx*sy)/(len(X)*ssx-sx*sx)
    b=(sy-a*sx)/len(X)
    print("SX =",sx)
    print("SY =",sy)
    print("SXY =",ssxy)
    xmean=sx/len(X)
    ymean=sx/len(y)
    print("XMEAN = ",xmean)
    print("YMEAN = ",ymean)
    reg = LinearRegression().fit(X, y)
    reg.score(X, y)
    reg.coef_
    reg.intercept_
    x=float(input("Enter the Price Value to predict the Fuel Efficiency coefficient :"))
    print("Value Of A : ",a)
    print("Value Of B : ",b)
    print("The Precicted Value is ",a*x+b)

    print("Linear Regression Coeff Is : ",reg.coef_)  


def Plot_Diagram():

    x=np.arange(0,3*np.pi,0.1) 
    y=np.sin(x) 
    plt.title("sine wave form") 

    # Plot the points using matplotlib 
    plt.plot(x,y) 
    plt.show() 

def Main():
    # Get_Data()
    # Select_Data()
    Clean_Data()
    # Plot_Diagram()
    # Analyse_Data()

if __name__ == '__main__':
    Main()