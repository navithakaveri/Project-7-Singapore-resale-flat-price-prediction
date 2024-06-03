import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import streamlit as st
from streamlit_option_menu import option_menu
import images
from PIL import Image
import seaborn as sns
from scipy.stats import skew 
import datetime as dt
import json
import pickle
df=pd.read_csv(r"C:\Users\navit\Downloads\singapore_pr.csv")
st.set_page_config(layout="wide")

def home():
    col1,col2=st.columns([4,4])
    with col1:
        
        st.title("SINGAPORE RESALE FLAT PRICE PREDICTING")
        icon=Image.open(r"C:\Users\navit\OneDrive\Pictures\Screenshots\sing.png")
        st.image(icon,use_column_width=True)
    with col2:
        st.header("STEPS INVOLVED IN THIS PROJECT")
        st.subheader("*DATA COLLECTION")
        st.subheader("*DATA PREPROCESSING")
        st.subheader("*EDA ANALYSIS")
        st.subheader("*RESALE FLAT PRICE PREDICTION")
def eda():
    tab1,tab2,tab3=st.tabs(["UNIVARIATE","BIVARIATE","MULTIVARIATE"])
    with tab1:
        #univariate analysis 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(15,8))
        ax = sns.countplot(data=df,x=df.flat_type)
        ax.set_title("Flat_type")
        st.pyplot()
        
        plt.figure(figsize=(15,8))
        ax = sns.countplot(data=df,x=df.storey_range)
        ax.set_title("storey_range")
        st.pyplot()
    
        
        #univariate analysis 
#analysing numerical variables using boxplot
        plt.figure(figsize=(10,8))
        ax = sns.boxplot(data=df,x=df.resale_price)
        ax.set_title("resale_price")
        st.pyplot()
        
        #univariate analysis 
#analysing numerical variables using boxplot
        plt.figure(figsize=(10,8))
        ax = sns.boxplot(data=df,x=df.minimum_storey)
        ax.set_title("minimum_storey")
        st.pyplot()
   
        
    def skewness_plot(df, *column):
        nrow = len(column)
        plot_no=0
        for col_name in column:
            if  'sqrt' in col_name:
                title= "After Treatment"
            else:
                title = "Before Treatment"

            plt.figure(figsize=(16, 8))

            plot_no+=1
            plt.subplot(nrow, 3, plot_no)
            sns.boxplot(x=col_name, data=df)
            plt.title('Boxplot - '+ title)

            plot_no+=1
            plt.subplot(nrow, 3, plot_no)
            sns.histplot(df[col_name])
            plt.title(f'histplot - Skewness: {skew(df[col_name])}')

            plot_no+=1
            plt.subplot(nrow, 3, plot_no)
            sns.violinplot(x=col_name, data=df)
            plt.title('Violinplot - ' + title)

        plt.tight_layout()
    
        return plt.show()
    numerical_columns = ['floor_area_sqm', 'resale_price']
    skewness_plot(df, *numerical_columns)
    st.pyplot()
        
    def Square_Root_Transformation(df, *column):

        for col_name in column:
            # Square Root Tansformation
            df[col_name+'_sqrt'] = np.sqrt(df[col_name])
            

        column =[i for i in df.columns if 'sqrt' in i]

        return skewness_plot(df, * column)
    Square_Root_Transformation(df, *numerical_columns)
    
    
    def outlier_plot(df):

        plt.figure(figsize=(16, 10))

        plt.subplot(2, 2, 1)
        sns.boxplot(x='floor_area_sqm', data=df)
        plt.title('Boxplot - floor area sqm')

        plt.subplot(2, 2, 2)
        sns.boxplot(x='floor_area_sqm_sqrt', data=df)
        plt.title('Boxplot - floor area sqm sqrt')

        plt.subplot(2, 2, 3)
        sns.boxplot(x='resale_price', data=df)
        plt.title('Boxplot - '+ 'resale price')

        plt.subplot(2, 2, 4)
        sns.boxplot(x='resale_price_sqrt', data=df)
        plt.title('Boxplot - '+ 'resale price sqrt')
        plt.tight_layout()
        
        return plt.show()
    outlier_plot(df)
    st.pyplot()
    with tab2:
        #Plotting a pair plot for bivariate analysis
        g = sns.PairGrid(df,vars=['floor_area_sqm','resale_price','maximum_storey','minimum_storey','year'])
        #setting color
        g.map_upper(sns.scatterplot, color='crimson')
        g.map_lower(sns.scatterplot, color='limegreen')
        g.map_diag(plt.hist, color='orange')
        #show figure
        plt.show()
        st.pyplot()
    with tab3:
        # Generate a heatmap to visualize the correlation matrix for the specified columns
        column_name = ['month', 'year', 'floor_area_sqm', 'floor_area_sqm_sqrt', 'lease_commence_date', 'minimum_storey','maximum_storey',
                    'is_remaining_lease', 'resale_price', 'resale_price_sqrt']
        sns.heatmap(df[column_name].corr(), annot= True)
        st.pyplot()
        
def regression_model(test_data):
    with open(r'Decision_Tree_Model.pkl', 'rb' ) as file:
        model = pickle.load(file)
        data = model.predict(test_data)[0] ** 2
        return data


    #  Load the JSON data into a Python dictionary
def main():
    
    page=option_menu("select",["HOME","EDA ANALYSIS","INSIGHTS"],orientation="horizontal")
    if page=="HOME":
        home()
    elif page=="EDA ANALYSIS":
         eda()
    elif page=="INSIGHTS":
        with open(r'Category_Columns_Encoded_Data.json', 'r') as file:
            data = json.load(file)
        st.title(":red[Singapore Resale] :blue[Flat Prices] :orange[Prediction]")

        col1, col2 = st.columns(2, gap= 'large')
        with col1:
            date = st.date_input("Select the **Item Date**", dt.date(2017, 1,1), min_value= dt.date(1990, 1, 1), max_value= dt.date(2023, 9,1))
            town = st.selectbox('Select the **Town**', data['town'])
            flat_type = st.selectbox('Select the **Flat Type**', data['flat_type'])
            block = st.selectbox('Select the **Block**', data['block']) 
            street_name = st.selectbox('Select the **Street Name**', data['street_name'])
        with col2:
            storey_range = st.selectbox('Select the **Storey Range**', data['storey_range'])
            floor_area_sqm = st.number_input('Enter the **Floor Area** in square meter', min_value = 28.0, max_value= 173.0, value = 60.0 )
            flat_model	= st.selectbox('Select the **flat_Model**', data['flat_model'])
            lease_commence_date	=st.number_input('Enter the **Lease Commence Year**', min_value = 1966.0, max_value= 2022.0, value = 2017.0 )
            remaining_lease	= st.selectbox('Select the **Remainig Lease**', data['remaining'])


        storey = storey_range.split(' TO ')
        if remaining_lease == 'Not Specified':
                is_remaining_lease = 0
        else:
                is_remaining_lease = 1

        test_data =[[date.month, data['town'][town], data['flat_type'][flat_type], data['block'][block], data['street_name'][street_name],
                                data['storey_range'][storey_range], floor_area_sqm, data['flat_model'][flat_model], lease_commence_date,
                                data['remaining'][remaining_lease], date.year, int(storey[0]), int(storey[1]),is_remaining_lease, 
                                np.sqrt(floor_area_sqm)]]

        st.markdown('Click below button to predict the **Flat Resale Price**')
        prediction = st.button('**Predict**')
        if prediction and test_data:
                st.markdown(f"### :bule[Flat Resale Price is] :green[$ {round(regression_model(test_data),3)}]")
                st.markdown(f"### :bule[Flat Resale Price in INR] :green[₹ {round(regression_model(test_data)*61.99,3)}]")
if __name__ == "__main__":
    main()

