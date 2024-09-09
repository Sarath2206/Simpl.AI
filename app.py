import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# Make data accessible across pages
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar for navigation
app_mode = st.sidebar.selectbox('Navigation', ['Home', 'Data Loading', 'Data Analysis', 'Data Visualization', 'Modelling'])

# Home Page
if app_mode == 'Home':
    st.title("SIMPL.AI")
    st.subheader("A No-Code AI Platform")
    st.write("Upload your data and get insights in minutes!")

    # Add a description of the platform
    st.write("""
    SIMPL.AI is designed to provide fast, efficient, and user-friendly access to data insights and machine learning models.
    With our no-code platform, you can upload datasets, analyze them, visualize trends, and even build machine learning models without writing a single line of code!
    """)

    # Add a GIF or image to make the page interactive
    st.image("https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif", caption="No-Code AI in Action!")

# Data Loading Page
elif app_mode == 'Data Loading':
    st.title('Data Loading')
    
    # Allow data upload
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "pdf", "jpg", "png", "zip"])
    
    # Process the uploaded file
    if uploaded_file:
        file_type = uploaded_file.type
        
        if file_type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.write("CSV Data uploaded successfully!")
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            st.write("Excel Data uploaded successfully!")
        elif file_type == "application/pdf":
            # Code for PDF parsing can be added here
            st.write("PDF data parsing coming soon!")
        elif file_type in ["image/jpeg", "image/png"]:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image")
        elif file_type == "application/zip":
            # Code for handling zip files (for directories of images)
            st.write("Handling ZIP data (for directories) coming soon!")

        # Store the data globally
        st.session_state.df = df

# Data Analysis Page
elif app_mode == 'Data Analysis':
    if st.session_state.df is not None:
        df = st.session_state.df
        st.title("Data Analysis Report:")
        st.write("Data Shape:", df.shape)
        st.write("Data Types:", df.dtypes)
        st.write("Missing Values:", df.isnull().sum())
        st.write("Summary Statistics:", df.describe())

        # Perform SHAP analysis if applicable
        try:
            explainer = shap.Explainer(df)
            shap_values = explainer(df)
            st.write("SHAP Values:")
            st.write(shap_values)
        except:
            st.write("SHAP analysis is not applicable for this dataset.")
    else:
        st.write("Please upload data in the 'Data Loading' section.")

# Data Visualization Page
elif app_mode == 'Data Visualization':
    if st.session_state.df is not None:
        df = st.session_state.df
        st.title("Data Visualization")

        # Display loaded data
        st.write(df)

        # Choose columns for visualization
        columns = st.multiselect("Select columns to visualize", df.columns)
        if len(columns) >= 2:
            graph_type = st.multiselect("Choose graph types", ["Scatter Plot", "Line Plot", "Bar Plot", "Pie Chart", "Heatmap"])

            for graph in graph_type:
                fig, ax = plt.subplots()
                if graph == "Scatter Plot":
                    sns.scatterplot(x=df[columns[0]], y=df[columns[1]], ax=ax)
                elif graph == "Line Plot":
                    sns.lineplot(x=df[columns[0]], y=df[columns[1]], ax=ax)
                elif graph == "Bar Plot":
                    sns.barplot(x=df[columns[0]], y=df[columns[1]], ax=ax)
                elif graph == "Pie Chart" and len(columns) == 1:
                    df[columns[0]].value_counts().plot.pie(autopct="%1.1f%%")
                elif graph == "Heatmap" and len(columns) >= 2:
                    sns.heatmap(df[columns].corr(), annot=True, ax=ax)

                st.pyplot(fig)
    else:
        st.write("Please upload data in the 'Data Loading' section.")

# Modelling Page
elif app_mode == 'Modelling':
    if st.session_state.df is not None:
        df = st.session_state.df
        st.title("Modelling")

        # Split Data
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]   # Target

        # Train-test-validation sliders
        st.write("Select data split:")
        train_size = st.slider("Training Data", 0.1, 1.0, 0.6, step=0.1)
        test_size = st.slider("Testing Data", 0.1, 1.0 - train_size, 0.2, step=0.1)
        validation_size = 1.0 - train_size - test_size

        if train_size + test_size + validation_size == 1.0:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-train_size, random_state=42)
            X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=validation_size/(test_size+validation_size), random_state=42)

            st.write(f"Training Size: {train_size}, Testing Size: {test_size}, Validation Size: {validation_size}")

            # Model selection
            model_type = st.selectbox("Select Model Type", ["Classification", "Regression"])
            if model_type == "Classification":
                models = ["Logistic Regression", "Decision Tree", "Random Forest", "Neural Network"]
            else:
                models = ["Linear Regression", "Decision Tree", "Random Forest", "Neural Network"]

            selected_model = st.selectbox("Choose a model", models)

            if model_type == "Classification":
                if selected_model == "Logistic Regression":
                    model = LogisticRegression()
                elif selected_model == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif selected_model == "Random Forest":
                    model = RandomForestClassifier()
                elif selected_model == "Neural Network":
                    model = MLPClassifier()

            else:
                if selected_model == "Linear Regression":
                    model = LinearRegression()
                elif selected_model == "Decision Tree":
                    model = DecisionTreeRegressor()
                elif selected_model == "Random Forest":
                    model = RandomForestRegressor()
                elif selected_model == "Neural Network":
                    model = MLPRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display results
            if model_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.write("Model Accuracy:", accuracy)
                st.write("Classification Report:")
                st.write(classification_report(y_test, y_pred))
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
            else:
                mse = mean_squared_error(y_test, y_pred)
                st.write("Model MSE:", mse)
    else:
        st.write("Please upload data in the 'Data Loading' section.")
