from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)
        df.head()
        return df.to_json()

@app.route('/data-analysis', methods=['POST'])
def data_analysis():
    data = request.get_json()
    df = pd.DataFrame(data)
    # Example visualization
    fig = px.scatter_matrix(df)
    graph = fig.to_html(full_html=False)
    return graph

@app.route('/data-engineering', methods=['POST'])
def data_engineering():
    data = request.get_json()
    df = pd.DataFrame(data)
    # Example feature scaling
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled.to_json()

@app.route('/modeling', methods=['POST'])
def modeling():
    data = request.get_json()
    df = pd.DataFrame(data)
    y = df.pop('target')
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

    # Example model selection
    model_type = request.args.get('model', 'rf')
    if model_type == 'rf':
        model = RandomForestClassifier()
    elif model_type == 'logistic':
        model = LogisticRegression()
    elif model_type == 'nn':
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return jsonify({'accuracy': accuracy, 'report': report})

if __name__ == '__main__':
    app.run(debug=True)
