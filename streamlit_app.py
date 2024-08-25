import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

# Custom CSS for styling with colors
page_bg_color = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-color: #1a1a1a;
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #2a2a2a;
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}

footer {visibility: hidden;}
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #2a2a2a;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    color: #f0f0f0;
}
.footer a {
    color: #f97316;
    text-decoration: none;
}
.footer a:hover {
    color: #ea580c;
}
</style>
<div class="footer">
    <p>Developed with ❤️ by <a href="https://zaintheanalyst.com" target="_blank">Zain Haidar</a></p>
    <p>
        <a href="https://github.com/zainhaidar16" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/zain-haidar" target="_blank">LinkedIn</a> |
        <a href="mailto:contact@zaintheanalyst.com">Email</a>
    </p>
</div>
"""

st.markdown(page_bg_color, unsafe_allow_html=True)

# Load the dataset
data = pd.read_csv("candy-data.csv")

# App Title and Introduction
st.title("Candy Bar Prediction App")
st.markdown("""
This app predicts whether a candy is classified as a **bar** or not based on various attributes like sugar percentage, price, and others.
Use the sidebar to explore the dataset, view the heatmap, and make predictions.
""")

# Sidebar for options
st.sidebar.title("Candy Bar Prediction")
st.sidebar.subheader("Options")

# Sidebar options for viewing the dataset and heatmap
view_data = st.sidebar.checkbox("View Dataset")
view_heatmap = st.sidebar.checkbox("View Heatmap")

if view_data:
    st.subheader("Candy Bar Dataset")
    st.write(data.head(11))  # Show top 11 rows of the dataset

if view_heatmap:
    st.subheader("Feature Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[float, int])  # Select only numeric columns
    figure = plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_data.corr(), annot=True, vmin=-1, vmax=1)
    st.pyplot(figure)

# Prepare data for model training
names = data['competitorname']
data.drop('competitorname', axis=1, inplace=True)

y = data['bar']  # Target variable
X = data.drop('bar', axis=1)  # Features

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Model training or loading
try:
    model = load("model.joblib")
except FileNotFoundError:
    model = LogisticRegression()
    model.fit(X_train, y_train)
    dump(model, "model.joblib")

# Display model accuracy
st.sidebar.text(f"Model Accuracy: {model.score(X_test, y_test):.2f}")

# Sidebar for predictions
st.sidebar.subheader("Make a Prediction")
selected_candy = st.sidebar.selectbox("Select a Candy", names)

if selected_candy:
    candy_index = names[names == selected_candy].index[0]
    candy_features = X.iloc[candy_index, :].values.reshape(1, -1)
    prediction = model.predict(candy_features)
    probability = model.predict_proba(candy_features)[0][1]

    st.subheader(f"Prediction for {selected_candy}")
    if prediction[0] == 1:
        st.success(f"{selected_candy} is classified as a bar with a probability of {probability:.2%}.")
    else:
        st.warning(f"{selected_candy} is not classified as a bar with a probability of {(1 - probability):.2%}.")