import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="Accident Severity Prediction", page_icon="🚦", layout="wide")

# --------------------------
# Custom CSS
# --------------------------
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .highlight-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(255, 75, 43, 0.1);
        border: 1px solid rgba(255, 75, 43, 0.4);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved model & data
model = pickle.load(open("final_accident_severity_model.pkl", "rb"))         # trained pipeline
train_columns = pickle.load(open("train_columns.pkl", "rb"))  # feature names used
df = pd.read_csv("cleaned_accident_data_SL.csv")            # dataset

# Sidebar menu
menu = ["Home", "Data Exploration", "Visualisations", "Prediction", "Model Performance", "About"]
choice = st.sidebar.selectbox("📂 Menu", menu)


# HOME PAGE
if choice == "Home":
    st.markdown('<p class="big-font">🚦 Accident Severity Prediction App</p>', unsafe_allow_html=True)
    st.write("""
    Road accidents happen every day and can cause injuries, damage, or even death. Some accidents are minor, while others are very serious. **Predicting accident severity in advance** 
    can help emergency services respond faster and save lives. With the power of **Artificial Intelligence (AI)** and **Machine Learning (ML)**. This project studies past accident data to detect patterns and predict the **severity of road accidents** 
    based on **time, weather, location, and road conditions**.  

    **Why is this important?**  
    - Emergency services often arrive without knowing how serious the accident is.  
    - Delays in correct response can cost valuable lives.  
    - No smart, proactive system exists today to assess accident severity in real time.  

    **Our Solution:**  
    - Build an AI system that predicts accident severity (Low, Medium, High).  
    - Use geo-temporal, weather, and road features to estimate risk.  
    - Provide faster, data-driven insights for emergency responders and road safety planners.  

    """)

    st.markdown('<div class="highlight-box">Explore dataset<br>Visualise patterns<br>Predict accident severity<br>View model performance</div>', unsafe_allow_html=True)

    # Dataset stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📄 Dataset Rows", value=df.shape[0])
    with col2:
        st.metric(label="🔢 Features", value=df.shape[1])

    # Fun facts
    road_facts = [
        "🚗 Speeding is a factor in about 1/3 of all fatal crashes.",
        "🛵 Wearing a helmet reduces risk of death by 37% for motorcycle riders.",
        "🚦 Human error contributes to over 90% of road accidents.",
        "🌧 Wet roads increase crash risk by 34%.",
        "📱 Texting while driving makes a crash 23x more likely."
    ]
    st.info(random.choice(road_facts))


# DATA EXPLORATION
elif choice == "Data Exploration":
    st.header("Data Exploration")
    st.write("Preview and explore the accident dataset.")

    # Check missing values per column
    missing_df = df.isnull().sum()
    total_missing = int(missing_df.sum())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Total Rows", df.shape[0])
    with col2:
        st.metric("🔢 Total Columns", df.shape[1])
    with col3:
        st.metric("❓ Missing Values", total_missing)

    if total_missing > 0:
        st.warning("⚠ Dataset still contains missing values. Example:")
        st.dataframe(missing_df[missing_df > 0].sort_values(ascending=False).head(10))

    st.markdown("### Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)


# VISUALISATIONS
elif choice == "Visualisations":
    st.header("Data Visualisations")

    # Severity distribution
    st.markdown("### Distribution of Accident Severity")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Severity", palette="Reds", ax=ax1)
    st.pyplot(fig1)

    # Correlation heatmap
    st.markdown("### Feature Correlation Heatmap")
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)


# PREDICTION
elif choice == "Prediction":
    st.header("Predict Accident Severity")
    st.write("Enter accident conditions to predict severity level.")

    lat = st.number_input("📍 Latitude", float(df["Start_Lat"].min()), float(df["Start_Lat"].max()), step=0.01)
    lng = st.number_input("📍 Longitude", float(df["Start_Lng"].min()), float(df["Start_Lng"].max()), step=0.01)
    distance = st.number_input("✏️ Distance (mi)", 0.01, 100.0, step=0.01)
    hour = st.slider("🕒 Hour of Day", 0, 23, 8)
    temp = st.slider("🌡️ Temperature (°F)", 0, 120, 70)
    humidity = st.slider("💧 Humidity (%)", 0, 100, 50)

    wind_options = df["Wind_Direction"].unique().tolist()
    wind_dir = st.selectbox("💨 Wind Direction", wind_options)

    weather_options = df["Weather_Condition"].unique().tolist()
    weather = st.selectbox("🌤 Weather Condition", weather_options)

    input_dict = {
        "Start_Lat": lat,
        "Start_Lng": lng,
        "Distance(mi)": distance,
        "Hour": hour,
        "Temperature(F)": temp,
        "Humidity(%)": humidity,
        "Wind_Direction": wind_dir,
        "Weather_Condition": weather
    }

    input_df = pd.DataFrame([input_dict])

    input_encoded = pd.get_dummies(input_df)

    input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

   
    # Predict
    if st.button("Predict Severity"):
        prediction = model.predict(input_encoded)[0]

        severity_mapping = {
            1: "Low severity — accident probability is low 🚙",
            2: "Moderate severity — accident probability is moderate ⚠️",
            3: "High severity — accident probability is high 🚨",
            4: "Very high severity — accident probability is very high 🔴"
        }

        st.success(f"Predicted Severity Level: {prediction}")
        st.info(severity_mapping.get(prediction, "Unknown severity level"))



# MODEL PERFORMANCE 
elif choice == "Model Performance":
    st.subheader("Model Performance")
    st.markdown("This section shows the accuracy of the trained Accident Severity model.")

    # Prepare features and target
    X = df.drop("Severity", axis=1)
    y = df["Severity"]

    # Encode categorical features
    X_encoded = pd.get_dummies(X)
    X_encoded = X_encoded.reindex(columns=train_columns, fill_value=0)

    # Predict using the final model
    y_pred = model.predict(X_encoded)
    accuracy = (y_pred == y).mean()

    st.markdown(f"**Final Model Accuracy:** {accuracy:.2%}")

    
    # Bar chart visualization
    st.markdown("### Accuracy Visualization")

    import matplotlib.pyplot as plt

    models = ["Final Model"]
    accuracies = [accuracy]

    fig, ax = plt.subplots(figsize=(3, 5)) 
    bars = ax.bar(models, accuracies, color="skyblue", width=0.4)

    # Set limits and labels
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accident Severity Model Accuracy")

    # Display exact accuracy value on top of the bar
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{acc:.2%}", ha='center', fontsize=10)

    st.pyplot(fig)

# ABOUT Section
elif choice == "About":
    st.header("ℹ️ About This Project")
    st.write("""
    This **Accident Severity Prediction App** is part of the mini-project 
    *AI-Driven Road Accident Detection and Severity Prediction with Geo-Temporal Intelligence*.

    ### Key Highlights:
    - **Data Source:** Accident dataset (Sri Lanka sample from US dataset).
    - **Features:** Geo-temporal, weather, and road features.
    - **Model Used:** Decision Tree Classifier (final model).
    - **Deployment:** Streamlit app for interactive exploration and prediction.

    ### Functionality:
    - Explore and visualize accident data.
    - Predict accident severity based on user inputs.
    - View model performance (accuracy & bar chart visualization).

    ---  
    **Developed with Python, Scikit-learn, Pandas, Matplotlib, Seaborn, and Streamlit.**

    ### Deployed App
    You can interact with the live app here:  
    [🚦 Accident Severity Prediction App](https://prashoharan-accident-severity-prediction-app-c56g9f.streamlit.app/)
    """)
