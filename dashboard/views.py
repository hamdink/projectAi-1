from django.shortcuts import render
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

def display_data(request):
    try:
        # Load data
        data = pd.read_csv("city_day.csv")
        data.dropna(axis=0, inplace=True)

        # Convert 'Date' column to datetime
        data['Date'] = pd.to_datetime(data['Date'])

        # Plot AQI trend over time
        fig1 = px.line(data, x='Date', y='AQI', color='City', title='AQI Trend Over Time')

        # Plot AQI distribution by City
        fig2 = px.box(data, x='City', y='AQI', title='AQI Distribution by City')
        fig2.update_layout(xaxis={'categoryorder': 'total descending'})

        # Plot scatter plot matrix for selected features
        selected_features = ['PM2.5', 'NO2', 'CO', 'O3', 'AQI']
        fig3 = px.scatter_matrix(data[selected_features], title='Scatter Plot Matrix')

        # Check if model file exists
        model_file = "model.h5"
        if os.path.exists(model_file):
            # Load the trained model
            model = tf.keras.models.load_model(model_file)

            # Standardize input data
            scaler = StandardScaler()
            user_input = pd.DataFrame({'PM2.5': [81], 'PM10': [124], 'NO': [1.44], 'NO2': [20], 'NOx': [12], 'NH3': [10], 'CO': [0.1], 'SO2': [15], 'O3': [127], 'Benzene': [0.20], 'Toluene': [6], 'Xylene': [0.06]})
            user_input_scaled = scaler.transform(user_input)

            # Predict AQI
            user_pred = model.predict(user_input_scaled)
            predicted_aqi = user_pred[0][0]
        else:
            predicted_aqi = "Model file not found"

        # Prepare data for template
        context = {
            'fig1': fig1.to_html(),
            'fig2': fig2.to_html(),
            'fig3': fig3.to_html(),
            'predicted_aqi': predicted_aqi
        }

    except Exception as e:
        context = {
            'error_message': f"An error occurred: {str(e)}"
        }

    return render(request, 'index.html', context)
