from flask import Flask, render_template, jsonify, request
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from datetime import datetime
import os
import logging
import joblib

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiabetesAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.load_data()
        self.train_model()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df = self.clean_data(self.df)
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, df):
        # Replace 0 values with NaN for certain columns where 0 is not possible
        zero_not_possible = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in zero_not_possible:
            df[column] = df[column].replace(0, np.nan)

        # Fill NaN values with median
        for column in df.columns:
            if df[column].isnull().any():
                df[column] = df[column].fillna(df[column].median())

        return df

    def train_model(self):
        try:
            X = self.df.drop('Outcome', axis=1)
            y = self.df['Outcome']

            # Scaling the features
            X_scaled = self.scaler.fit_transform(X)

            # Train the logistic regression model
            self.model = LogisticRegression(random_state=42)
            self.model.fit(X_scaled, y)

            # Evaluate model performance using cross-validation
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
            logger.info(f"Cross-validation scores: {cv_scores}")

            # Evaluate metrics
            y_pred = self.model.predict(X_scaled)
            logger.info(f"Accuracy: {accuracy_score(y, y_pred)}")
            logger.info(f"Precision: {precision_score(y, y_pred)}")
            logger.info(f"Recall: {recall_score(y, y_pred)}")
            logger.info(f"F1 Score: {f1_score(y, y_pred)}")
            logger.info(f"ROC AUC Score: {roc_auc_score(y, y_pred)}")

            # Save the trained model
            joblib.dump(self.model, 'diabetes_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            logger.info("Model and scaler saved successfully")

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def reload_model(self):
        try:
            self.model = joblib.load('diabetes_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            logger.info("Model and scaler reloaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict_probability(self, patient_data):
        try:
            patient_scaled = self.scaler.transform(patient_data)
            probability = self.model.predict_proba(patient_scaled)[0][1]
            return round(probability * 100, 2)
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def get_feature_importance(self):
        features = self.df.drop('Outcome', axis=1).columns
        importance = abs(self.model.coef_[0])
        return dict(zip(features, importance))

    def create_visualizations(self):
        try:
            plots = {}

            # 1. Distribution of Glucose levels by Outcome
            glucose_fig = px.histogram(self.df, x="Glucose", color="Outcome",
                                       color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
                                       marginal="box")
            glucose_fig.update_layout(title="Glucose Distribution by Diabetes Status")
            plots['glucose_plot'] = glucose_fig.to_json()

            # 2. BMI vs Age interactive scatter
            scatter_fig = px.scatter(self.df, x="BMI", y="Age", color="Outcome",
                                     size="Glucose", hover_data=["BloodPressure", "DiabetesPedigreeFunction"],
                                     color_discrete_map={0: "#3B82F6", 1: "#EF4444"})
            scatter_fig.update_layout(title="BMI vs Age Analysis")
            plots['scatter_plot'] = scatter_fig.to_json()

            # 3. Feature Correlation Heatmap
            corr_matrix = self.df.corr().round(2)
            heatmap = px.imshow(corr_matrix,
                                color_continuous_scale="RdBu",
                                aspect="auto")
            heatmap.update_layout(title="Feature Correlation Matrix")
            plots['heatmap_plot'] = heatmap.to_json()

            # 4. Feature Importance Plot
            importance = self.get_feature_importance()
            imp_fig = px.bar(x=list(importance.keys()),
                             y=list(importance.values()),
                             title="Feature Importance in Prediction")
            plots['importance_plot'] = imp_fig.to_json()

            # 5. Age Distribution by Outcome
            age_dist = px.violin(self.df, x="Outcome", y="Age", box=True, points="all",
                                 color="Outcome", color_discrete_map={0: "#3B82F6", 1: "#EF4444"})
            age_dist.update_layout(title="Age Distribution by Diabetes Status")
            plots['age_dist_plot'] = age_dist.to_json()

            return plots
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def get_summary_stats(self):
        try:
            stats = {
                'total_patients': len(self.df),
                'diabetic_patients': len(self.df[self.df['Outcome'] == 1]),
                'diabetic_percentage': round(len(self.df[self.df['Outcome'] == 1]) / len(self.df) * 100, 1),
                'avg_glucose': round(self.df['Glucose'].mean(), 1),
                'avg_bmi': round(self.df['BMI'].mean(), 1),
                'avg_age': round(self.df['Age'].mean(), 1),
                'high_risk_patients': len(self.df[
                                              (self.df['Glucose'] > 140) &
                                              (self.df['BMI'] > 30) &
                                              (self.df['Age'] > 40)
                                              ]),
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return stats
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            raise

    def generate_recommendations(self, probability):
        if probability < 30:
            return "Your risk of diabetes is low. It's important to maintain a healthy lifestyle with regular exercise and a balanced diet. Continue monitoring your health and stay active."
        elif 30 <= probability < 70:
            return "You have a moderate risk of diabetes. It's advisable to adopt a healthier lifestyle, including a balanced diet, regular physical activity, and monitoring your glucose levels. Consider regular check-ups with your doctor."
        else:
            return "You have a high risk of diabetes. We strongly recommend consulting with a healthcare professional for further assessment and potential interventions. A medical professional can provide you with a personalized plan for managing your health."


analyzer = DiabetesAnalyzer('diabetes.csv')


@app.route('/')
def dashboard():
    try:
        plots = analyzer.create_visualizations()
        stats = analyzer.get_summary_stats()
        return render_template('dashboard.html', plots=plots, stats=stats)
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        return render_template('error.html', error=str(e))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        patient_data = pd.DataFrame([data])
        probability = analyzer.predict_probability(patient_data)

        # Generate recommendations based on risk level
        recommendations = analyzer.generate_recommendations(probability)

        return jsonify({'probability': probability, 'recommendations': recommendations})
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
