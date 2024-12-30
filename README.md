# Diabetes Prediction and Visualization Dashboard

This project is a **Flask-based Diabetes Prediction Dashboard** that allows users to visualize diabetes-related data, predict diabetes risk based on patient data, and gain insights through interactive charts.

---

## 📋 **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Run the Application](#run-the-application)
6. [API Endpoints](#api-endpoints)
7. [Folder Structure](#folder-structure)
8. [Usage](#usage)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)

---

## 🚀 **Project Overview**
The **Diabetes Prediction Dashboard** utilizes machine learning and data visualization tools to predict diabetes risk based on patient data. It also displays key insights through interactive graphs and provides recommendations based on risk levels.

---

## 🌟 **Features**
- **Data Visualization:** Interactive plots for glucose distribution, BMI vs Age analysis, and more.
- **Diabetes Risk Prediction:** Predict diabetes probability using Logistic Regression.
- **Recommendations:** Personalized health recommendations based on prediction results.
- **Interactive Dashboard:** User-friendly interface to explore data insights.

---

## 🛠️ **Technologies Used**
- **Python:** Flask, Pandas, NumPy, Scikit-learn
- **Visualization:** Plotly
- **Backend Framework:** Flask
- **Deployment:** Localhost (Development Server)

---

## ⚙️ **Setup and Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/varun021/Diabetes-Prediction-Mode.git
cd diabetes-dashboard
```

### **2. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **3. Install Required Packages**
```bash
pip install -r requirements.txt
```

### **4. Add Dataset**
- Place your `diabetes.csv` dataset in the project root directory.

**Sample Dataset Format:**
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

### **5. Run Migrations (if applicable)**
```bash
flask db init
flask db migrate -m "Initial migration."
flask db upgrade
```

---

## ▶️ **Run the Application**
```bash
python app.py
```
Visit `http://127.0.0.1:5000/` in your browser.

---

## 🔗 **API Endpoints**

### **1. Dashboard**
- **URL:** `/`
- **Method:** GET
- **Description:** Renders the interactive dashboard with visualizations.

### **2. Predict Diabetes Risk**
- **URL:** `/predict`
- **Method:** POST
- **Description:** Predict diabetes risk based on patient data.
- **Sample Request:**
```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 25.5,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 30
}
```
- **Response:**
```json
{
  "probability": 75.5,
  "recommendations": "You have a high risk of diabetes..."
}
```

---

## 📂 **Folder Structure**
```
├── app.py
├── diabetes.csv
├── templates/
│   ├── dashboard.html
│   ├── error.html
├── static/
│   ├── style.css
├── venv/
├── requirements.txt
└── README.md
```

---

## 📊 **Usage**
- Access the dashboard at `http://127.0.0.1:5000/`
- Upload patient data and predict diabetes risk.
- Explore interactive charts for better insights.

---

## 🚀 **Future Improvements**
- Support for more machine-learning models.
- User authentication.
- Deployment on cloud platforms.

---

## 🤝 **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and test thoroughly.
4. Submit a pull request.

---

## 📜 **License**
This project is licensed under the MIT License.

---

## 📞 **Contact**
For any inquiries or feedback, feel free to reach out!

**Author:** Varun Nair
**Email:** nvarun018@gmail.com
**LinkedIn:** [Your LinkedIn Profile]([https://www.linkedin.com/in/varun-nair-504616238/])

---

Happy Coding! 🎯

