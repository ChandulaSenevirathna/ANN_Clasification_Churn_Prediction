# Customer Churn Prediction using Artificial Neural Networks

This project implements a customer churn prediction model using an Artificial Neural Network (ANN). The Streamlit app provides an intuitive interface for users to input customer details and receive predictions about the likelihood of churn.

## Getting Started

Follow the steps below to set up and run the project on your local machine.

### Step 1: Clone the Repository
Clone the project repository from GitHub:
```bash
git clone https://github.com/ChandulaSenevirathna/ANN_Clasification_Churn_Prediction.git
```

### Step 2: Install Required Dependencies
Navigate to the project directory and install the necessary Python packages:
```bash
cd ANN_Clasification_Churn_Prediction
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
Start the Streamlit app to interact with the churn prediction model:
```bash
streamlit run app.py
```
## Folder Structure
```
ANN_Clasification_Churn_Prediction/
├── app.py                 # Main application file
├── churn_model.h5         # Trained ANN model
├── label_encorder_gender.pkl   # Label encoder for gender
├── onehot_encorder_geo.pkl     # One-hot encoder for geography
├── scaler.pkl             # Scaler for input data normalization
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation (this file)
```

## Model Details
The model is trained on a dataset of customer information to predict the likelihood of churn. It uses the following inputs:
- Credit Score
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card (Yes/No)
- Is Active Member (Yes/No)
- Estimated Salary
- Geography

The model was built using TensorFlow and has been serialized into `churn_model.h5` for deployment.

## Technologies Used
- **Programming Language**: Python
- **Framework**: Streamlit for the web app
- **Machine Learning**: TensorFlow for the ANN model
- **Data Preprocessing**: pandas, scikit-learn