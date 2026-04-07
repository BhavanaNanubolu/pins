**🧠 Finding Psychological Instability using AI/ML**

**📌 Overview**

This project is a web-based application designed to detect psychological instability using Machine Learning and Natural Language Processing techniques. It analyzes user responses from structured forms and scenario-based inputs to assess mental health conditions and provide meaningful feedback.

**🚀 Features**
Multi-stage psychological assessment

Machine Learning-based prediction (Random Forest)

NLP-based text analysis (TF-IDF + Classification)

AI-powered chatbot for mental health support

SMS notification system using Twilio

User-friendly web interface using Flask

**🛠️ Technologies Used**
Python

Flask

Scikit-learn

Natural Language Processing (TF-IDF)

HTML, CSS

Google Generative AI API

Twilio API

**📂 Project Structure**
├── model.py

├── app.py

├── mental_health_rf_model.pkl

├── text_rf_model.pkl

├── tfidf_vectorizer.pkl

├── templates/

│   ├── login.html

│   ├── general.html

│   ├── hr.html

│   ├── scenario.html

│   ├── result.html

│   └── index.html

└── README.md

**⚙️ Installation & Setup**
Step 1: Install Python

Ensure Python 3.x is installed.
Download from: https://www.python.org

Step 2: Install Required Libraries

Install dependencies using:

pip install -r requirements.txt
Step 3: Run the Model
python model.py
Step 4: Configure API Keys

Edit app.py and add your credentials:

**# Google Generative AI**
genai.configure(api_key="YOUR_GOOGLE_API_KEY")

**# Twilio**
TWILIO_ACCOUNT_SID = "YOUR_ACCOUNT_SID"
TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN"
Step 5: Run the Application
python app.py
Step 6: Open in Browser

Go to:

http://127.0.0.1:5000
**🔐 Test Credentials**
Mobile Number: phone_number(XXXXXXXXXX)
Password: password123
🧠 How It Works
🔹 Step 1: User Input

User logs in and completes a 3-stage assessment:

General Information
HR-Based Questions
Scenario-Based Questions
🔹 Step 2: Model Processing
HR Inputs → Label Encoding → Random Forest Model
Scenario Text → TF-IDF → Text Classification Model
Combined output determines mental state
🔹 Step 3: Result & Feedback
Prediction displayed on result page
SMS sent to user via Twilio
Optional chatbot support available
🤖 Chatbot Feature

Users can interact with an AI-powered chatbot for mental health support by visiting:

/index
**🎯 Outcome**
This system helps in early detection of psychological instability by analyzing behavioral patterns and textual inputs, enabling proactive mental health support.

**⚠️ Disclaimer**
This project is for educational purposes only and should not be considered a substitute for professional medical advice.
