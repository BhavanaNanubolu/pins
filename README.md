﻿# Finding-psychological-instability-
Step 1: Install Python 
Ensure that Python 3.x is installed on your system. 
Download from the official website: https://www.python.org 
Step 2: Install Required Python Libraries 
Mentioned in softwares file. 
Step 3: Run model 
python model.py 
Step 4: Configure API Keys 
Edit app.py to include the appropriate credentials: 
Google Generative AI Key: 
genai.configure(api_key="YOUR_GOOGLE_API_KEY") 
Twilio Credentials (for SMS feedback): 
TWILIO_ACCOUNT_SID = "YOUR_ACCOUNT_SID" 
TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN" 
Step 5: Organize HTML Template Files 
Create a folder named templates/ and place the following HTML files inside it: - - - 
login.html 
general.html 
hr.html 
- - - 
scenario.html 
result.html 
index.html (chatbot interface) 
Step 6: Run the Flask Application 
Start the application using the following command: 
python app.py 
Step 7: Open the Web Application Open a web browser and go to: 
http://127.0.0.1:5000 
Step 8: Log in Using Dummy Credentials 
Use test credentials (example): 
Mobile Number: 7680844295 
Password: password123 
Step 9: Complete the Multi-Stage Form 
Proceed through the three-step testing interface: 
1. 
2. 
3. 
General Information Page 
HR-Based Questions Page - Analyzed using Random Forest model 
Scenario-Based Questions Page - Processed using NLP and classification 
Step 10: Model Execution and Prediction 
HR inputs -> label encoders -> mental_health_rf_model.pkl 
Scenario text -> tfidf_vectorizer.pkl -> text_rf_model.pkl 
Combined prediction assesses user's mental state 
Step 11: Feedback and Notification 
Final prediction is shown on the Result Page 
An SMS containing the result is sent to the user's registered number via Twilio 
Step 12: Access Mental Health Chatbot (Optional) 
Visit /index to interact with the Generative AI-powered chatbot for mental health support and advice.
