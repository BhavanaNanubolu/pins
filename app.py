from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from twilio.rest import Client  # Import Twilio client
import pandas as pd
import pickle
from email.mime.text import MIMEText
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
import re
import string
import google.generativeai as genai


app = Flask(__name__)
app.secret_key = 'your_secret_key'

genai.configure(api_key="AIzaSyAvYBlBirKb2oLJaIfMGGmQ7mzQZj2RF9o")

# Twilio configuration
TWILIO_ACCOUNT_SID = "ACc766e28466e702bc7b2cc72724b94c23"
TWILIO_AUTH_TOKEN = "cb03dcd44c39f2a1a5593920e3fbe688"
TWILIO_PHONE_NUMBER = "+1 256 474 7417"

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load the trained model
with open('mental_health_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load label encoders
with open("label_encoders.pkl", "rb") as le_file:
    label_encoders = pickle.load(le_file)

# Load a sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')
from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)


# Dummy users for login
users = {'7680844295': '1234567890', "1234567890": "1234567890","9121065009": "1234567890","79959 03496":"1234567890","7842173752":"1234567890","93810 42683":"1234567890","99999 99999":"1234567890"}

def preprocess_input(data):
    """Preprocess user input to match model requirements."""
    # Convert input data to a DataFrame
    df = pd.DataFrame([data])
    
    # Rename columns to match the feature names used during training
    feature_mapping = {
        'Age': 'Age',
        'Gender': 'Gender',
        'Sleep': 'Sleep',
        'Self_Employed': 'Self Employed',
        'Financial_Stress': 'Financial Stress',
        'Work_Pressure': 'Work Pressure',
        'Mood_Swings': 'Mood Swings',
        'Family_History': 'Family History',
        'Alcohol_Drug': 'Alcohol & Drug Consumption',
        'Friendly_CoWorkers': 'Friendly Co-Workers',
        'Physical_Health': 'Physical Health (1-10)',
        'Mental_Health': 'Mental Health (1-10)',
        'Work_Problems_Physical': 'Work Problems (Physical)',
        'Work_Problems_Emotional': 'Work Problems (Emotional)',
        'Felt_Low': 'Felt Low for a Week',
        'Diet_Changes': 'Diet Changes',
        'Last_Time_Happy': 'Last Time Happy',
        'Mental_Disorder_Diagnosis': 'Mental Disorder Diagnosis',
        'Family_History_Mental_Disorder': 'Family History of Mental Disorder',
        'Quality_of_Sleep': 'Quality of Sleep',
        'Drink_Smoke_Frequency': 'Drink/Smoke Frequency',
        'Tough_Emotional_Situation': 'Tough Emotional Situation',
        'Easily_Distracted': 'Easily Distracted',
        'Self_Critical': 'Self-Critical',
        'Headaches_Aches': 'Headaches/Aches',
        'Loss_of_Interest': 'Loss of Interest',
        'Feeling_Angry_Irritable': 'Feeling Angry/Irritable',
        'Feeling_Bad_About_Self': 'Feeling Bad About Self',
        'Anxiety_Phobias': 'Anxiety or Phobias',
        'Mood_Swings1': 'Mood Swings.1',
        'Support_from_Others': 'Support from Others',
        'Life_Satisfaction': 'Life Satisfaction',
        'Guilt_Regret': 'Guilt/Regret',
        'Meeting_Expectations': 'Meeting Expectations'
    }
    df.rename(columns=feature_mapping, inplace=True)

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen values gracefully
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Ensure all features match model training
    df = df[model.feature_names_in_]  # Reorder columns

    print("Test data columns:", df.columns)
    print("Model expects:", model.feature_names_in_)

    # Add missing features with default values
    '''for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0  # Default value (adjust as needed)'''
    
    # Validate that all required features are present
    missing_features = set(model.feature_names_in_) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    return df




@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        mobile = request.form.get('mobile')
        password = request.form.get('password')
        if not mobile or not password:
            return render_template('login.html', error="Please provide both mobile number and password")
        if mobile in users and users[mobile] == password:
            session['user'] = mobile
            return redirect(url_for('general_questions'))
        else:
            flash("Invalid mobile number or password", "error")
            return render_template('login.html', error="Invalid mobile number or password")
    return render_template('login.html')

@app.route('/general', methods=['GET', 'POST'])
def general_questions():
    if request.method == 'POST':
        session['general'] = request.form.to_dict()
        return redirect(url_for('hr_questions'))
    return render_template('general.html')

@app.route('/hr', methods=['GET', 'POST'])
def hr_questions():
    if request.method == 'POST':
        session['hr'] = request.form.to_dict()
        return redirect(url_for('result'))
    return render_template('hr.html')


# Route to render the scenario-based questions form
@app.route('/scenario')
def scenario():
    return render_template('scenario.html')

# Route to handle form submission
@app.route('/submit_scenario', methods=['POST'])
def submit_scenario():
    # Get the user's responses from the form
    question1 = request.form.get('question1')
    question2 = request.form.get('question2')
    question3 = request.form.get('question3')
    question4 = request.form.get('question4')
    question5 = request.form.get('question5')


    # Combine the responses into a single text input
    combined_text = f"{question1} {question2} {question3}{question4}{question5}"

    # Preprocess the text (same preprocessing as in m.py)
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text

    # Preprocess the combined text
    processed_text = preprocess_text(combined_text)
    
    # Load the vectorizer and model
    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    with open("text_rf_model.pkl", "rb") as model_file:
        rf_model = pickle.load(model_file)

    
    # Vectorize the processed text
    text_tfidf = vectorizer.transform([processed_text])

    # Predict the mental health status
    prediction = rf_model.predict(text_tfidf)[0]

    # Map the prediction back to the label
    reverse_label_map = {0:"Anxiety",1:"Normal",2:"Depression",3:"Suicidal",4:"Stress",5:"Bipolar",6:"Personality disorder"}
    predicted_status = reverse_label_map[prediction]

    # Return the result as a JSON response
    return jsonify({
        "Predicted Status": predicted_status
    })

@app.route('/result')
def result():
    user_data = {**session.get('general', {}), **session.get('hr', {})}
    processed_data = preprocess_input(user_data)
    prediction = model.predict(processed_data)[0]
    
    result_text = 'No treatment needed' if prediction == 1 else 'Treatment needed'
    
    # Send feedback to the user's registered mobile number
    user_mobile = session.get('user')  # Get the logged-in user's mobile number
    send_sms(user_mobile, f"Your mental health assessment result: {result_text}.Thankyou for attending the test.Have a good day!")
    
    return render_template('result.html', result=result_text)

def send_sms(to, message):
    """
    Send an SMS to the user's mobile number using Twilio.
    """
    try:
        # Ensure the phone number is in E.164 format
        if not to.startswith("+"):
            to = f"+91{to}"  # Add the country code for India if not present

        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to
        )
        print(f"SMS sent to {to}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

# Prompt to guide the chatbot to respond with health-related information
HEALTH_PROMPT = """
You are a highly knowledgeable health assistant. Your role is to provide accurate, helpful, 
and reliable information only related to health(physical,mental), wellness, nutrition, medicine, mental health, 
and fitness.If user says their result as Treatment needed that indicates they are mentally instable so,you should give advice to them about mental health remedies.Do not respond to any query that is not related to health. give response in 2-3 lines
"""

def get_chatbot_response(user_input):
    
    """
    Generate a response from the chatbot.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"{HEALTH_PROMPT}\nUser: {user_input}\nAssistant:")
    return response.text.strip()  # Return the text from the model

@app.route("/index")
def index():
    """
    Serve the frontend HTML page (index.html).
    """
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle chatbot queries from the frontend.
    """
    data = request.json  # Get the incoming data as JSON
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Get the response from the chatbot logic
    ai_response = get_chatbot_response(user_input)
    
    return jsonify({"response": ai_response})  # Return the chatbot's response in JSON format

@app.route("/dashboard")
def dashboard():
    return "<h1>Welcome to the Dashboard!</h1>"

if __name__ == '__main__':
    app.run(debug=True)
