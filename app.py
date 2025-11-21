import streamlit as st
import pandas as pd
import joblib
import base64
import os
from groq import Groq

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- 0. BACKGROUND & STYLING ---
def add_bg_and_style(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read())
        
        bg_css = f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(bg_css, unsafe_allow_html=True)

    style_css = """
    <style>
    div[data-testid="stForm"] {
        background-color: rgba(20, 20, 20, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    h1, h2, h3, h4, p, label {
        color: white !important;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.8);
    }
    div[data-testid="stMetricValue"] {
        color: #ffbd45 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 20, 0.95);
    }
    </style>
    """
    st.markdown(style_css, unsafe_allow_html=True)

add_bg_and_style('photo.jpg')

# --- 1. LOAD ML MODELS ---
@st.cache_resource
def load_models():
    try:
        grade_models = joblib.load('multi_level_grade_models.pkl')
        risk_models = joblib.load('multi_level_risk_models.pkl')
        level_courses = joblib.load('course_structure.pkl')
        return grade_models, risk_models, level_courses
    except FileNotFoundError:
        st.error("❌ Critical Error: Model files not found.")
        st.stop()

models, risk_models, level_courses = load_models()

# --- 2. HELPER FUNCTIONS ---
def get_grade_point(letter):
    mapping = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0, "Not Taken": 0}
    return mapping[letter]

def get_grade_letter(num):
    mapping = {5: 'A', 4: 'B', 3: 'C', 2: 'D', 1: 'E', 0: 'F'}
    return mapping.get(int(round(num)), "F")

# --- 3. GROQ API INTEGRATION (FIXED) ---
def get_ai_recommendation(api_key, risk_status, predicted_grades, student_history):
    if not api_key:
        return "⚠️ Please enter a Groq API Key in the sidebar to generate recommendations."
    
    try:
        # Initialize Groq Client
        client = Groq(api_key=api_key)
        
        prompt = f"""
        Act as an empathetic Academic Advisor.
        
        Context:
        A student has just predicted their performance for the next academic session.
        - Predicted Risk Level: {risk_status}
        - Predicted Grades: {predicted_grades}
        - Past Grades: {student_history}
        
        Task:
        1. Identify the main weak point.
        2. Provide 3 short, actionable study tips to improve specific courses.
        3. Write a motivating closing statement.
        
        Keep it concise (under 150 words).
        """
        
        # Updated to the latest Llama 3.3 model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {str(e)}"

# --- 4. SIDEBAR ---
st.sidebar.title("🎓 Student Portal")

# Logic: Check Streamlit Secrets first (for Online), otherwise ask User (for Local)
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", help="Get a free key from console.groq.com")

st.sidebar.divider()

current_level = st.sidebar.selectbox(
    "I have completed results for:",
    options=[100, 200, 300],
    format_func=lambda x: f"{x} Level"
)
target_level = current_level + 100

st.sidebar.info(f"Predicting: **{target_level} Level**")
# --- 5. MAIN FORM ---
st.title(f"📊 Academic Predictor: {target_level} Level")
st.markdown("Enter your past grades. The AI will forecast your next session.")

input_data = {}
history_summary = {}

with st.form("transcript_form"):
    required_levels = [L for L in [100, 200, 300] if L <= current_level]
    
    for level in required_levels:
        st.subheader(f"📝 {level} Level Results")
        courses = level_courses.get(level, [])
        
        if not courses:
            st.warning(f"No courses found for {level} Level.")
            continue
        
        cols = st.columns(3)
        for i, course in enumerate(courses):
            col = cols[i % 3] 
            with col:
                grade = st.selectbox(f"{course}", ["A", "B", "C", "D", "E", "F", "Not Taken"], index=6, key=f"g_{course}")
                input_data[course] = get_grade_point(grade)
                if grade != "Not Taken":
                    history_summary[course] = grade
    
    st.markdown("---")
    submitted = st.form_submit_button(f"🚀 Analyze Performance", use_container_width=True)

# --- 6. PREDICTION & AI RESULTS ---
if submitted:
    if target_level not in models:
        st.error(f"⚠️ No models for {target_level} Level.")
    else:
        student_df = pd.DataFrame([input_data])
        
        # --- A. RISK ---
        st.divider()
        st.subheader("1️⃣ Risk Assessment")
        risk_label = "Unknown"
        
        if target_level in risk_models:
            risk_engine = risk_models[target_level]
            req_cols = risk_engine.feature_names_in_
            model_input = student_df.reindex(columns=req_cols, fill_value=0)
            
            risk_prob = risk_engine.predict_proba(model_input)[0][1]
            risk_percent = risk_prob * 100
            
            c1, c2 = st.columns([1, 3])
            with c1:
                if risk_prob > 0.5:
                    st.error("⚠️ HIGH RISK")
                    risk_label = "High Risk"
                else:
                    st.success("✅ SAFE")
                    risk_label = "Safe"
            with c2:
                st.progress(int(risk_percent))
                st.caption(f"Risk Probability: {risk_percent:.1f}%")
        
        # --- B. GRADES ---
        st.subheader("2️⃣ Predicted Grades")
        target_courses = models[target_level]
        predictions = []
        pred_summary = {}
        
        for course, model in target_courses.items():
            req_cols = model.feature_names_in_
            model_input = student_df.reindex(columns=req_cols, fill_value=0)
            pred_num = model.predict(model_input)[0]
            pred_letter = get_grade_letter(pred_num)
            
            status = "Pass" if pred_num >= 2 else "Fail"
            predictions.append({"Course": course, "Grade": pred_letter, "Status": status})
            pred_summary[course] = pred_letter

        if predictions:
            res_df = pd.DataFrame(predictions)
            def style_status(val):
                return f'background-color: {"#ff4b4b" if val == "Fail" else "#1c83e1"}'
            st.dataframe(res_df.style.applymap(style_status, subset=['Status']), use_container_width=True, hide_index=True)
            
            # --- C. GROQ AI RECOMMENDATION ---
            st.divider()
            st.subheader("🤖 AI Academic Advisor (Powered by Groq)")
            
            with st.spinner("Generating study plan..."):
                recommendation = get_ai_recommendation(api_key, risk_label, pred_summary, history_summary)
                
            st.info(recommendation)
        else:
            st.warning("No predictions available.")