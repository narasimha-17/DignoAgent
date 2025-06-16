import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from langchain_core.messages import HumanMessage
import os
import streamlit as st
from langchain_openai import ChatOpenAI
import google.generativeai as genai
import streamlit.components.v1 as components
import speech_recognition as sr
import graphviz






genai.configure(api_key="AIzaSyDCToalcS0jGdZyNiFxRnJOnDkoWCYd6zA")

system_prompt = """
you are advanced AI Medical Image Analysis Assistant, you will be given a medical image and you will provide a detailed analysis of the image, including any abnormalities or conditions that may be present. You will also provide a differential diagnosis and any recommendations for further testing or treatment. Your analysis should be thorough and based on the latest medical knowledge and guidelines.

your responsibilities include:
1. Analyze the medical image provided by the user.
2. Identify any abnormalities or conditions present in the image.
3. Provide a exact diagnosis based on the analysis.
4.you are a Master doctor, who has MBBS, MD, and PhD in medicine.
5. Ensure that your analysis is thorough and based on the latest medical knowledge and guidelines.
6. Maintain a professional and empathetic tone in your responses.
7. Provide clear and concise explanations of your findings.
8. Where possible, provide Confidence score.
9. Integrate symptoms, patient history, lab results, and imaging.
10. specific Disease Detection:
Detect potential diseases using visual and clinical data
Focus on relevant feature.
Clearly distinguish between possibilities, likelihood, and certainty.
11. Detailed Image Explanation
Interpret medical images (X-rays, MRIs, CTs, pathology slides, etc.)
Describe anatomical landmarks, abnormalities, and findings
Offer clinical explanations
Use precise terminology but avoid fear-inducing phrasing
12.provide a comprehensive report about furture steps and recommendations for the patient based on the analysis.
13.provide medications.
14 generate a mindmap for treatment plan as image.
15. Provide a detailed report about the analysis and recommendations.
16.note that you are not a medical professional and your analysis is based on the information provided by the user. You should always recommend that the user consult a qualified medical professional for any medical concerns or questions.
17.consider symptoms provided by the user in the analysis.
18.remove disclaimer from the response.

"""
generation_config = {
    "temperature":1,
    "top_p": 0.95,
    "top_k": 50,
    
}

def create_mind_map():
    dot = graphviz.Digraph()
    dot.node("Diagnosis")
    return dot



safety_settings = {
    "safety_settings": [
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_AND_WARN",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_AND_WARN",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_AND_WARN",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_AND_WARN",
        },
         
    ]
}
 #layout


st.set_page_config(page_title="Medical Image Analysis", layout="wide",page_icon=":hospital:robot:")
col1,col2,col3 = st.columns([1,2,1])
with col2:
    #st.image("hospital_robot.png", use_column_width=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Medical Image Analysis</h1>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            color: #000000;
        }
        .stButton > button {
            background-color: #0D6EFD;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stFileUploader label {
            font-weight: bold;
        }
                .stFileUploader label {
        font-weight: bold;
        color: #0D6EFD; /* Bootstrap primary blue */
        font-size: 18px;
    }

    /* Style the uploader area */
    .stFileUploader div[data-baseweb="file-uploader"] {
        border: 2px dashed #0D6EFD;
        background-color: #F9FAFB;
        padding: 10px;
        border-radius: 10px;
    }

    /* Style the uploaded file name text */
    .stFileUploader .uploadedFileName {
        color: #198754;
        font-weight: 600;
    }

    /* Customize the submit button too for consistency */
    .stButton > button {
        background-color: #0D6EFD;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
          
                
                

    </style>
""", unsafe_allow_html=True)
    





uploaded_file = st.file_uploader("Upload a medical image (JPEG, PNG, DICOM, etc.)", type=["jpg", "jpeg", "png", "dicom"])


record_voice = st.button("Record Symptoms")

voice_text = ""
if record_voice:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("Recording... Speak now.")
        audio = recognizer.listen(source, timeout=5)
    try:
        voice_text = recognizer.recognize_google(audio)
        st.success(f"🗣️ Recognized: {voice_text}")
    except sr.UnknownValueError:
        st.error("Could not understand your voice.")
    except sr.RequestError:
        st.error("Speech recognition service failed.")



mindmap_prompt = f"""
Generate a mind map in poster format for the following diagnosis.
Include sections for:
- Root Node (Diagnosis)
- Causes
- Symptoms
- Complications
- Treatments
- Follow-ups

Format it using indentation or tree-like branches.
"""



submit_button = st.button("Generate Image Analysis")

if submit_button :
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Medical Image", width=350)


    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
    )

    prompt_parts=[
        image,
        system_prompt,
        voice_text,
        mindmap_prompt,
    ]
       

    # Initialize the Gemini model before using it
    
    response = model.generate_content(prompt_parts)
    print(response.text)

    st.write(response.text)


    