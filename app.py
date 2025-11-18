import streamlit as st
import easyocr
import google.generativeai as genai
from PIL import Image
import numpy as np
import tempfile
import os
from dotenv import load_dotenv
import os
import google.generativeai as genai
import cv2
load_dotenv()  # load key from .env

API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)


# ------------------------------------------------------
# 🔑 GEMINI API CONFIG
# ------------------------------------------------------
model = genai.GenerativeModel("gemini-2.0-flash")

# OCR Reader
reader = easyocr.Reader(['en'])

# ------------------------------------------------------
# 🔍 OCR FUNCTIONS
# ------------------------------------------------------
import fitz  # PyMuPDF
import tempfile

def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.getvalue()  # FIXED

    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    all_text = []

    for page in pdf:
        pix = page.get_pixmap(dpi=150)  # FIXED
        img_bytes = pix.tobytes("png")

        img_array = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(img_array, 1)  # FIXED

        result = reader.readtext(img_np, detail=0)
        all_text.append("\n".join(result))

    return "\n\n".join(all_text)



def extract_text_from_image(image):
    img_array = np.array(image)
    result = reader.readtext(img_array, detail=0)
    return "\n".join(result)


# ------------------------------------------------------
# 🤖 AI DIAGNOSIS
# ------------------------------------------------------
def diagnose_text(report_text):
    prompt = f"""
    You are a highly accurate medical analysis system specializing in interpreting lab reports.
    Your objective is to give the **best possible medical summary**, with clarity, correctness, and clinical relevance.
    You must avoid unnecessary storytelling, emotional tone, or extra disclaimers.

    🔍 **RULES FOR OUTPUT**
    - Be medically precise, concise, and easy to understand.
    - Do NOT add extra paragraphs outside the required format.
    - Keep every section balanced: not too long, not too short.
    - Do NOT repeat values from the report unless necessary.
    - Do NOT add warnings like “consult a doctor” unless the values truly indicate risk.
    - Focus ONLY on what the lab report logically indicates.
    - If values are normal → clearly say NORMAL, but still give useful health advice.
    - If values are abnormal → explain the *most likely medical meaning*.

    ## 🩺 Disease / Condition
    - Name
    - 2–3 line explanation

    ## ⚠️ Precautions
    - 4 practical safety steps

    ## 🍎 Diet Plan
    - Foods to eat
    - Foods to avoid
    - Daily nutrition guide

    ## 🏃 Exercise Plan
    - 3 exercises (with duration & weekly frequency)

    Lab Report:
    {report_text}
    """

    response = model.generate_content(prompt)
    return response.text


# ------------------------------------------------------
# 🎨 UI CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Medical Report Analyzer", page_icon="🧪", layout="centered")

st.markdown("""
    <style>
        .main {
            background: #f4f9ff;
        }
        .title-box {
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(120deg, #0077ff, #00c6ff);
            color: white;
            text-align: center;
            margin-bottom: 25px;
        }
        .upload-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .result-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
            margin-top: 20px;
        }
        .stButton button {
            border-radius: 10px;
            background-color: #0077ff;
            color: white;
            font-size: 18px;
            padding: 10px 18px;
        }
        .stButton button:hover {
            background-color: #005fcc;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 📌 UI ELEMENTS
# ------------------------------------------------------
st.markdown("<div class='title-box'><h1>🧪 Medical Report Analyzer</h1><p>Upload a lab report (Image or PDF) to get diagnosis, precautions, diet & exercise plan.</p></div>", unsafe_allow_html=True)

st.markdown("<div class='upload-box'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Choose Lab Report (Image or PDF)", type=["png", "jpg", "jpeg", "pdf"])

diagnosis = None

if uploaded_file:
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        st.write("📄 **PDF detected**")

        if st.button("🔍 Analyze Report"):
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)

            with st.spinner("Generating medical diagnosis..."):
                diagnosis = diagnose_text(text)

    else:
        # IMAGE
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Report", use_container_width=True)

        if st.button("🔍 Analyze Report"):
            with st.spinner("Extracting text..."):
                text = extract_text_from_image(image)

            with st.spinner("Generating medical diagnosis..."):
                diagnosis = diagnose_text(text)

    st.markdown("</div>", unsafe_allow_html=True)

    if diagnosis:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        st.subheader("🩺 Diagnosis Result")
        st.markdown(diagnosis)

        st.download_button(
            label="⬇ Download Diagnosis",
            data=diagnosis,
            file_name="diagnosis.txt",
            mime="text/plain"
        )

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("</div>", unsafe_allow_html=True)
    st.info("⬆ Upload a lab report to begin.")


