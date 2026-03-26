import streamlit as st
import io
from fpdf import FPDF
import requests
import json

# ===== STREAMLIT APP CONFIG =====
st.set_page_config(
    page_title="🧬 CancerRAGent: Cancer QA Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# FASTAPI_URL = "http://150.65.183.91:8000/ask"   # Đổi lại nếu server FastAPI đặt chỗ khác
FASTAPI_URL = "http://localhost:8000/ask"   # Đổi lại nếu server FastAPI đặt chỗ khác
# FASTAPI_URL = "http://192.168.10.62:8000/ask"   # Đổi lại nếu server FastAPI đặt chỗ khác

def ask_pipeline_api(question):
    try:
        response = requests.post(FASTAPI_URL, json={"question": question}, timeout=240)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# ===== SESSION STATE FOR HISTORY =====
if "history" not in st.session_state:
    st.session_state["history"] = []

# ===== HEADER =====
st.markdown(
    "<h2 style='text-align: center; color: #a64ac9;'>🧬 CancerRAGent: Cancer QA Assistant</h2>",
    unsafe_allow_html=True
)
st.markdown("Ask any cancer-related medical question: symptoms, diagnosis, treatment, or prognosis.")

# ===== MAIN USER INPUT =====
question = st.text_input("Enter your medical question", placeholder="e.g. What are the early symptoms of lung cancer?")
ask_button = st.button("Ask")

# ===== HANDLE NORMAL Q&A AND RED FLAG INSIDE PIPELINE =====
if ask_button and question.strip():
    with st.spinner("🤖 Generating answer..."):
        response = ask_pipeline_api(question)
        if "error" in response:
            st.error(f"Error: {response['error']}")
            st.stop()

    if response.get("red_flag"):
        st.markdown(
            """
            <div style='
                background-color: #ff4d4f;
                color: white;
                padding: 24px 20px;
                border-radius: 12px;
                text-align: center;
                font-size: 1.3em;
                font-weight: bold;
                margin-bottom: 16px;
                border: 2px solid #b80000;
                box-shadow: 0 0 16px #ffcccc;
            '>
                🚨 <b>This may be a medical emergency!</b> <br>
                Please seek immediate medical attention or call emergency services.
            </div>
            """, unsafe_allow_html=True
        )
        st.stop()

    # ===== Save to history =====
    st.session_state["history"].append({
        "question": question,
        "answer": response["answer"]
    })

    # ===== Show answer =====
    st.subheader("📝 Answer")
    st.markdown(
        f"""
        <div style='
            font-size: 1.08em;
            color: #bbbbbb;
            background-color: #22242a;
            padding: 16px 18px 14px 18px;
            border-radius: 8px;
            word-wrap: break-word;
            margin-bottom: 10px;
        '>{response["answer"]}
        <br><br>
        <i>Note: This information is for reference only. Please consult your physician for personal advice.</i>
        </div>
        """, unsafe_allow_html=True
    )

    guided_path = response['guided_path']
    # ===== Roadmap/Guided Exploration Path =====
    st.markdown(
        f"**Recommended learning step for you:** <span style='color:#56ffe8; background:#222b33; padding:4px 10px; border-radius:7px'>{guided_path}</span>",
        unsafe_allow_html=True,
    )

    # ===== Next questions (clickable chips/buttons) =====
    if response.get("next_questions"):
        st.markdown("#### 🧩 Related questions you may ask next:")
        st.markdown(response["next_questions"])

# ===== HANDLE PENDING QUESTION FROM GUIDED PATH/NEXT QUESTION =====
if "pending_question" in st.session_state:
    st.markdown(f"**Follow-up query:** {st.session_state['pending_question']}")

# ===== SHOW HISTORY =====
st.markdown("### 🗂️ Your Question & Answer History:")
for idx, item in enumerate(st.session_state["history"]):
    st.markdown(
        f"""
        <div style='margin-bottom:12px;'>
            <span style='color:#888;font-size:1em;'><b>Q{idx+1}:</b> {item['question']}</span><br>
            <span style='color:#d0d0d0;font-size:0.97em;padding-left:12px;'><b>A:</b> {item['answer']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== PDF EXPORT =====
def save_history_to_pdf(history):
    pdf = FPDF()
    pdf.add_page()
    # pdf.set_font("Arial", size=12)
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=14)
    for idx, item in enumerate(history):
        pdf.cell(0, 10, f"Q{idx+1}: {item['question']}", ln=1)
        pdf.multi_cell(0, 10, f"A: {item['answer']}")
        # pdf.ln(2)
        pdf.ln()
    # pdf_buffer = io.BytesIO()
    # # pdf.output(pdf_buffer)
    # pdf.output("Medical_QA_History.pdf")
    # pdf_buffer.seek(0)
    # return pdf_buffer
    # pdf_bytes = pdf.output(dest='S').encode('latin1', 'replace')
    pdf_bytes = pdf.output(dest='S')
    # pdf_bytes = pdf.output(dest='S').encode('utf-8')
    if isinstance(pdf_bytes, str):
        # nếu là str thì mã hoá utf-8 (hiếm khi cần nếu fpdf2 đúng version)
        pdf_bytes = pdf_bytes.encode("utf-8")
    if isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)
    return pdf_bytesppp

st.download_button(
        label="Download Medical QA History (PDF)",
        data=save_history_to_pdf(st.session_state["history"]),
        file_name="Medical_QA_History.pdf",
        mime="application/pdf"
    )

# if st.button("📄 Export Q&A History (PDF)"):
#     pdf_file = save_history_to_pdf(st.session_state["history"])
#     st.download_button(
#         label="Download Medical QA History (PDF)",
#         data=pdf_file,
#         file_name="Medical_QA_History.pdf",
#         mime="application/pdf"
#     )
# streamlit run end2end.py

# 6: bond0: <BROADCAST,MULTICAST,MASTER,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
#     link/ether 08:c0:eb:c1:ba:67 brd ff:ff:ff:ff:ff:ff
#     inet 150.65.183.91/23 brd 150.65.183.255 scope global bond0
#        valid_lft forever preferred_lft forever
#     inet6 fe80::ac0:ebff:fec1:ba67/64 scope link
#        valid_lft forever preferred_lft forever
# 7: idrac: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UNKNOWN group default qlen 1000
#     link/ether ec:2a:72:09:4b:5d brd ff:ff:ff:ff:ff:ff
#     inet 169.254.0.2/16 brd 169.254.255.255 scope global idrac
#        valid_lft forever preferred_lft forever
#     inet6 fde1:53ba:e9a0:de11:ee2a:72ff:fe09:4b5d/64 scope global dynamic mngtmpaddr  
#        valid_lft 86377sec preferred_lft 14377sec
#     inet6 fe80::ee2a:72ff:fe09:4b5d/64 scope link
#        valid_lft forever preferred_lft forever