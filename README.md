# CancerRAGent
- [Demonstration Video](https://youtu.be/I_HzKWnsHb4)
- [Demo website](https://cancerragent.github.io/)
- Download Faiss vector from this [link](https://drive.google.com/drive/folders/138WAux_fSDoYtT_kbFMxlHZa9vt-mbat?usp=sharing) and put under folder "INDEX"
- python version: 3.9.23
- install dependency: pip install -r requirements.txt
- run frontend by: streamlit run end2end.py
- run backend by: uvicorn server_llm_fastapi:app --host 0.0.0.0 --port 8000
