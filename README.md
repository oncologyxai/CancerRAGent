# CancerRAGent
- Download Faiss vector from this [link](https://drive.google.com/drive/folders/138WAux_fSDoYtT_kbFMxlHZa9vt-mbat?usp=sharing) and put under folder "INDEX"
- python version: 3.9.23
- install dependency: pip install -r requirements.txt
- run frontend by: streamlit run end2end.py
- run backend by: uvicorn server_llm_fastapi:app --host 0.0.0.0 --port 8000
# Note
- For the incorporation with a ollama/vllm server, import PipelineRunner from file multi_retrieval_riken_vllm.py
- For loading model locally, import PipelineRunner from file multi_retrieval_riken_CoT.py