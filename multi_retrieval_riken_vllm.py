from email import generator
from operator import is_
import os
import json
import faiss
from narwhals import String
import numpy as np
from sklearn.metrics import recall_score
import torch
import re
import heapq
import pickle
from rank_bm25 import BM25Okapi
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from nltk.translate.meteor_score import meteor_score
import bert_score
import pandas as pd
import time
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from huggingface_hub import login
from rapidfuzz import fuzz, process
import io
from fpdf import FPDF
import nltk
nltk.download('wordnet')
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_URL= "http://192.168.10.62:11434/v1/chat/completions"

import requests
def query_vllm(prompt: str,
               model: str = "default",
               server_url: str = OLLAMA_URL) -> str:
    """
    Send a prompt to a local vLLM server and return the response.

    Args:
        prompt (str): The input text prompt.
        model (str): The model name registered on vLLM (default: "default").
        server_url (str): The vLLM server URL (default: http://localhost:8000/v1/completions).

    Returns:
        str: The generated text from the model.
    """
    headers = {"Content-Type": "application/json"}
    # FOR VLLM setup
    # payload = {
    #     "model": model_name,
    #     "prompt": prompt,
    #     "max_tokens": 512
    # }

    # FOR OLLAMA setup
    model_name="qwen2.5:7b"
    prompt_data = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    payload = {
        "model": model_name,
        "messages": prompt_data,
        "max_tokens": 512
    }

    response = requests.post(server_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")

    data = response.json()
    response = data["choices"][0]["message"]["content"].strip()
    return response




MAX_LENGTH_FINAL_ANSWER = 256
MAX_LENGTH_SUB_QUESTION = 128
MAX_LENGTH_SUB_ANSWER = 192
MAX_LENGTH_NEXT_QUESTION = 128
MAX_LENGTH_EVALUATION = 128

# =========================
# Hybrid Retriever (BM25 + Dense FAISS)
# =========================
class HybridRetriever:
    def __init__(self, index_dir, tokenizer_bge, model_bge, device='cpu', alpha=0.7):
        self.index_dir = index_dir
        self.tokenizer = tokenizer_bge
        self.model = model_bge
        self.device = device
        self.alpha = alpha

        print("\n[Retriever] Loading metadata...")
        self.metadatas = self.load_or_build_metadata()

        print("[Retriever] Loading FAISS index...")
        self.faiss_index = self.construct_index_BGE(self.metadatas)

        print("[Retriever] Setting up BM25...")
        self.bm25 = self.setup_bm25(self.metadatas)

    def filter_cancer_related_documents(self, dataset):
        cancer_keywords = ["cancer", "tumor", "oncology", "carcinoma", "neoplasm"]
        def is_cancer_related(title, content):
            text = (title + "\n" + content).lower()
            return any(keyword in text for keyword in cancer_keywords)
        cancer_documents = [
            {"title": doc["title"], "content": doc["content"]}
            for doc in dataset
            if is_cancer_related(doc["title"], doc["content"])
        ]
        print(f"Filtered {len(cancer_documents)} cancer-related documents.")
        return cancer_documents

    def save_metadata(self, metadata, output_path):
        with open(output_path, "w") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")

    def load_or_build_metadata(self):
        metadata_path = os.path.join(self.index_dir, "pubmed_metadatas.jsonl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        print("[Retriever] Building metadata...")
        dataset = load_dataset("MedRAG/pubmed", split="train")
        cancer_documents = self.filter_cancer_related_documents(dataset)
        os.makedirs(self.index_dir, exist_ok=True)
        self.save_metadata(cancer_documents, metadata_path)
        print(f"[Retriever] Metadata saved to {metadata_path}.")
        return [json.dumps(item) for item in cancer_documents]

    def setup_bm25(self, sentences):
        bm25_path = os.path.join(self.index_dir, "pubmed_bm25.pkl")
        if os.path.exists(bm25_path):
            print(f"[Retriever] Loading BM25 from {bm25_path}")
            with open(bm25_path, "rb") as f:
                return pickle.load(f)
        print("[Retriever] Building BM25 index from scratch...")
        def process_line(line):
            try:
                doc = json.loads(line)
                return (doc['title'] + "\n" + doc['content']).split()
            except:
                return None
        tokenized = []
        for i, sentence in enumerate(sentences):
            processed = process_line(sentence)
            if processed: tokenized.append(processed)
        print(f"[Retriever] BM25 index built with {len(tokenized)} documents.")
        bm25 = BM25Okapi(tokenized)
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
            print(f"[Retriever] BM25 saved to {bm25_path}")
        return bm25

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings_bge_m3(self, texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"[Retriever] Processing batch {i // batch_size + 1}/{len(texts) // batch_size + 1}...")
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=4096, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def construct_index_BGE(self, metadatas, batch_size=32):
        index_path = os.path.join(self.index_dir, "pubmed_faiss_BGE_M3.index")
        if os.path.exists(index_path):
            print("[Retriever] FAISS index loaded from disk.")
            return faiss.read_index(index_path)
        print("[Retriever] Building FAISS index...")
        def parse_metadata(line):
            try:
                doc = json.loads(line)
                return doc['title'] + "\n" + doc['content']
            except Exception as e:
                return None
        num_workers = multiprocessing.cpu_count()
        print(f"[Retriever] Using {num_workers} CPU cores for JSON parsing")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            parsed_data = list(executor.map(parse_metadata, metadatas))
        data = [d for d in parsed_data if d is not None]
        print(f"[Retriever] Total valid documents to embed: {len(data)}")
        embeddings = self.get_embeddings_bge_m3(data, batch_size=batch_size)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        print(f"[Retriever] FAISS index saved to {index_path}.")
        return index

    def retrieve(self, query, top_k=10):
        print(f"Query: {query}\n")
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        query_embedding = self.get_embeddings_bge_m3([query]).reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        faiss_scores = {idx: 1 / (1 + dist) for idx, dist in zip(indices[0], distances[0])}
        bm25_scores = {i: bm25_scores[i] for i in range(len(self.metadatas))}
        common_indices = set(bm25_scores.keys()).intersection(faiss_scores.keys())
        hybrid_scores = {
            i: self.alpha * bm25_scores[i] + (1 - self.alpha) * faiss_scores[i]
            for i in common_indices
        }
        top_k_indices = heapq.nlargest(top_k, hybrid_scores, key=hybrid_scores.get)
        return [json.loads(self.metadatas[i])['title'] + "\n" + json.loads(self.metadatas[i])['content'] for i in top_k_indices]

# =========================
# Cross-Encoder Reranker
# =========================
cache_dir  = "/SSD_data1/shared/huggingface_models/cache"
cache_dir = ""
device = None or ('cuda' if torch.cuda.is_available() else 'cpu')

class CrossEncoderReranker:
    def __init__(self, model_name="ncbi/MedCPT-Cross-Encoder", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if cache_dir == "":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True).to(device).eval()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True, cache_dir="/SSD_data1/shared/huggingface_models/cache").to(device).eval()


    def rerank(self, query, candidates, top_k=5):
        pairs = [(query, candidate) for candidate in candidates]
        encodings = self.tokenizer.batch_encode_plus(pairs, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            scores = self.model(**encodings).logits.squeeze(-1).cpu().numpy()
        scored_candidates = list(zip(candidates, scores))
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:top_k]

# =========================
# Verifier (Entailment Check)
# =========================
class AnswerVerifier:
    def __init__(self, model_name='facebook/bart-large-mnli', device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if cache_dir == "":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True).to(device).eval()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True, cache_dir="/SSD_data1/shared/huggingface_models/cache").to(device).eval()
        self.label_mapping = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def verify(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", padding=True,truncation=True).to(device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        prediction = probabilities.argmax()
        return self.label_mapping[prediction], float(probabilities[prediction])

RED_FLAG_KEYWORDS = [
        # General alarming symptoms
        "unintentional weight loss",
        "rapid weight loss",
        "persistent fever",
        "night sweats",
        "unexplained fatigue",
        "severe fatigue",
        "loss of appetite",
        "anorexia",
        "persistent pain",
        "new onset pain",
        "bone pain",
        "severe headache",
        "new lump",
        "rapidly growing lump",
        "hard mass",
        "fixed mass",
        "lymph node swelling",
        "enlarged lymph nodes",
        # Bleeding symptoms
        "blood in stool",
        "rectal bleeding",
        "black stool",
        "melena",
        "blood in urine",
        "hematuria",
        "coughing up blood",
        "hemoptysis",
        "vomiting blood",
        "hematemesis",
        "vaginal bleeding after menopause",
        "unusual vaginal bleeding",
        # Respiratory and swallowing
        "shortness of breath",
        "difficulty breathing",
        "new persistent cough",
        "cough lasting more than 3 weeks",
        "chest pain",
        "difficulty swallowing",
        "dysphagia",
        "hoarseness",
        # Neurological
        "new seizure",
        "confusion",
        "focal weakness",
        "paralysis",
        "sudden vision loss",
        "sudden hearing loss",
        # Gastrointestinal and hepatic
        "jaundice",
        "yellow skin",
        "yellow eyes",
        "severe abdominal pain",
        "ascites",
        # Urinary
        "difficulty urinating",
        "urinary retention",
        "blood in urine",
        "painful urination",
        # Breast cancer-specific
        "nipple discharge",
        "nipple inversion",
        "skin dimpling on breast",
        "redness of breast",
        "breast skin ulceration",
        # Colorectal/Prostate
        "change in bowel habits",
        "constipation alternating with diarrhea",
        "pencil-thin stool",
        "pelvic pain",
        # Liver
        "abdominal swelling",
        # Others (systemic)
        "persistent vomiting",
        "persistent diarrhea",
        "unexplained bruising",
        # Hemorrhage/Bleeding or abnormal discharges (see: [1,2])
        "coughing up blood",  # Hemoptysis (lung cancer, advanced GI)
        "vomiting blood",     # Hematemesis (GI cancer)
        "blood in urine",     # Hematuria (urinary tract, bladder, kidney cancer)
        "blood in stool",     # GI, colorectal cancer
        "rectal bleeding",
        "vaginal bleeding after menopause",  # Endometrial, cervical cancer
        "unexplained vaginal bleeding",
        "unexplained bleeding",
        "black stools",       # Melena (upper GI bleeding)
        "melena",
        "hematuria",
        "hematemesis",
        "hemoptysis",
        "bleeding gums",
        ""

        # Sudden, severe, or persistent pain (see: [1,3])
        "severe chest pain",
        "acute abdominal pain",
        "persistent abdominal pain",
        "severe headache",
        "new onset severe pain",
        "bone pain at night",   # bone, metastatic cancers
        "back pain with weight loss",  # pancreatic, metastatic cancers
        "persistent bone pain",
        "persistent headache",
        "new onset back pain",

        # Neurological dysfunction (see: [1,3,4])
        "loss of consciousness",
        "syncope",
        "paralysis",
        "numbness",
        "seizure",
        "difficulty walking",
        "difficulty speaking",
        "weakness on one side",
        "sudden vision loss",
        "double vision",
        "confusion",
        "memory loss",

        # Respiratory symptoms (see: [1,5])
        "difficulty breathing",
        "shortness of breath",
        "stridor",
        "persistent cough",
        "hoarseness",
        "wheezing not resolving",
        "chest tightness",
        "painful swallowing",

        # Jaundice and abnormal color changes 
        "sudden jaundice",
        "yellowing of skin",
        "yellowing of eyes",

        # Swallowing/GI symptoms (see: [1,2])
        "progressive difficulty swallowing",  # esophageal, head and neck cancer
        "dysphagia",
        "persistent vomiting",
        "inability to eat",
        "persistent nausea",

        # Unexplained weight/appetite loss (see: [1,3])
        "unexplained weight loss",
        "unintentional weight loss",
        "loss of appetite",
        "anorexia",

        # Rapidly growing or persistent lumps (see: [1,7])
        "rapidly growing lump",
        "painless lump in neck",
        "lump in breast with skin changes",
        "hard lump under skin",
        "new lump not resolving",

        # Fever, night sweats, severe fatigue (see: [1,8])
        "persistent fever",
        "night sweats",
        "unexplained fever",
        "persistent fatigue",
        "extreme fatigue",

        # Persistent GI or urinary changes (see: [1,2])
        "persistent diarrhea",
        "persistent constipation",
        "change in bowel habits",
        "persistent abdominal distension",
        "new onset urinary retention",

        # Skin/mucosa changes (see: [1,7])
        "non-healing ulcer",
        "bleeding mole",
        "change in mole appearance",
        "new skin lesion not healing",
        "unexplained bruising",
        "unusual skin rash",

        # Genitourinary abnormalities (see: [1,2])
        "testicular pain",
        "enlarged testicle",
        "painful urination",
        "blood in semen",
        "abnormal vaginal discharge",

        # Other severe or multisystem symptoms (see: [1])
        "spinal cord compression",
        "sudden hearing loss",
        "persistent hoarseness",
        "persistent unexplained cough",
        "fracture after minor injury",
        "sudden severe dizziness",
        "swelling of face or arms",

        # Visual/hearing disturbances (see: [1,3])
        "sudden blindness",
        "sudden deafness",
        "vision loss in one eye",

        # Cardiovascular events (see: [1])
        "unexplained chest pain",
        "palpitations with fainting",

        # Persistent urinary symptoms (see: [1,2])
        "persistent difficulty urinating",
        "urinary incontinence",
        "new urinary urgency",

        # Rapid general decline
        "multiple unexplained symptoms",
        "rapid decline in general health",
        
        ######
        # Key PubMed / Clinical References
        
        # NICE Guidelines – Suspected Cancer: Recognition and Referral
        # https://pubmed.ncbi.nlm.nih.gov/27929613/
        # [NICE guideline NG12 – UK National Institute for Health and Care Excellence]

        # Bowel cancer symptoms: UK NICE, BMJ
        # https://pubmed.ncbi.nlm.nih.gov/24874006/

        # Unexplained weight loss and cancer: British Journal of General Practice
        # https://pubmed.ncbi.nlm.nih.gov/24567621/

        # Red flag symptoms in patients with brain tumors: BMJ
        # https://pubmed.ncbi.nlm.nih.gov/29880419/

        # Red flags in lung cancer diagnosis: Cancer Research UK
        # https://pubmed.ncbi.nlm.nih.gov/30067550/

        # Obstructive jaundice as a presentation of cancer: World J Gastroenterol
        # https://pubmed.ncbi.nlm.nih.gov/25320533/

        # Red flag signs in skin and soft tissue tumors: JAMA Dermatology
        # https://pubmed.ncbi.nlm.nih.gov/29459933/

        # Night sweats, fever, and fatigue in lymphoma: Blood
        # https://pubmed.ncbi.nlm.nih.gov/26124478/
        ######
    ]

FUZZY_THRESHOLD = 85  # Set 80-90% as suitable threshold


###
# "Overview → Causes & Risk → Prevention & Screening → Symptoms & Concerns → Diagnosis → Staging → Prognosis → Treatment Planning → Treatment Options → Management During Treatment → Psychosocial/Practical → After Treatment → Advanced Disease → Additional Resources"

CANCER_PATHWAY_TEMPLATE = [
    # 1. Overview & Understanding
    ("Overview & Understanding", "Overview", "What is {cancer}?"),
    ("Overview & Understanding", "Types & Subtypes", "What are the different types or subtypes of {cancer}?"),
    ("Overview & Understanding", "Cancer vs. Benign Tumor", "How does {cancer} differ from benign tumors?"),

    # 2. Causes & Risk
    ("Causes & Risk", "Causes & Risk Factors", "What causes {cancer}? What are the main risk factors?"),
    ("Causes & Risk", "Genetics & Family History", "Is {cancer} hereditary or related to genetics?"),
    ("Causes & Risk", "Environmental & Occupational Risks", "Are there environmental or work-related risk factors for {cancer}?"),

    # 3. Prevention & Screening
    ("Prevention & Screening", "Prevention", "How can the risk of {cancer} be reduced or prevented?"),
    ("Prevention & Screening", "Vaccines & Prevention", "Are there vaccines or preventive measures available for {cancer}?"),
    ("Prevention & Screening", "Screening & Early Detection", "Are there screening tests for {cancer}? How can it be detected early?"),
    ("Prevention & Screening", "Who Should Get Screened?", "Who should get screened for {cancer}, and how often?"),

    # 4. Symptoms & Concerns
    ("Symptoms & Concerns", "Recognizing Signs", "What are the warning signs and symptoms of {cancer}?"),
    ("Symptoms & Concerns", "When to See a Doctor", "When should someone see a doctor about possible {cancer}?"),

    # 5. Diagnosis
    ("Diagnosis", "Diagnosis", "How is {cancer} diagnosed? What tests are needed?"),
    ("Diagnosis", "Second Opinions", "Should I get a second opinion for a {cancer} diagnosis or treatment?"),

    # 6. Staging & Classification
    ("Staging & Classification", "Staging & Classification", "How is {cancer} staged or classified? What do the different stages mean?"),

    # 7. Prognosis (Prognostic Factors)
    ("Prognosis", "Prognostic Factors", "What factors affect the prognosis of {cancer}?"),

    # 8. Treatment Planning
    ("Treatment Planning", "Treatment Planning", "How is the best treatment plan for {cancer} determined?"),
    ("Treatment Planning", "Multidisciplinary Care", "What is the role of a multidisciplinary team in treating {cancer}?"),
    ("Treatment Planning", "Standard Treatment Pathways", "What is the standard sequence of treatment steps for {cancer}?"),

    # 9. Treatment Options
    ("Treatment Options", "Treatment Options", "What are the treatment options for {cancer}?"),
    ("Treatment Options", "Surgery", "When is surgery recommended for {cancer}?"),
    ("Treatment Options", "Chemotherapy & Radiation", "What are the roles of chemotherapy and radiation in treating {cancer}?"),
    ("Treatment Options", "Targeted Therapy & Immunotherapy", "Are targeted therapies or immunotherapies available for {cancer}?"),
    ("Treatment Options", "Clinical Trials & New Treatments", "Are there ongoing clinical trials or new treatments for {cancer}?"),

    # 10. Management During Treatment
    ("Management During Treatment", "Side Effects Management", "What are common side effects of {cancer} treatment and how can they be managed?"),
    ("Management During Treatment", "Inpatient vs. Outpatient Care", "Will treatment for {cancer} require hospitalization or can it be done as an outpatient?"),
    ("Management During Treatment", "Coping with Treatment", "How can patients cope with treatment physically and emotionally?"),
    ("Management During Treatment", "Nutrition & Lifestyle", "What nutrition and lifestyle advice is important during {cancer} treatment?"),
    ("Management During Treatment", "Exercise & Physical Activity", "Is exercise safe during or after {cancer} treatment?"),
    ("Management During Treatment", "Work & Daily Activities", "How can {cancer} treatment affect my ability to work or do daily activities?"),
    ("Management During Treatment", "Supportive & Palliative Care", "What supportive and palliative care options exist for {cancer}?"),
    ("Management During Treatment", "Pain Management", "How can pain related to {cancer} or its treatment be managed?"),
    ("Management During Treatment", "Rehabilitation", "What rehabilitation options are available after {cancer} treatment?"),
    ("Management During Treatment", "Complementary & Integrative Therapies", "Are complementary or alternative therapies helpful for {cancer}?"),

    # 11. Psychosocial & Practical Support
    ("Psychosocial & Practical Support", "Emotional & Mental Health", "How does a {cancer} diagnosis affect mental health, and what support is available?"),
    ("Psychosocial & Practical Support", "Family & Caregiver Support", "What support and advice are available for families and caregivers of {cancer} patients?"),
    ("Psychosocial & Practical Support", "Financial & Legal Issues", "What financial, insurance, or legal considerations come with a {cancer} diagnosis?"),
    ("Psychosocial & Practical Support", "Workplace Rights", "What workplace rights and protections exist for people with {cancer}?"),
    ("Psychosocial & Practical Support", "Transportation & Access to Care", "How can patients with {cancer} get help with transportation or accessing treatment?"),
    ("Psychosocial & Practical Support", "Fertility & Sexual Health", "How can {cancer} or its treatment affect fertility and sexual health?"),
    ("Psychosocial & Practical Support", "Pregnancy & Cancer", "What should patients know about {cancer} during pregnancy or childbearing age?"),
    ("Psychosocial & Practical Support", "Cancer in Children/Teens", "How is {cancer} different in children or adolescents compared to adults?"),
    ("Psychosocial & Practical Support", "Older Adults & {cancer}", "What special considerations are there for older adults with {cancer}?"),
    ("Psychosocial & Practical Support", "Rare Cancers", "What should I know about rare forms of {cancer}?"),

    # 12. After Treatment: Survivorship & Monitoring
    ("After Treatment: Survivorship & Follow-up", "Survivorship & Life After Treatment", "What should survivors of {cancer} know about long-term follow-up and life after treatment?"),
    ("After Treatment: Survivorship & Follow-up", "Follow-up Care", "What follow-up is needed after {cancer} treatment? How often are check-ups needed?"),
    ("After Treatment: Survivorship & Follow-up", "Managing Recurrence", "What are the risks and signs of {cancer} recurrence? How is it managed?"),
    ("After Treatment: Survivorship & Follow-up", "Late Effects", "What are possible late or long-term effects of {cancer} and its treatment?"),

    # 13. Advanced Disease
    ("Advanced Disease", "Metastatic Disease", "What does it mean if {cancer} is metastatic? What are treatment options for advanced stages?"),

    # 14. Additional Resources & Community
    ("Additional Resources & Community", "Patient Stories & Community", "Where can patients find support groups or patient communities for {cancer}?"),
    ("Additional Resources & Community", "Questions to Ask the Doctor", "What questions should I ask my doctor about {cancer}?"),
    ("Additional Resources & Community", "Telemedicine & Virtual Care", "Can {cancer} care be provided through telemedicine or virtual visits?"),
    ("Additional Resources & Community", "Advance Directives & Planning", "What should patients with {cancer} know about advance care planning or directives?"),
    ("Additional Resources & Community", "Additional Resources", "Where can I find reliable information or resources about {cancer}?"),
]



CANCER_SYNONYMS = {
    "breast cancer": [
        "breast cancer", "mammary cancer", "mammary carcinoma", "carcinoma of the breast", "breast carcinoma", 
        "ductal carcinoma", "lobular carcinoma"
    ],
    "lung cancer": [
        "lung cancer", "pulmonary cancer", "pulmonary carcinoma", "nsclc", "non-small cell lung cancer",
        "non-small cell lung carcinoma", "sclc", "small cell lung cancer", "small cell lung carcinoma",
        "bronchogenic carcinoma", "lung neoplasm"
    ],
    "colorectal cancer": [
        "colorectal cancer", "colon cancer", "rectal cancer", "colon carcinoma", "rectal carcinoma",
        "colorectal carcinoma", "colon neoplasm", "rectal neoplasm", "bowel cancer"
    ],
    "prostate cancer": [
        "prostate cancer", "prostatic cancer", "prostate carcinoma", "carcinoma of prostate", "prostatic adenocarcinoma"
    ],
    "liver cancer": [
        "liver cancer", "hepatocellular carcinoma", "hcc", "hepatic cancer", "liver carcinoma", "hepatoma", "primary liver cancer"
    ],
    "stomach cancer": [
        "stomach cancer", "gastric cancer", "gastric carcinoma", "stomach carcinoma", "carcinoma of the stomach"
    ],
    "pancreatic cancer": [
        "pancreatic cancer", "pancreas cancer", "pancreatic carcinoma", "pancreatic adenocarcinoma", "pancreas carcinoma"
    ],
    "esophageal cancer": [
        "esophageal cancer", "esophagus cancer", "esophageal carcinoma", "carcinoma of the esophagus", 
        "esophageal squamous cell carcinoma", "esophageal adenocarcinoma"
    ],
    "bladder cancer": [
        "bladder cancer", "bladder carcinoma", "urothelial carcinoma", "transitional cell carcinoma", "bladder neoplasm"
    ],
    "kidney cancer": [
        "kidney cancer", "renal cell carcinoma", "rcc", "kidney carcinoma", "renal carcinoma", "renal cancer", "renal neoplasm"
    ],
    "thyroid cancer": [
        "thyroid cancer", "thyroid carcinoma", "thyroid neoplasm", "papillary thyroid carcinoma", 
        "follicular thyroid carcinoma", "medullary thyroid carcinoma", "anaplastic thyroid carcinoma"
    ],
    "ovarian cancer": [
        "ovarian cancer", "ovary cancer", "ovarian carcinoma", "carcinoma of the ovary", "epithelial ovarian cancer"
    ],
    "cervical cancer": [
        "cervical cancer", "cervix cancer", "cervical carcinoma", "carcinoma of the cervix", "cervical neoplasm"
    ],
    "endometrial cancer": [
        "endometrial cancer", "uterine cancer", "endometrial carcinoma", "carcinoma of the uterus", "uterine carcinoma"
    ],
    "testicular cancer": [
        "testicular cancer", "testis cancer", "testicular carcinoma", "germ cell tumor", "seminoma", "nonseminoma"
    ],
    "oral cancer": [
        "oral cancer", "mouth cancer", "oral cavity cancer", "oral carcinoma", "oral squamous cell carcinoma"
    ],
    "nasopharyngeal cancer": [
        "nasopharyngeal cancer", "nasopharynx cancer", "nasopharyngeal carcinoma", "npc", "carcinoma of the nasopharynx"
    ],
    "skin cancer": [
        "skin cancer", "cutaneous cancer", "melanoma", "malignant melanoma", "non-melanoma skin cancer", "basal cell carcinoma",
        "squamous cell carcinoma", "bcc", "scc", "cutaneous carcinoma"
    ],
    "head and neck cancer": [
        "head and neck cancer", "head and neck carcinoma", "oral cavity cancer", "pharyngeal cancer", "laryngeal cancer"
    ],
    "brain cancer": [
        "brain cancer", "brain tumor", "glioblastoma", "glioblastoma multiforme", "gbm", "astrocytoma",
        "meningioma", "oligodendroglioma", "medulloblastoma", "brain neoplasm", "primary brain tumor"
    ],
    "bone cancer": [
        "bone cancer", "bone tumor", "sarcoma", "osteosarcoma", "ewing sarcoma", "rhabdomyosarcoma", 
        "chondrosarcoma", "primary bone tumor"
    ],
    "leukemia": [
        "leukemia", "aml", "acute myeloid leukemia", "cml", "chronic myeloid leukemia",
        "all", "acute lymphoblastic leukemia", "cll", "chronic lymphocytic leukemia",
        "acute promyelocytic leukemia", "apl", "acute monocytic leukemia", "t-cell leukemia", "b-cell leukemia"
    ],
    "lymphoma": [
        "lymphoma", "hodgkin lymphoma", "hodgkin's disease", "non-hodgkin lymphoma", "nhl", "burkitt lymphoma",
        "primary central nervous system lymphoma", "cutaneous lymphoma", "diffuse large b-cell lymphoma",
        "dlbcl", "follicular lymphoma", "mantle cell lymphoma", "anaplastic large cell lymphoma"
    ],
    "multiple myeloma": [
        "multiple myeloma", "plasma cell myeloma", "myelomatosis", "kahler's disease"
    ],
    "mesothelioma": [
        "mesothelioma", "malignant mesothelioma", "pleural mesothelioma", "peritoneal mesothelioma"
    ],
    "gallbladder cancer": [
        "gallbladder cancer", "gallbladder carcinoma", "carcinoma of the gallbladder"
    ],
    "bile duct cancer": [
        "bile duct cancer", "cholangiocarcinoma", "bile duct carcinoma", "extrahepatic cholangiocarcinoma", "intrahepatic cholangiocarcinoma"
    ],
    "anal cancer": [
        "anal cancer", "anal carcinoma", "carcinoma of the anus"
    ],
    "small intestine cancer": [
        "small intestine cancer", "small bowel cancer", "duodenal cancer", "jejunal cancer", "ileal cancer"
    ],
    "appendix cancer": [
        "appendix cancer", "appendiceal cancer", "appendix carcinoma", "appendiceal carcinoma"
    ],
    "adrenal gland cancer": [
        "adrenal gland cancer", "adrenal cancer", "adrenocortical carcinoma", "adrenal cortical carcinoma", "adrenal carcinoma"
    ],
    "penile cancer": [
        "penile cancer", "penis cancer", "penile carcinoma", "carcinoma of the penis"
    ],
    "vaginal cancer": [
        "vaginal cancer", "vagina cancer", "vaginal carcinoma", "carcinoma of the vagina"
    ],
    "vulvar cancer": [
        "vulvar cancer", "vulva cancer", "vulvar carcinoma", "carcinoma of the vulva"
    ],
    "salivary gland cancer": [
        "salivary gland cancer", "salivary cancer", "salivary gland carcinoma", "parotid gland cancer", "submandibular gland cancer"
    ],
    "eye cancer": [
        "eye cancer", "ocular cancer", "retinoblastoma", "uveal melanoma", "intraocular melanoma", "ocular melanoma"
    ],
    "thymus cancer": [
        "thymus cancer", "thymoma", "thymic carcinoma", "carcinoma of the thymus"
    ],
    "mediastinal tumor": [
        "mediastinal tumor", "mediastinal neoplasm", "mediastinal mass"
    ],
    "neuroblastoma": [
        "neuroblastoma", "neuroblastoma tumor", "nb"
    ],
    "wilms tumor": [
        "wilms tumor", "nephroblastoma", "wilms' tumor", "wilms nephroblastoma"
    ],
    "carcinoid tumor": [
        "carcinoid tumor", "neuroendocrine tumor", "net", "typical carcinoid", "atypical carcinoid", "carcinoid neoplasm"
    ],
    "gastrointestinal stromal tumor": [
        "gastrointestinal stromal tumor", "gist", "gastrointestinal stromal neoplasm"
    ],
    "plasma cell neoplasm": [
        "plasma cell neoplasm", "plasma cell disorder"
    ],
    "myelodysplastic syndrome": [
        "myelodysplastic syndrome", "mds", "preleukemia", "myelodysplasia"
    ],
    "myeloproliferative neoplasm": [
        "myeloproliferative neoplasm", "mpn", "myeloproliferative disorder"
    ],
    "chronic myelomonocytic leukemia": [
        "chronic myelomonocytic leukemia", "cmml"
    ],
    "primary peritoneal cancer": [
        "primary peritoneal cancer", "primary peritoneal carcinoma"
    ],
    "gestational trophoblastic disease": [
        "gestational trophoblastic disease", "gtd", "gestational trophoblastic neoplasia", "choriocarcinoma", "hydatidiform mole"
    ],
    "langerhans cell histiocytosis": [
        "langerhans cell histiocytosis", "lch", "histiocytosis x"
    ],
    # Additional rare or syndromic tumors:
    "pheochromocytoma": [
        "pheochromocytoma", "adrenal pheochromocytoma"
    ],
    "paraganglioma": [
        "paraganglioma", "extra-adrenal paraganglioma"
    ],
    "sarcoma": [
        "sarcoma", "soft tissue sarcoma", "liposarcoma", "leiomyosarcoma", "angiosarcoma"
    ],
    "desmoid tumor": [
        "desmoid tumor", "aggressive fibromatosis"
    ],
# Sarcoma subtypes (rare soft tissue/bone cancers)
    "angiosarcoma": [
        "angiosarcoma", "hemangiosarcoma", "vascular sarcoma"  # Rare blood vessel cancer
    ],
    "chordoma": [
        "chordoma", "notochordal sarcoma"  # Rare bone cancer (spine/skull base)
    ],
    "chondrosarcoma": [
        "chondrosarcoma", "cartilage cancer"  # Rare cartilage cancer
    ],
    "ewing sarcoma": [
        "ewing sarcoma", "ewing's sarcoma", "peripheral primitive neuroectodermal tumor", "ppnet"
    ],
    "synovial sarcoma": [
        "synovial sarcoma", "malignant synovioma"  # Rare soft tissue cancer
    ],
    "alveolar soft part sarcoma": [
        "alveolar soft part sarcoma", "asps"  # Very rare soft tissue sarcoma
    ],
    "desmoplastic small round cell tumor": [
        "desmoplastic small round cell tumor", "dsrct"  # Very rare, aggressive
    ],
    "clear cell sarcoma": [
        "clear cell sarcoma", "malignant melanoma of soft parts"
    ],
    # Pediatric/Childhood rare cancers
    "retinoblastoma": [
        "retinoblastoma", "rb", "malignant retinal tumor"
    ],
    "medulloblastoma": [
        "medulloblastoma", "primitive neuroectodermal tumor of the cerebellum", "pnet cerebellum"
    ],
    "atypical teratoid/rhabdoid tumor": [
        "atypical teratoid rhabdoid tumor", "atrt", "rhabdoid tumor"
    ],
    "pleuropulmonary blastoma": [
        "pleuropulmonary blastoma", "ppb"  # Rare pediatric lung cancer
    ],
    "hepatoblastoma": [
        "hepatoblastoma", "malignant liver tumor of childhood"
    ],
    "rhabdomyosarcoma": [
        "rhabdomyosarcoma", "embryonal rhabdomyosarcoma", "alveolar rhabdomyosarcoma", "pleomorphic rhabdomyosarcoma"
    ],
    # Rare CNS/brain tumors
    "oligodendroglioma": [
        "oligodendroglioma", "oligodendroglial tumor"
    ],
    "ependymoma": [
        "ependymoma", "ependymal tumor"
    ],
    "pineoblastoma": [
        "pineoblastoma", "pinealoblastoma"
    ],
    "choroid plexus carcinoma": [
        "choroid plexus carcinoma", "cpc"
    ],
    # Rare endocrine/neuroendocrine
    "adrenocortical carcinoma": [
        "adrenocortical carcinoma", "acc", "adrenal cortex cancer"
    ],
    "parathyroid carcinoma": [
        "parathyroid carcinoma", "parathyroid cancer"
    ],
    "insulinoma": [
        "insulinoma", "islet cell tumor"
    ],
    "glucagonoma": [
        "glucagonoma", "pancreatic alpha cell tumor"
    ],
    "pheochromocytoma": [
        "pheochromocytoma", "adrenal medulla tumor"
    ],
    "paraganglioma": [
        "paraganglioma", "extra-adrenal paraganglioma"
    ],
    # Rare germ cell/mixed tumors
    "yolk sac tumor": [
        "yolk sac tumor", "endodermal sinus tumor"
    ],
    "choriocarcinoma": [
        "choriocarcinoma", "gestational choriocarcinoma", "non-gestational choriocarcinoma"
    ],
    "embryonal carcinoma": [
        "embryonal carcinoma", "embryonal cell carcinoma"
    ],
    # Blood/lymph/immune rare cancers
    "hairy cell leukemia": [
        "hairy cell leukemia", "hcl"
    ],
    "waldenstrom macroglobulinemia": [
        "waldenstrom macroglobulinemia", "lymphoplasmacytic lymphoma"
    ],
    "sezary syndrome": [
        "sezary syndrome", "sezary disease", "cutaneous t-cell lymphoma"
    ],
    "mycosis fungoides": [
        "mycosis fungoides", "cutaneous t-cell lymphoma"
    ],
    # Other rare or site-specific cancers
    "merkel cell carcinoma": [
        "merkel cell carcinoma", "mcc", "primary neuroendocrine carcinoma of the skin"
    ],
    "esthesioneuroblastoma": [
        "esthesioneuroblastoma", "olfactory neuroblastoma"
    ],
    "adenoid cystic carcinoma": [
        "adenoid cystic carcinoma", "cylindroma"
    ],
    "sebaceous carcinoma": [
        "sebaceous carcinoma", "sebaceous gland carcinoma"
    ],
    "mucinous carcinoma": [
        "mucinous carcinoma", "colloid carcinoma"
    ],
    "fallopian tube cancer": [
        "fallopian tube cancer", "fallopian tube carcinoma"
    ],
    "primary peritoneal carcinoma": [
        "primary peritoneal carcinoma", "extraovarian peritoneal carcinoma"
    ],
    "appendiceal cancer": [
        "appendiceal cancer", "appendix cancer", "appendiceal carcinoma"
    ],
    "peritoneal mesothelioma": [
        "peritoneal mesothelioma", "malignant peritoneal mesothelioma"
    ],
    # Vascular/lymphatic/other rare tumors
    "kaposi sarcoma": [
        "kaposi sarcoma", "kaposi's sarcoma"
    ],
    "hemangioendothelioma": [
        "hemangioendothelioma", "epithelioid hemangioendothelioma"
    ],
        # Angiosarcoma: Rare, aggressive cancer of blood vessels
    "angiosarcoma": [
        "angiosarcoma", "hemangiosarcoma", "vascular sarcoma"
    ],  # A rare, aggressive cancer of blood vessel (vascular) tissue.

    # Chordoma: Rare bone tumor at base of skull or spine
    "chordoma": [
        "chordoma", "notochordal sarcoma"
    ],  # Rare bone cancer, usually at base of skull or spine.

    # Ewing sarcoma: Rare bone/soft tissue tumor in children/young adults
    "ewing sarcoma": [
        "ewing sarcoma", "ewing's sarcoma", "peripheral primitive neuroectodermal tumor", "ppnet"
    ],  # Rare bone or soft tissue cancer, mainly in children/young adults.

    # Adenoid cystic carcinoma: Rare salivary gland/head and neck tumor
    "adenoid cystic carcinoma": [
        "adenoid cystic carcinoma", "cylindroma"
    ],  # Rare tumor of salivary glands or head and neck.

    # Medulloblastoma: Pediatric malignant brain tumor
    "medulloblastoma": [
        "medulloblastoma", "primitive neuroectodermal tumor of the cerebellum", "pnet cerebellum"
    ],  # Common pediatric malignant brain tumor, rare overall.

    # Insulinoma: Rare pancreatic neuroendocrine tumor
    "insulinoma": [
        "insulinoma", "islet cell tumor"
    ],  # Rare pancreatic neuroendocrine tumor, can cause hypoglycemia.

    # Kaposi sarcoma: Vascular cancer, often in immunocompromised
    "kaposi sarcoma": [
        "kaposi sarcoma", "kaposi's sarcoma"
    ],  # Rare cancer seen often in immunocompromised (HIV/AIDS), involves blood/lymphatic vessels.
}

class LLMAnsweringQuestion:
    def __init__(self, model_name='Qwen/Qwen2.5-14B-Instruct', device=None, is_llama = False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_llama = is_llama
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model_name = model_name
        # if cache_dir == "":
        #     self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config).to(device).eval()
        # else:
        #     self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config, cache_dir="/SSD_data1/shared/huggingface_models/cache").to(device).eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.decompose_prompt_template = """
You are a helpful assistant designed to break down complex cancer-related medical queries.

Your task is to simplify a given question into multiple independent sub-questions. 
- If the input query contains multiple parts (e.g., different conditions, treatments, comparisons), separate each into its own question.
- If the query is already simple and focused on a single aspect, keep it as-is.

Follow the format below:

Example:
Query: What are the symptoms of lung cancer and how is it diagnosed?
Decomposed Questions:
- What are the symptoms of lung cancer?
- How is lung cancer diagnosed?

Query: What are the survival rates for breast cancer compared to lung cancer?
Decomposed Questions:
- What is the survival rate for breast cancer?
- What is the survival rate for lung cancer?

Query: How effective is chemotherapy for pancreatic cancer, and what are its side effects?
Decomposed Questions:
- How effective is chemotherapy for pancreatic cancer?
- What are the side effects of chemotherapy for pancreatic cancer?

Now process the following:
Query: {question}
Decomposed Questions:
<RESPONSE>
        """
        
        self.next_prediction_prompt_template = """
You are a helpful medical assistant. Your task is to suggest exactly one most relevant follow-up question that a user would naturally ask next, based on the type of the original question, the user's question, and the provided answer.

Inputs:
- Original question: {question}
- Final answer: {answer}
- Type of question: {question_type}  # (Type 1: Basic Information, Type 2: Decision-Oriented, Type 3: Personal Symptom Inquiry)

Instructions:
- Carefully consider the type and content of the original question as well as the final answer.
- For Type 1 (Basic Information), generate a follow-up that deepens understanding or explores related facts.
- For Type 2 (Decision-Oriented), do NOT offer further advice, recommendations, or clarify options. Instead, suggest a follow-up question that encourages users to discuss their situation with a healthcare provider, or invites them to consider any additional information they may want to share with their doctor.
- For Type 3 (Personal Symptom Inquiry), do NOT generate a follow-up question. Instead, clearly instruct the user that this could be an emergency and they must visit a hospital or healthcare provider immediately, as urgent care may be critical for survival. Do NOT provide a diagnosis or any personal medical advice. 
- Only generate exactly one medically relevant follow-up question, listed as a bullet point.
- The question should be clear, concise, and directly related to the prior exchange.

Follow-up Question or Instruction:
<RESPONSE>
"""

        self.generate_prompt_template = """
You are an AI assistant tasked with generating a well-structured and informative response.

Query: {query}

Relevant Information:
{context}

Instruction:
Using the provided relevant information, generate a comprehensive, accurate, and well-structured response to the query.
<RESPONSE>
        """
        
        self.check_red_flag_question_prompt = """
You are an experienced medical triage assistant.
Does the following question describe a possible urgent or life-threatening medical situation?
If YES, answer 'Yes' and clearly state which word(s) or phrase(s) in the question indicate an emergency. If NO, answer 'No' and briefly explain why this is not an emergency.

Examples:
Question: My sister is having severe chest pain and trouble breathing, what should she do?
Answer: Yes. The phrases "severe chest pain" and "trouble breathing" are emergency symptoms. Chest pain with respiratory distress in people over 40 is highlighted in NICE NG12 as urgent signs for suspected lung cancer or pulmonary disease.

Question: My brother is vomiting blood. Should he go to the hospital?
Answer: Yes. The phrase "vomiting blood" refers to hematemesis, which NICE NG12 describes as an alarm symptom requiring urgent GI investigation for possible oesophageal or stomach cancer.

Question: My friend has noticed blood in her stool for several days. Should she be worried?
Answer: Yes. "Blood in stool for several days" suggests rectal bleeding, which NICE NG12 specifically flags as a symptom needing urgent referral for colorectal cancer assessment.

Question: My sister has been losing weight rapidly without trying. Is this dangerous?
Answer: Yes. "Rapid, unintentional weight loss" is cited in NICE NG12 as a non-site-specific red flag warranting urgent cancer pathway referral.

Question: My aunt suddenly had a seizure and is confused. What should she do?
Answer: Yes. The phrases "suddenly had a seizure" and "is confused" match NICE NG12's criteria for progressive neurological deficit, which necessitates urgent imaging for possible brain tumour.

Question: Peter has had persistent hoarseness and difficulty swallowing over 3 weeks. Should he see a doctor urgently?
Answer: Yes. "Persistent hoarseness" and "difficulty swallowing" lasting > 3 weeks in someone over 45 are listed in NICE NG12 under head and neck cancers, indicating the need for urgent referral.

Question: My friend is feeling tired lately, is it serious?
Answer: No. Feeling tired is common and not necessarily an emergency unless it is severe or associated with other symptoms.

Question: What are the symptoms of colon cancer?
Answer: No. This is a general question about medical information, not an emergency situation.

Now analyze the following:
Question: {question}
Answer:
<RESPONSE>
        """

        self.type_question_template = """      
You are a careful and helpful medical assistant that classifies user questions into three types:

Type 1 (Basic Information): These questions ask for factual, descriptive, or definitional knowledge about diseases, symptoms, treatments, or medical concepts in general. These questions do NOT refer to the user's or their family member's personal health situation.
- Examples include:
    - What is liver cancer?
    - How many stages of liver cancer are there?
    - What are the symptoms of stomach cancer?
    - What are the symptoms of lung cancer?
    - What are risk factors for colon cancer?
    - What treatments are available for breast cancer?

Type 2 (Decision-Oriented): These questions ask for recommendations, comparisons, or judgment calls.
- These questions ask for recommendations, comparisons, or require making a judgment between options. These questions often include words like "best", "compare", "should", "better", or ask for a decision or a choice, comparing options, or seeking expert advice on what should be done.
- Examples include:
    - What is the best treatment for liver cancer?
    - Compare chemotherapy and immunotherapy for colon cancer.
    - Should a patient with stage II lung cancer undergo surgery?
    - Which is more effective: surgery or radiation for prostate cancer?

Type 3 (Personal Symptom Inquiry): These questions ask about symptoms or health concerns that the user or a specific person is experiencing.
- These questions ask specifically about the user's or a family member's symptoms, health experiences, or concerns. They usually refer to "I", "my", "we", "our", "my mother", "my father", "my brother", "my son", "my friend", etc., and often include phrases like "I have", "I am experiencing", "my father has", "should I be worried", "should he/she see a doctor", etc.
- Examples include:
    - I have had blood in my stool for a week, should I be worried?
    - My father has had a persistent cough for three weeks, should he see a doctor?
    - My mother is losing weight rapidly, is this a sign of cancer?
    - I have a lump in my breast, what should I do?
    - Our family member has night sweats, is it cancer?
    - I have had blood in my stool for a week, should I be worried?
    - My brother has had a persistent cough for three weeks, should he see a doctor?
    - My friend is losing weight rapidly, is this a sign of cancer?
    - I am experiencing chest pain and shortness of breath. What should I do?

Important:
- Only classify as Type 3 if the question is about a specific person's (the user's or someone they know) symptoms or experiences.
- Classify as Type 2 if the question asks for a decision, recommendation, or comparison.
- All other general questions, including those asking about symptoms in general (not about a specific person's situation), should be classified as Type 1. 

Now, classify the following question into one of the three types by answering with only the number:
Type: <1, 2, or 3>

Question: {question}
<RESPONSE>
        """
        
        self.information_query_template_zero_shot = """
You are a helpful medical assistant.

Original question: {question}

Your task:
- Carefully analyze the question and provide a single, comprehensive, and accurate answer.
- Base your answer on established medical knowledge.

Return only a complete answer, ending with a full sentence.

Final answer:
<RESPONSE>
"""

        self.decision_query_template_zero_shot = """
You are a helpful assistant designed to support medical decision-making by summarizing options, not making final recommendations.

Original question: {question}

Your task:
- Summarize the key medical information and considerations that would be useful for decision-making regarding this question.
- Do not make a specific recommendation or provide personal medical advice.
- At the end of your answer, include a disclaimer reminding users to consult a healthcare provider for diagnosis or treatment decisions.

Return only a complete answer, ending with a full sentence.

Final answer:
<RESPONSE>
"""

        self.symptom_query_template_zero_shot = """
You are a helpful assistant that responds to personal health or symptom questions with caution and care.

Original question: {question}

Your task:
- Provide general medical information related to the described symptoms, but do **not** attempt to diagnose, rule out, or recommend treatment for any specific individual.
- If the described symptoms may indicate a serious or urgent medical condition (such as sudden severe pain, bleeding, difficulty breathing, neurological changes, or other red-flag signs), clearly advise the user to seek immediate medical attention.
- Emphasize that online information cannot replace a professional medical evaluation.
- End your response with a clear, full-sentence warning to seek medical care immediately.

Return only a complete answer, ending with a full sentence.

Final answer:
<RESPONSE>
"""

        self.information_query_template = """
You are a helpful assistant that answers complex medical questions.

Original question: {question}

Below are the sub-questions and their respective answers:
{sub_questions_list}

{qa_context_pairs}

Before answering, carefully re-read the original question and all sub-questions to ensure your response is thorough and fully addresses every aspect of the user's inquiry.

Your task: 
- Synthesize all the information above and generate a single, comprehensive, and accurate answer for the original question.

Return only a complete answer, ending with a full sentence.

Final answer:
<RESPONSE>
        """

        self.decision_query_template = """
You are a helpful assistant designed to support medical decision-making by summarizing options, not making final recommendations.

Original question: {question}

Below are the sub-questions and their respective answers:
{sub_questions_list}

{qa_context_pairs}

Before answering, carefully re-read the original question and all sub-questions to ensure your response is thorough and addresses all relevant decision factors.

Your task: 
- Summarize the key medical information and considerations that would be useful for decision-making regarding this question. Do not make a specific recommendation or provide personal medical advice.
- At the end of your answer, include a disclaimer reminding users to consult a healthcare provider for diagnosis or treatment decisions.

Return only a complete answer, ending with a full sentence.

Final answer:
<RESPONSE>
        """
        
        self.symptom_query_template = """
You are a helpful assistant that responds to personal health or symptom questions with caution and care.

Original question: {question}

Below are the sub-questions and their respective answers:
{sub_questions_list}

{qa_context_pairs}

Before answering, carefully review the original question and all sub-questions. Your primary goal is to provide general medical information and context related to the described symptoms, but do **not** attempt to diagnose, rule out, or recommend treatment for any specific individual.

If the described symptoms may indicate a serious or urgent medical condition (such as sudden severe pain, bleeding, difficulty breathing, neurological changes, or other red-flag signs), clearly advise the user to seek immediate medical attention.

Your task: 
- Briefly explain that, while general information is provided for context, these symptoms require prompt evaluation by a medical professional.
- Emphasize state that online information cannot replace a professional medical evaluation and do not give a diagnosis or suggest home management.
- End your response with a clear, full-sentence warning to seek medical care immediately.

Return only a complete answer, ending with a full sentence.

Final answer:
<RESPONSE>
        """

        self.improvement_generation_answer_prompt = """
The following answer was previously provided but may not be clear or comprehensive enough.

Question: {question}
Previous Answer: {previous_answer}

Before answering, carefully re-read the original question and ensure that your improved answer fully addresses all aspects, with more detail, clarity, and accuracy.

Return only a complete answer, ending with a full sentence.

Improved Answer:
<RESPONSE>
        """
        
        self.verify_score_answer_prompt = """
You are an experienced medical assistant. Assess the confidence that the following answer is correct, relevant, and sufficiently supported for the given question.

Question:
{question}

Answer:
{answer}

On a scale from 0% (no confidence) to 100% (absolute confidence), what is your confidence score in this answer?

Only return a single integer value from 0 to 100. Do not explain your reasoning.
<RESPONSE>
        """

    def predict_next_question(self, complex_question, answer, decision, max_new_tokens=MAX_LENGTH_SUB_QUESTION):
        prompt = self.next_prediction_prompt_template.format(question = complex_question, answer=answer, question_type=decision)
        
        if self.is_llama == False:
            # input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            # outputs = self.model.generate(
            #     **input_ids,
            #     max_new_tokens=max_new_tokens,
            #     do_sample=False,
            # )
            # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response = query_vllm(prompt,model=self.model_name)
            match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
            if match:
                response_content = match.group(1).strip()
            else:
                response_content = response.split("<RESPONSE>")[1]
            follow_ups = response_content.strip()
            lines = follow_ups.split("\n")
            questions = [line.strip().removeprefix("- ").strip() for line in lines if line.strip()]
            unique_questions = list(set(questions))
            return unique_questions
        else:
            # messages = [
            #     {"role": "system", "content": "You are a medical expert."},
            #     {"role": "user", "content": prompt}
            # ]
            
            # llama_prompt = self.tokenizer.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )
            # inputs = self.tokenizer(
            #     llama_prompt, return_tensors="pt"
            # ).to(self.model.device)

            # with torch.no_grad():
            #     output = self.model.generate(
            #         **inputs,
            #         max_new_tokens=max_new_tokens,
            #         pad_token_id=self.tokenizer.eos_token_id,
            #         do_sample=False,
            #         temperature=0.01
            #     )

            # # Decode, skip input
            # response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            prompt = f"You are a medical expert. {prompt}"
            response = query_vllm(prompt,model=self.model_name)
            return response.strip()
            
    def decompose(self, complex_question, max_new_tokens=MAX_LENGTH_SUB_QUESTION):
        prompt = self.decompose_prompt_template.format(question=complex_question)
        


        if self.is_llama == False:
            # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            # outputs = self.model.generate(
            #     input_ids,
            #     max_new_tokens=max_new_tokens,
            #     do_sample=False
            # )
            # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = query_vllm(prompt,model=self.model_name)
            match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
            if match:
                response_content = match.group(1).strip()
            else:
                response_content = response.split("<RESPONSE>")[1]
            sub_questions = []
            pattern = r"Decomposed Questions:\s*([\s\S]*?)(?=\n\n|$)"
            match = re.search(pattern, response_content)
            if match:
                decomposed_questions_raw = match.group(1).strip()
                sub_questions = decomposed_questions_raw.strip().split("\n")
                sub_questions = [q.strip().removeprefix("- ") for q in sub_questions if q != ""]
            else:
                sub_questions.append(complex_question)
            return sub_questions
        else:
            messages = [
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": prompt}
            ]
            
            # llama_prompt = self.tokenizer.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )
            # inputs = self.tokenizer(
            #     llama_prompt, return_tensors="pt"
            # ).to(self.model.device)

            # with torch.no_grad():
            #     output = self.model.generate(
            #         **inputs,
            #         max_new_tokens=max_new_tokens,
            #         pad_token_id=self.tokenizer.eos_token_id,
            #         do_sample=False,
            #         temperature=0.01
            #     )

            # Decode, skip input
            # response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            prompt = f"You are a medical expert. {prompt}"
            response = query_vllm(prompt,model=self.model_name)
            if response.count("-") > 0:
                sub_questions = response.strip().split("\n")
                sub_questions = [q.strip().removeprefix("- ") for q in sub_questions if q != ""]
            else:
                sub_questions = [complex_question]
            return sub_questions

    def generate_sub_answers(self, query, context, prompt_template, max_new_tokens=MAX_LENGTH_SUB_ANSWER):
        prompt = prompt_template.format(query=query, context=context)
        if self.is_llama == False:
            # input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            # output = self.model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=False)

            # response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = query_vllm(prompt,model=self.model_name)
            match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
            if match:
                response_content = match.group(1).strip()
            else:
                response_content = response.split("<RESPONSE>")[1]
            return response_content.strip()
        else:
            # messages = [
            #     {"role": "system", "content": "You are a medical expert."},
            #     {"role": "user", "content": prompt}
            # ]
            
            # llama_prompt = self.tokenizer.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )
            # inputs = self.tokenizer(
            #     llama_prompt, return_tensors="pt"
            # ).to(self.model.device)

            # with torch.no_grad():
            #     output = self.model.generate(
            #         **inputs,
            #         max_new_tokens=max_new_tokens,
            #         pad_token_id=self.tokenizer.eos_token_id,
            #         do_sample=False,
            #         temperature=0.01
            #     )
            prompt = f"You are a medical expert. {prompt}"
            response = query_vllm(prompt,model=self.model_name)
            # Decode, skip input
            # response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
    
    def run_pipeline_for_question_with_improvement(self, question, previous_answer):
        prompt = self.improvement_generation_answer_prompt.format(query=question, context=previous_answer)
        return self.safe_generate_response(prompt, max_new_tokens=MAX_LENGTH_FINAL_ANSWER)
    
    def safe_generate_response(self, prompt, 
                          max_new_tokens=MAX_LENGTH_FINAL_ANSWER, 
                          max_tries=3, 
                          stop_tokens=None, 
                          end_punctuations=('.','!','?')):
        for attempt in range(max_tries):
            if self.is_llama == False:
                # input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                # output = self.model.generate(
                #     **input_ids,
                #     max_new_tokens=max_new_tokens,
                #     do_sample=False,
                #     eos_token_id=(stop_tokens if stop_tokens is not None else None)
                # )
                # response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                response = query_vllm(prompt,model=self.model_name)
                match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                else:
                    parts = response.split("<RESPONSE>")
                    answer = parts[1].strip() if len(parts) > 1 else response.strip()
                answer = answer.rstrip()
                if answer and answer[-1] in end_punctuations:
                    return answer
            else:
                # messages = [
                #     {"role": "system", "content": "You are a medical expert."},
                #     {"role": "user", "content": prompt}
                # ]
                
                # llama_prompt = self.tokenizer.apply_chat_template(
                #     messages, tokenize=False, add_generation_prompt=True
                # )
                # inputs = self.tokenizer(
                #     llama_prompt, return_tensors="pt"
                # ).to(self.model.device)

                # with torch.no_grad():
                #     output = self.model.generate(
                #         **inputs,
                #         max_new_tokens=max_new_tokens,
                #         pad_token_id=self.tokenizer.eos_token_id,
                #         do_sample=False,
                #         temperature=0.01
                #     )

                # # Decode, skip input
                # response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                prompt = f"You are a medical expert. {prompt}"
                response = query_vllm(prompt,model=self.model_name)
                answer = response.rstrip()
                if answer and answer[-1] in end_punctuations:
                    return answer
            max_new_tokens *= 1.25
        return answer

# =========================
# LLM as judge
# =========================
class LLMEvaluator:
    def __init__(self, model_name="ContactDoctor/Bio-Medical-Llama-3-8B", device=None, is_mistral=False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if cache_dir == "":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                quantization_config=quant_config,
            ).to(device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                quantization_config=quant_config,
                cache_dir="/SSD_data1/shared/huggingface_models/cache"
            ).to(device).eval()
            
        self.model_name = model_name
        self.is_mistral = is_mistral
        # Store all evaluations
        self.correctness_scores = []
        self.faithfulness_scores = []
        self.coherence_scores = []
        self.readability_scores = []
        self.entailment_values = []
        self.meteor_scores = []
        self.bertscore_scores = []
        
        self.correctness_question_scores = []
        self.faithfulness_question_scores = []
        self.coherence_question_scores = []
        self.readability_question_scores = []
        self.entailment_question_values = []
        self.meteor_question_scores = []
        self.bertscore_question_scores = []

    def evaluate_meteor(self, answer, gold):
        if gold:
            score = meteor_score([gold.split()], answer.split())
            return "meteor", score
        return "meteor", 0.0

    def evaluate_bertscore(self, answer, gold):
        if gold:
            P, R, F1 = bert_score.score([answer], [gold], lang='en', verbose=False)
            return "bertscore", F1.item()
        return "bertscore", 0.0

    def generate(self, prompt, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "You are an expert in medical domain."},
            {"role": "user", "content": f"{prompt}"}
        ]
        if not self.is_mistral:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
            terminators = [self.tokenizer.eos_token_id]
            generated_output = self.model.generate(input_ids, eos_token_id=terminators, max_new_tokens=max_new_tokens, do_sample=False)
            response = generated_output[0][input_ids.shape[-1]:]
            response_content = self.tokenizer.decode(response, skip_special_tokens=True)
            return response_content
        else:
            encoded_input = self.tokenizer(prompt, return_tensors='pt').to(device)
            output = self.model.generate(**encoded_input, max_new_tokens=max_new_tokens, do_sample=False)
            response_content = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response_content
        
    def extract_explanation_and_score(self, response):
        # Everything before 'Final rating' is explanation
        explanation = response.split("Final rating")[0].strip()
        score_match = re.search(r"Final rating[: ]*\s*([1-5])", response)
        score = int(score_match.group(1)) if score_match else 0
        # Clean up: If explanation is empty and only score was returned, fill with default text.
        if not explanation:
            explanation = "No explanation provided."
        return explanation, score
    
    ### Evaluate for generated answer
    def evaluate_correctness(self, question, answer, gold=None):
        if gold:
            band = """
    Score meanings:
    1 = Completely incorrect; the answer is factually wrong or contradicts established knowledge.
    2 = Mostly incorrect; significant factual errors.
    3 = Partially correct; contains both correct and incorrect information.
    4 = Mostly correct; minor factual inaccuracies or omissions.
    5 = Completely correct; factually accurate, comprehensive, and reliable.
    """
            prompt = f"""
    You are a medical expert evaluating the factual accuracy of a generated answer.

    Question: {question}
    Gold Answer: {gold}
    Generated Answer: {answer}

    First, explain step by step how factually correct the generated answer is, based on your medical knowledge.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
            """
        else:
            band = """
    Score meanings:
    1 = Completely incorrect; the answer is factually wrong or contradicts established knowledge.
    2 = Mostly incorrect; significant factual errors.
    3 = Partially correct; contains both correct and incorrect information.
    4 = Mostly correct; minor factual inaccuracies or omissions.
    5 = Completely correct; factually accurate, comprehensive, and reliable.
    """
            prompt = f"""
    You are a medical expert evaluating the factual accuracy of a generated answer.

    Question: {question}
    Generated Answer: {answer}

    First, explain step by step how factually correct the generated answer is, based on your medical knowledge.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
            """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("correctness", explanation, score)
    
    def evaluate_faithfulness(self, question, answer, gold=None):
        if gold:
            band = """
    Score meanings:
    1 = Completely unfaithful; the answer is mostly hallucinated or unsupported.
    2 = Mostly unfaithful; significant unsupported or hallucinated content.
    3 = Somewhat faithful; answer partially aligns but contains notable unsupported content.
    4 = Mostly faithful; minor unsupported or hallucinated elements.
    5 = Fully faithful; all information is supported by the gold answer.
    """
            prompt = f"""
    You are a medical evaluator assessing whether the generated answer includes any unsupported or hallucinated content.

    Question: {question}
    Gold Answer: {gold}
    Generated Answer: {answer}

    First, explain step by step how faithful the generated answer is to the gold answer, and whether it introduces any unsupported or hallucinated content.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
            """
        else:
            band = """
    Score meanings:
    1 = Completely unfaithful to the question; mostly hallucinated or unsupported.
    2 = Mostly unfaithful; significant unsupported or irrelevant information.
    3 = Somewhat faithful; partially addresses the question but with notable unsupported details.
    4 = Mostly faithful; minor unsupported elements.
    5 = Fully faithful; all information relevant and supported by the question.
    """
            prompt = f"""
    You are a medical evaluator assessing whether the generated answer includes any unsupported or hallucinated content.
    Question: {question}
    Generated Answer: {answer}

    First, explain step by step how faithful the generated answer is to the original question, and whether it introduces any unsupported or hallucinated content.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
            """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("faithfulness", explanation, score)


    def evaluate_coherence(self, answer):
        band = """
    Score meanings:
    1 = Not coherent; answer is disorganized or very hard to follow.
    2 = Mostly incoherent; frequent logical jumps or unclear structure.
    3 = Somewhat coherent; answer is understandable but contains some logical gaps or poor organization.
    4 = Mostly coherent; well-structured with minor clarity issues.
    5 = Highly coherent; logical, well-organized, and easy to follow.
    """
        prompt = f"""
    You are a medical evaluator evaluating the logical structure and clarity of this medical answer.

    Generated Answer: {answer}

    First, explain step by step how logically structured and clear the response is.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
        """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("coherence", explanation, score)


    def evaluate_readability(self, answer):
        band = """
    Score meanings:
    1 = Not readable; response is confusing, fragmented, or ungrammatical.
    2 = Mostly unreadable; difficult to read due to language or grammar problems.
    3 = Somewhat readable; partially understandable, but with noticeable awkwardness or mistakes.
    4 = Mostly readable; clear with minor language issues.
    5 = Very readable; fluent, natural, and easy to understand.
    """
        prompt = f"""
    You are a medical evaluator assessing the readability and fluency of this medical answer.

    Generated Answer: {answer}

    First, explain step by step how readable and fluent the response is.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
        """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("readability", explanation, score)


    def evaluate_relevance_next_question(self, original_question, generated_answer, next_question):
        band = """
    Score meanings:
    1 = Irrelevant; next question is unrelated to the topic or answer.
    2 = Mostly irrelevant; weak or tangential connection.
    3 = Somewhat relevant; some connection, but not strong or focused.
    4 = Mostly relevant; good connection, minor drift.
    5 = Highly relevant; directly follows and builds on the original question and answer.
    """
        prompt = f"""
    You are a medical expert evaluating the relevance of a suggested follow-up medical question for a patient.

    Original Question: {original_question}
    Generated Answer: {generated_answer}
    Suggested Next Question: {next_question}

    First, explain step by step how relevant the next question is to the conversation and how well it connects to the original question and answer.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
        """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("relevance_next_question", explanation, score)


    def evaluate_informativeness_next_question(self, original_question, generated_answer, next_question):
        band = """
    Score meanings:
    1 = Not informative; adds no new information or insight.
    2 = Mostly uninformative; very little added value.
    3 = Somewhat informative; some new information or direction, but limited value.
    4 = Mostly informative; adds useful depth or direction with minor limitations.
    5 = Very informative; provides significant new insight, depth, or direction.
    """
        prompt = f"""
    You are a medical expert evaluating the informativeness of a suggested follow-up medical question for a patient.

    Original Question: {original_question}
    Generated Answer: {generated_answer}
    Suggested Next Question: {next_question}

    First, explain step by step whether the next question adds new depth, explores a new aspect, or moves the conversation forward.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
        """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("informativeness_next_question", explanation, score)


    def evaluate_correctness_next_question(self, original_question, generated_answer, next_question):
        band = """
    Score meanings:
    1 = Incorrect; factually wrong or misleading.
    2 = Mostly incorrect; significant factual errors.
    3 = Somewhat correct; partially accurate but with some mistakes.
    4 = Mostly correct; minor factual issues.
    5 = Completely correct; factually accurate and appropriate.
    """
        prompt = f"""
    You are a medical expert evaluating the factual accuracy of a follow-up medical question suggested for a patient.

    Original Question: {original_question}
    Generated Answer: {generated_answer}
    Suggested Next Question: {next_question}

    First, explain step by step whether the next question is medically accurate, factually correct, and appropriate in the given context.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
        """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("correctness_next_question", explanation, score)


    def evaluate_faithfulness_next_question(self, original_question, generated_answer, next_question):
        band = """
    Score meanings:
    1 = Not faithful; next question is mostly unrelated or introduces hallucinated content.
    2 = Mostly unfaithful; weak or incorrect connection to context.
    3 = Somewhat faithful; partial alignment, some unrelated content.
    4 = Mostly faithful; good connection with minor issues.
    5 = Fully faithful and relevant; next question is grounded and appropriate for context.
    """
        prompt = f"""
    You are a medical evaluator assessing the faithfulness of a suggested follow-up question.

    Original Question: {original_question}
    Generated Answer: {generated_answer}
    Suggested Next Question: {next_question}

    First, explain step by step how faithfully the next question builds on the original question and answer, and whether it introduces unrelated or hallucinated content.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
        """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("faithfulness_next_question", explanation, score)


    def evaluate_coherence_next_question(self, original_question, generated_answer, next_question):
        band = """
    Score meanings:
    1 = Not coherent; confusing or lacks logical structure.
    2 = Mostly incoherent; significant issues in clarity or logic.
    3 = Somewhat coherent; partially logical, but contains notable clarity gaps.
    4 = Mostly coherent; good structure, minor issues.
    5 = Highly coherent; very clear and logically structured.
    """
        prompt = f"""
    You are a medical evaluator evaluating the logical structure and clarity of a suggested follow-up question in a medical dialogue.

    Original Question: {original_question}
    Generated Answer: {generated_answer}
    Suggested Next Question: {next_question}

    First, explain step by step how logically structured and clear the next question is in context.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
        """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("coherence_next_question", explanation, score)


    def evaluate_readability_next_question(self, original_question, generated_answer, next_question):
        band = """
    Score meanings:
    1 = Not readable; difficult or impossible to understand.
    2 = Mostly unreadable; significant language or grammar problems.
    3 = Somewhat readable; partially understandable, but with clarity or grammar issues.
    4 = Mostly readable; clear with minor problems.
    5 = Very readable; fluent and easy to understand.
    """
        prompt = f"""
    You are a medical evaluator assessing the readability and fluency of a suggested follow-up question in a medical conversation.

    Original Question: {original_question}
    Generated Answer: {generated_answer}
    Suggested Next Question: {next_question}

    First, explain step by step how readable and fluent the next question is.

    Then, on a final line, write your final rating from 1 to 5 in the following format:
    {band}
    Final rating (1-5):
    Do not skip the explanation step.
    <RESPONSE>
        """
        response = self.generate(prompt)
        explanation, score = self.extract_explanation_and_score(response)
        return ("readability_next_question", explanation, score)

    def score_answer(self, question, answer, entailment, next_question, gold=None):
        _, explaination_correctness_answer, corr_score_answer = self.evaluate_correctness(question, answer, gold)
        _, explaination_faithfulness_answer, faith_score_answer = self.evaluate_faithfulness(question, answer, gold)
        _, explaination_coherence_answer, coh_score_answer = self.evaluate_coherence(answer)
        _, explaination_readability_answer, read_score_answer = self.evaluate_readability(answer)
        
        _, meteor = self.evaluate_meteor(answer, gold)
        _, bert = self.evaluate_bertscore(answer, gold)
        
        # self.correctness_scores.append(corr_score)
        # self.faithfulness_scores.append(faith_score)
        # self.coherence_scores.append(coh_score)
        # self.readability_scores.append(read_score)
        # self.entailment_values.append(entailment)
        # self.meteor_scores.append(meteor)
        # self.bertscore_scores.append(bert)
        
        _, explaination_relevance_question, relevance_score_question = self.evaluate_relevance_next_question(question, answer, next_question)
        _, explaination_informativeness_question, info_score_question = self.evaluate_informativeness_next_question(question, answer, next_question)
        _, explaination_correctness_question, corr_score_question = self.evaluate_correctness_next_question(question, answer, next_question)
        _, explaination_faithfulness_question, faith_score_question = self.evaluate_faithfulness_next_question(question, answer, next_question)
        _, explaination_coherence_question, coh_score_question = self.evaluate_coherence_next_question(question, answer, next_question)
        _, explaination_readability_question, read_score_question = self.evaluate_readability_next_question(question, answer, next_question)

        result = {
                "question": question,
                "generated_answer": answer,
                "gold_answer": gold,
                "entailment": entailment,
                
                # Answer metrics
                "explanation_correctness": explaination_correctness_answer,
                "score_correctness": corr_score_answer,
                "explanation_faithfulness": explaination_faithfulness_answer,
                "score_faithfulness": faith_score_answer,
                "explanation_coherence": explaination_coherence_answer,
                "score_coherence": coh_score_answer,
                "explanation_readability": explaination_readability_answer,
                "score_readability": read_score_answer,
                "meteor": meteor,
                "bertscore": bert,

                # Next-question
                "next_question": next_question,
                "explanation_relevance_nextq": explaination_relevance_question,
                "score_relevance_nextq": relevance_score_question,
                "explanation_informativeness_nextq": explaination_informativeness_question,
                "score_informativeness_nextq": info_score_question,
                "explanation_correctness_nextq": explaination_correctness_question,
                "score_correctness_nextq": corr_score_question,
                "explanation_faithfulness_nextq": explaination_faithfulness_question,
                "score_faithfulness_nextq": faith_score_question,
                "explanation_coherence_nextq": explaination_coherence_question,
                "score_coherence_nextq": coh_score_question,
                "explanation_readability_nextq": explaination_readability_question,
                "score_readability_nextq": read_score_question,
            }
        return {question: result}
        # self.correctness_question_scores.append(corr_score)
        # self.faithfulness_question_scores.append(faith_score)
        # self.coherence_question_scores.append(coh_score)
        # self.readability_question_scores.append(read_score)


    def report(self, generator_llm_name):
        def avg(lst): return sum(lst) / len(lst) if lst else 0.0
        # Entailment
        series = pd.Series(self.entailment_values)
        counts = series.value_counts()
        percentages = series.value_counts(normalize=True) * 100
        print("\n==== LLM Evaluation Report ====\n")
        print("\n==== Generated Answer ====\n")
        print(f"\nEvaluated Examples: {len(self.correctness_scores)}")
        print(f"\nAverage Correctness:  {avg(self.correctness_scores):.2f}")
        print(f"\nAverage Faithfulness: {avg(self.faithfulness_scores):.2f}")
        print(f"\nAverage Coherence:    {avg(self.coherence_scores):.2f}")
        print(f"\nAverage Readability:  {avg(self.readability_scores):.2f}")
        print("\n==== Next Question ====\n")
        print(f"\nEvaluated Examples: {len(self.correctness_question_scores)}")
        print(f"\nAverage Correctness:  {avg(self.correctness_question_scores):.2f}")
        print(f"\nAverage Faithfulness: {avg(self.faithfulness_question_scores):.2f}")
        print(f"\nAverage Coherence:    {avg(self.coherence_question_scores):.2f}")
        print(f"\nAverage Readability:  {avg(self.readability_question_scores):.2f}")
        print(f"\n\nEntailment Counts:", counts)
        print("\nPercentages:", percentages)
        print(f"\nMETEOR:       {avg(self.meteor_scores):.4f}")
        print(f"\nBERTScore:    {avg(self.bertscore_scores):.4f}")
        print("\n================================\n")
        prefix_model_name = self.model_name.split("/")[1]
        with open(f"INDEX/evaluation_report_{prefix_model_name}_{generator_llm_name.replace('/','_')}.txt", "w") as f:
            f.write("==== LLM Evaluation Report ====\n")
            f.write("\n==== Generated Answer ====\n")
            f.write(f"Evaluated Examples: {len(self.correctness_scores)}\n")
            f.write(f"Average Correctness:  {avg(self.correctness_scores):.2f}\n")
            f.write(f"Average Faithfulness: {avg(self.faithfulness_scores):.2f}\n")
            f.write(f"Average Coherence:    {avg(self.coherence_scores):.2f}\n")
            f.write(f"Average Readability:  {avg(self.readability_scores):.2f}\n")
            f.write("\n==== Next Questions ====\n")
            f.write(f"\nEvaluated Examples: {len(self.correctness_question_scores)}")
            f.write(f"\nAverage Correctness:  {avg(self.correctness_question_scores):.2f}")
            f.write(f"\nAverage Faithfulness: {avg(self.faithfulness_question_scores):.2f}")
            f.write(f"\nAverage Coherence:    {avg(self.coherence_question_scores):.2f}")
            f.write(f"\nAverage Readability:  {avg(self.readability_question_scores):.2f}")
            
            f.write(f"\n\nMETEOR:       {avg(self.meteor_scores):.4f}\n")
            f.write(f"BERTScore:    {avg(self.bertscore_scores):.4f}\n")
            f.write("\nEntailment Counts:\n")
            f.write(counts.to_string() + "\n")
            f.write("Percentages:\n")
            f.write(percentages.to_string() + "\n")
            f.write("================================\n")


# =========================
# Pipeline Runner
# =========================
# =========================
# UI/UX ENHANCEMENT ADDED HERE
# =========================

def save_history_to_json(history, path="history.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        

def save_history_to_pdf(history, path="history.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    page_width = pdf.w - 2 * pdf.l_margin  # Lấy chiều rộng trang, trừ lề

    for idx, item in enumerate(history):
        pdf.set_font("Arial", style="B", size=10)
        pdf.multi_cell(page_width, 8, f"Q{idx+1}: {item['question']}", align="L")
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(page_width, 8, f"Answer: {item['answer']}", align="L")
        pdf.multi_cell(page_width, 8, f"Label: {item['label']} | Confidence: {item['confidence_percent']}%", align="L")
        pdf.multi_cell(page_width, 8, f"Red Flag: {'YES' if item['red_flag'] else 'NO'}", align="L")
        pdf.multi_cell(page_width, 8, "-"*50, align="L")
        pdf.ln(2)
    pdf.output(path)

# =========================
# PipelineRunner
# =========================

class PipelineRunner:
    def __init__(self, lazy_load=True):
        print("\n[Pipeline] Initializing components...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.index_dir = "INDEX"
        self.lazy_load = lazy_load
        self.history = []  
        # self.generator_llm_name = "Qwen/Qwen2.5-14B-Instruct"
        self.generator_llm_name = "OpenMeditron/Meditron3-8B"
        self.model_name = self.generator_llm_name
        self.st_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.tokenizer_bge = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.model_bge = AutoModel.from_pretrained("BAAI/bge-m3", trust_remote_code=True, use_safetensors=True).to(self.device)
        self.retriever = HybridRetriever(self.index_dir, self.tokenizer_bge, self.model_bge, device=self.device)
        self.reranker = CrossEncoderReranker(device=self.device)
        self.verifier = AnswerVerifier(model_name="pritamdeka/PubMedBERT-MNLI-MedNLI", device=self.device)
        self.evaluator = LLMEvaluator(model_name="ContactDoctor/Bio-Medical-Llama-3-8B", device=self.device)
        
        self.llm_generator = LLMAnsweringQuestion(model_name=self.generator_llm_name, device=self.device, is_llama=True)
        
        self.red_flag_embs = self.st_model.encode(RED_FLAG_KEYWORDS)  # shape: (n_keywords, dim)
        
    def is_red_flag_llm(self, user_input):
        """
        Returns (is_red_flag, reason)
        """
        prompt = self.llm_generator.check_red_flag_question_prompt.format(
                    question=user_input,
        )
        if self.llm_generator.is_llama == False:
            input_ids = self.llm_generator.tokenizer(prompt, return_tensors="pt").to(self.llm_generator.model.device)
            output = self.llm_generator.model.generate(**input_ids, max_new_tokens=MAX_LENGTH_FINAL_ANSWER, do_sample=False)
            response = self.llm_generator.tokenizer.decode(output[0], skip_special_tokens=True)
            match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return response.split("<RESPONSE>")[1].strip()
        else:
            messages = [
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": prompt}
            ]
            
            # llama_prompt = self.llm_generator.tokenizer.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )
            # inputs = self.llm_generator.tokenizer(
            #     llama_prompt, return_tensors="pt"
            # ).to(self.llm_generator.model.device)

            # with torch.no_grad():
            #     output = self.llm_generator.model.generate(
            #         **inputs,
            #         max_new_tokens=MAX_LENGTH_FINAL_ANSWER,
            #         pad_token_id=self.llm_generator.tokenizer.eos_token_id,
            #         do_sample=False,
            #         temperature=0.01
            #     )

            # response = self.llm_generator.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            response = query_vllm(prompt,model=self.generator_llm_name)
            response = response.strip()
            output_lower = response.lower()
            if output_lower.startswith("yes"):
                # Extract explanation (if needed)
                reason = response.split(":", 1)[-1].strip() if ":" in response else response
                return True, reason
            return False, None

    
    def is_red_flag_hybrid(self, user_input, threshold=0.75):
        """
        Returns (is_red_flag, reason, score, method)
        method: "keyword" or "llm"
        """
        # 1. Step 1: Cosine similarity with RED_FLAG_KEYWORDS
        q_emb = self.st_model.encode([user_input])[0]
        sims = cosine_similarity([q_emb], self.red_flag_embs)[0]  # array shape (n_keywords,)
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]
        if best_score >= threshold:
            return True, RED_FLAG_KEYWORDS[best_idx]

        # 2. Step 2: LLM semantic reasoning
        is_llm_red_flag, llm_reason = self.is_red_flag_llm(user_input)
        if is_llm_red_flag:
            return True, llm_reason

        # 3. Not a red flag
        return False, None, 
    

    def extract_cancer_type(self,question: str):
        q = question.lower()
        for canonical, syns in CANCER_SYNONYMS.items():
            for s in syns:
                if s in q:
                    return canonical
        return None

    def get_exploration_path(self, question: str, question_type: str, threshold=0.8):
        """
        Identify the current step that best matches the input question, then return the
        flow from that step's group to the end, formatted as a string of unique group names.
        """
        # 1. Urgent symptom case
        if "3" in question_type:
            return {
                "alert": "This symptom may require urgent medical attention. Please visit a hospital or healthcare provider immediately.",
                "current_step": None,
                "current_group": None,
                "flow_steps_string": "",
            }

        # 2. Cancer type extraction
        cancer = self.extract_cancer_type(question)
        if not cancer:
            return {
                "alert": "Could you specify which type of cancer you are interested in?",
                "current_step": None,
                "current_group": None,
                "flow_steps_string": "",
            }

        # 3. Prepare pathway steps (preserve order)
        items = [(group, label, template, template.format(cancer=cancer)) for group, label, template in CANCER_PATHWAY_TEMPLATE]
        q_emb = self.st_model.encode([question])[0]

        # 4. Compute similarity for each step, get best one
        best_score = -float("inf")
        best_idx = None
        for idx, (group, label, template, query) in enumerate(items):
            item_emb = self.st_model.encode([label + ". " + query])[0]
            score = cosine_similarity([q_emb], [item_emb])[0][0]
            if score > best_score:
                best_score = score
                best_idx = idx

        # Helper to get unique group names in order, from a list of items
        def unique_groups_from(idx, items):
            seen = set()
            result = []
            for i in range(idx, len(items)):
                group = items[i][0]
                if group not in seen:
                    seen.add(group)
                    result.append(group)
            return result

        # If score is strong enough, return flow from current step's group onward
        if best_score >= threshold and best_idx is not None:
            group_flow = unique_groups_from(best_idx, items)
            flow_steps_string = " → ".join(group_flow)
            return {
                "alert": None,
                "current_step": items[best_idx][1],
                "current_group": items[best_idx][0],
                "flow_steps_string": flow_steps_string,
            }

        # If no strong match, fallback: show full group pathway for orientation
        all_groups = []
        seen = set()
        for group, _, _, _ in items:
            if group not in seen:
                seen.add(group)
                all_groups.append(group)
        full_flow = " → ".join(all_groups)
        return {
            "alert": "Please consult an oncologist for professional guidance.",
            "current_step": None,
            "current_group": None,
            "flow_steps_string": full_flow,
        }

    def run_pipeline_for_question(self, complex_question, zero_shot = False):
        if zero_shot == True:
            # RED FLAG
            is_emergency,_ = self.is_red_flag_hybrid(complex_question)
            # PIPELINE
            decision = self.check_type_question(complex_question)
            guided_path = self.get_exploration_path(complex_question, decision)
            
            if guided_path["alert"] is None:
                guided_path = guided_path["flow_steps_string"]
            else:
                guided_path = guided_path["alert"]
            
            final_answer = self.reason_final_answer(complex_question, None, None, question_type=decision, zero_shot=zero_shot)
            # VERIFIER
            verification_label, verification_confidence = self.verifier.verify(complex_question, final_answer)
            
            # NEXT QUESTIONS
            next_questions = self.llm_generator.predict_next_question(complex_question, final_answer, decision)
            # CONFIDENCE SCORE
            confidence_score = verification_confidence
            # EXPORTABLE HISTORY 
            response = {
                "answer": final_answer,
                "label": verification_label,
                "guided_path": guided_path,
                "question_type": decision,
                "confidence": confidence_score,
                "confidence_percent": int(confidence_score * 100),
                "next_questions": next_questions,
                "red_flag": is_emergency,
                "guided_path": guided_path,
                "question": complex_question,
                "history_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.history.append(response)
            return response
        else:
            # RED FLAG
            is_emergency,_ = self.is_red_flag_hybrid(complex_question)
            # PIPELINE
            decision = self.check_type_question(complex_question)
            guided_path = self.get_exploration_path(complex_question, decision)
            
            if guided_path["alert"] is None:
                guided_path = guided_path["flow_steps_string"]
            else:
                guided_path = guided_path["alert"]
            
            sub_questions = self.llm_generator.decompose(complex_question)
            sub_answers, qa_pairs_text = self.collect_sub_answers(sub_questions)
            final_answer = self.reason_final_answer(complex_question, sub_questions, qa_pairs_text, question_type=decision)
            # VERIFIER
            verification_label, verification_confidence = self.verifier.verify(complex_question, final_answer)
            
            # NEXT QUESTIONS
            next_questions = self.llm_generator.predict_next_question(complex_question, final_answer, decision)
            # CONFIDENCE SCORE
            confidence_score = verification_confidence
            # EXPORTABLE HISTORY 
            response = {
                "answer": final_answer,
                "label": verification_label,
                "guided_path": guided_path,
                "question_type": decision,
                "confidence": confidence_score,
                "confidence_percent": int(confidence_score * 100),
                "next_questions": next_questions,
                "sub_questions": sub_questions,
                "sub_answers": sub_answers,
                "red_flag": is_emergency,
                "guided_path": guided_path,
                "question": complex_question,
                "history_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.history.append(response)
            return response


    def collect_sub_answers(self, sub_questions):
        sub_answers = []
        qa_pairs_text = ""
        for sub_q in sub_questions:
            retrieved = self.retriever.retrieve(sub_q, top_k=10)
            reranked = self.reranker.rerank(sub_q, retrieved, top_k=3)
            top_context = reranked[0][0] if reranked else ""
            answer = self.llm_generator.generate_sub_answers(sub_q, top_context, self.llm_generator.generate_prompt_template)
            sub_answers.append({
                "question": sub_q,
                "answer": answer,
                "context": top_context
            })
            qa_pairs_text += f"Question: {sub_q}\nAnswer: {answer}\n\n"
        return sub_answers, qa_pairs_text

    def check_type_question(self, complex_question):
        prompt = self.llm_generator.type_question_template.format(
            question=complex_question,
        )
        if self.llm_generator.is_llama == False:
            input_ids = self.llm_generator.tokenizer(prompt, return_tensors="pt").to(self.llm_generator.model.device)
            output = self.llm_generator.model.generate(**input_ids, max_new_tokens=MAX_LENGTH_FINAL_ANSWER, do_sample=False)
            response = self.llm_generator.tokenizer.decode(output[0], skip_special_tokens=True)
            match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return response.split("<RESPONSE>")[1].strip()
        else:
            # messages = [
            #     {"role": "system", "content": "You are a medical expert."},
            #     {"role": "user", "content": prompt}
            # ]
            
            # llama_prompt = self.llm_generator.tokenizer.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )
            # inputs = self.llm_generator.tokenizer(
            #     llama_prompt, return_tensors="pt"
            # ).to(self.llm_generator.model.device)

            # with torch.no_grad():
            #     output = self.llm_generator.model.generate(
            #         **inputs,
            #         max_new_tokens=MAX_LENGTH_FINAL_ANSWER,
            #         pad_token_id=self.llm_generator.tokenizer.eos_token_id,
            #         do_sample=False,
            #         temperature=0.01
            #     )

            # response = self.llm_generator.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            response = query_vllm(prompt,model=self.model_name)
            return response.strip()
        

    def reason_final_answer(self, complex_question, sub_questions, qa_pairs_text, question_type, zero_shot=False):
        if zero_shot == False:
            sub_questions_list = "\n".join([f"- {q}" for q in sub_questions])
            if "1" in question_type:
                prompt = self.llm_generator.information_query_template.format(
                    question=complex_question,
                    sub_questions_list=sub_questions_list,
                    qa_context_pairs=qa_pairs_text
                )
            elif "2" in question_type:
                prompt = self.llm_generator.decision_query_template.format(
                    question=complex_question,
                    sub_questions_list=sub_questions_list,
                    qa_context_pairs=qa_pairs_text
                )
            else:
                prompt = self.llm_generator.symptom_query_template.format(
                    question=complex_question,
                    sub_questions_list=sub_questions_list,
                    qa_context_pairs=qa_pairs_text
                )
            # answer =  self.llm_generator.safe_generate_response(prompt, max_new_tokens=MAX_LENGTH_FINAL_ANSWER)
            answer = query_vllm(prompt,model=self.generator_llm_name)
            return answer
        else:
            if "1" in question_type:
                    prompt = self.llm_generator.information_query_template_zero_shot.format(
                    question=complex_question
                )
            elif "2" in question_type:
                prompt = self.llm_generator.decision_query_template_zero_shot.format(
                    question=complex_question
                )
            else:
                prompt = self.llm_generator.symptom_query_template_zero_shot.format(
                    question=complex_question
                )
            # answer =  self.llm_generator.safe_generate_response(prompt, max_new_tokens=MAX_LENGTH_FINAL_ANSWER)
            answer = query_vllm(prompt,model=self.generator_llm_name)
            return answer

    def evaluate_dataset(self, dataset_path):
        print(f"\n[Pipeline] Evaluating dataset: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = data[:1]
        for idx, item in enumerate(data):
            print(f"\n --------------> Processing [{idx + 1}/{len(data)}]: {item['question']}")
            start_time = time.time()
            question = item["question"]
            reference_answer = item["answer"]
            response = self.run_pipeline_for_question(question)
            final_answer = response["answer"]
            verification_label = response["label"]
            verification_confidence = response["confidence"]
            next_questions = response["next_questions"]
            self.evaluator.score_answer(question, final_answer, verification_label, next_questions, reference_answer)
            end_time = time.time()
            print(f"Execution Time: {end_time - start_time:.2f} seconds")
        self.evaluator.report(self.generator_llm_name)

    def check_and_count_results(self, filename):
        # Load existing results if file exists
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    existing_results = json.load(f)
                except Exception:
                    existing_results = []
        else:
            existing_results = []
        return existing_results, len(existing_results)
        
    def get_final_answer_dataset(self, dataset_path, zero_shot=False):
        print(f"\n[Pipeline] Evaluating dataset: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefix_model_name = self.evaluator.model_name.split("/")[1]
        generator_llm_name = self.generator_llm_name.replace("/", "_")
        if zero_shot == True:
            output_file_path = f"INDEX/generated_answers_from_{generator_llm_name}_zero_shot.json"
        else:
            output_file_path = f"INDEX/generated_answers_from_{generator_llm_name}.json"

        resutls, cnt = self.check_and_count_results(output_file_path)
        print(f"\n[Pipeline] Already processed {cnt} questions. Continuing from there...")
        total_item = len(data)
        data = data[cnt:]  # Skip already processed items
        for idx, item in enumerate(data):
            print(f"\n --------------> Processing [{idx + 1 + cnt}/{total_item}]: {item['question']}")
            start_time = time.time()
            question = item["question"]
            reference_answer = item["answer"]
            response = self.run_pipeline_for_question(question, zero_shot=zero_shot)
            final_answer = response["answer"]
            verification_label = response["label"]
            verification_confidence = response["confidence"]
            next_questions = response["next_questions"]
            end_time = time.time()
            print(f"Execution Time: {end_time - start_time:.2f} seconds")
            qa_dict = {
                "question": question,
                "reference_answer": reference_answer,
                "final_answer": final_answer,
                "next_questions": next_questions,
                "label": verification_label,
                "confidence": verification_confidence,
            }
            resutls.append(qa_dict)
            with open(output_file_path, "w") as f:
                json.dump(resutls, f, indent=4, ensure_ascii=False)

    


