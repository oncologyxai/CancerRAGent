import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# from multi_retrieval_riken_new_1 import PipelineRunner   
from multi_retrieval_riken_vllm import PipelineRunner

print("🔹 Initializing PipelineRunner and loading models to memory (this may take 2-5 minutes)...")
pipeline_runner = PipelineRunner(lazy_load=False)
print("✅ Pipeline & LLM loaded. Ready to serve requests!")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    label: str
    confidence: float
    next_questions: list
    red_flag: bool
    sub_questions: list
    sub_answers: list
    guided_path: str
    history_time: str

# --- FastAPI App ---
app = FastAPI(title="LLM Medical Pipeline API", version="1.0")

@app.post("/ask", response_model=QueryResponse)
def ask_medical_question(payload: QueryRequest):
    question = payload.question
    result = pipeline_runner.run_pipeline_for_question(question)


    def normalize_list(val):
        if val is None:
            return []
        if isinstance(val, str):
            return [q.strip("-* ").strip() for q in val.split("\n") if q.strip()]
        if isinstance(val, list):
            return val
        return [val]

    result["next_questions"] = normalize_list(result.get("next_questions", []))
    result["sub_questions"] = normalize_list(result.get("sub_questions", []))
    result["sub_answers"] = normalize_list(result.get("sub_answers", []))

    return QueryResponse(**result)

@app.get("/")
def root():
    return {"message": "Medical LLM FastAPI Server is running!"}

# Optional: Endpoint to reload model without restarting server
@app.post("/reload")
def reload_model():
    global pipeline_runner
    pipeline_runner = PipelineRunner(lazy_load=False)
    return {"message": "Pipeline and LLM model reloaded!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


# command to run server
#  uvicorn server_llm_fastapi:app --host 0.0.0.0 --port 8000