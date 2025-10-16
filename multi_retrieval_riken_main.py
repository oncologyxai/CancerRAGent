from multi_retrieval_riken_new_CoT import PipelineRunner
from multi_retrieval_riken_new_CoT import *
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
if __name__ == "__main__":
    runner = PipelineRunner()
    runner.get_final_answer_dataset("RIKEN/MedQuAD-master/1_CancerGov_QA.json", zero_shot=False)
    
    
    # qs = [
    #     # "What is the liver cancer",
    #     # "What is the symptom of lung cancer?",
    #     "What are the early symptoms of liver cancer?",
    #     # "Can you tell me how many stages of liver cancer?",
    #     # "What is the best treatment hfor liver cancer?",
    #     # "What methods do you prefer for liver cancer treatment?",
    #     "My friend has severe chest pain and coughing up blood, what should he do?",
    # ]
    # for q in qs:
    #     print(f"\n>>> User: {q}")
    #     response = runner.run_pipeline_for_question(q)

    #     best_path = response.get("best_path", None)
    #     if best_path:
    #         print("Suggested step:", best_path)
    #     else:
    #         print("No relevant path step found.")
    #     print("Answer:", response["answer"])
    #     if response["red_flag"]:
    #         print("[WARNING] This may be an emergency! Please contact your doctor.")

    # save_history_to_json(runner.history, "history.json")
    # print("Exported history to history.json")

    # save_history_to_pdf(runner.history, "history.pdf")
    # print("Exported history to history.pdf")