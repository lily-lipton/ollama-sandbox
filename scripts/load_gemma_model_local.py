"""
ローカルに取得済みのGemmaモデル (google/gemma-2-2b) をCPUにロード
"""
# transformers: Hugging Faceのメインライブラリ
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/home/pd-user/finetuning/model/gemma-2-2b"

# Hub を経由しない
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)

print("Loaded:", MODEL_DIR)
