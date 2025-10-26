"""
Hugging Face上のGemmaモデル (google/gemma-2b-pytorch) をCPUにロード
Hugging Face のアクセストークン (規約同意済みの証明) を環境変数 HF_TOKEN に設定しておく必要がある
"""
import torch
import os

# transformers: Hugging Faceのメインライブラリ
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b-pytorch"
# model_id = "google/gemma-2-2b"

# Gemmaモデルのトークナイザーを Hugging Face からダウンロード
# HF_TOKEN は Hugging Face のアクセストークン (規約同意済みの証明)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.environ['HF_TOKEN']
)

# Gemmaモデルを Hugging Face からダウンロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=os.environ['HF_TOKEN']
)
