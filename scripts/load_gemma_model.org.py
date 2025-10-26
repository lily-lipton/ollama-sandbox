"""
Hugging Face上のGemmaモデル (google/gemma-2b-pytorch) を、4bit量子化してCPUにロードする
Hugging Face のアクセストークン (規約同意済みの証明) を環境変数 HF_TOKEN に設定しておく必要がある
"""
import torch
import os

# transformers: Hugging Faceのメインライブラリ
# BitsAndBytesConfig: モデルを量子化 (軽量化) して読み込む設定を定義するクラス
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/gemma-2b-pytorch"

# 量子化設定 (BitsAndBytesConfig) を定義
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Gemmaモデルのトークナイザーを Hugging Face からダウンロード
# HF_TOKEN は Hugging Face のアクセストークン (規約同意済みの証明)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.environ['HF_TOKEN']
)

# Gemmaモデルを Hugging Face からダウンロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"":0},
    token=os.environ['HF_TOKEN']
)
