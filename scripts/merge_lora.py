"""
LoRAで学習した差分 (outputs/lora_adapter) をベースモデルにマージして1つのフル重みモデルとして保存する

出力ディレクトリ (OUT_DIR) に揃うべきファイル群とその解説：
    下記が全て OUT_DIR 内にあれば、単一の通常モデルとして配布・変換・推論可能となる

    model.safetensors.index.json
        - safetensors形式の重み分割保存の目次ファイル
        - 大きなモデルは複数分割されるため、どの重みファイルに何があるかを示すインデックス

    model-0000x-of-0000y.safetensors
        - safetensors形式で保存された実際の重みファイル群
        - x: ファイル番号、y: 総分割数
            - 例えば 3分割なら model-00001-of-00003.safetensors など
            - 分割されない小さいモデルなら1つだけの場合も有り

    config.json
        - モデルの構造やハイパーパラメータ情報（層の数や隠れ次元数など）を含む設定ファイル

    tokenizer.*
        - トークナイザー関連のファイル群
        - 例えば tokenizer.json, tokenizer.model, tokenizer_config.json, special_tokens_map.json などが含まれる
        - モデル変換や推論時に必要となる
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = "./finetuning/model/gemma-2-2b"         # ベースモデル
ADAPTER_DIR = "./finetuning/outputs/lora_adapter"  # LoRA差分の出力先
OUT_DIR = "./finetuning/model/merged-gemma-2-2b"   # マージ後の保存先

# モデルを読み込む際の重み型 (dtype) を指定
# torch.float32 (FP32)
#     - CPU環境でも安定して動作する
# torch.bfloat16 (BF16) あるいは torch.float16 (FP16)
#     - GPU（特にNVIDIA）を使う場合は高速化・省メモリ化のために利用可能
#     - 但し、CPU Onlyだとエラーになる事が多いため、GPUのときのみ有効にする
dtype = torch.float32

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_DIR, local_files_only=True, dtype=dtype
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)  # LoRA層を注入

print("Merging LoRA into base weights (this may take a bit)...")
# ここでベースにLoRAを「焼き込んで」通常モデル化する
# LoRA層を統合してPEFT依存を外す
model = model.merge_and_unload()

print(f"Saving merged model to: {OUT_DIR}")
model.save_pretrained(OUT_DIR, safe_serialization=True)

# Tokenizer も同じ場所へ（llama.cpp 変換時に必要）
print("Saving tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True)
tok.save_pretrained(OUT_DIR)

print("Done. Merged model is at:", OUT_DIR)
