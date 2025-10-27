# ollama-sandbox
For fine-tuning verification of LLMs, etc.

## ディレクトリ構成
```
ollama-sandbox
├── .env
├── .gitignore
├── docker-compose.yml
├── Modelfiles/
├── models/                # GGUFやtokenizer等 (Git管理外)
├── data/                  # 学習用データ (Git管理外)
├── trainer/               # LoRA学習/ベース合成/GGUF変換等のツールボックス
│   └── Dockerfile         # llama.cppバイナリ等を整備したコンテナ環境
│   └── requirements.txt
└── scripts/               # LoRA学習/ベース合成/GGUF変換用スクリプト郡
```

## 手順

1. 初期化

```sh
docker compose up -d --build
```

2. LoRA学習 → ベース合成 → GGUF量子化 (`trainer`コンテナにて)

```sh
docker exec -it trainer bash

# LoRA学習
python3 scripts/lora_sft_light.py \
  --data /trainer/data/your_dataset.jsonl \
  --output /trainer/models/lora_adapter

# ベース合成
python3 scripts/merge_lora.py \
  --base /trainer/models/base_model/      # safetensors 等
  --lora /trainer/models/lora_adapter \
  --out  /trainer/models/merged-f32.gguf  # or merged-f16.bin 等

# 量子化 (llama-quantizeを使う場合)
llama-quantize /trainer/models/merged-f32.gguf \
  /trainer/models/gemma-2-2b-lora.Q4_K_M.gguf \
  Q4_K_M
```

3. Ollamaへ登録 (`ollama`コンテナにて)

```sh
# モデル作成 (Modelfileを使用)
docker exec -it ollama bash -lc \
  "ollama create gemma2-2b-lora -f /Modelfiles/gemma2-2b-lora.Modelfile"

# 作成できたことを確認
ollama list

# 動作確認
ollama run gemma2-2b-lora

# 動作確認 (curl)
curl -s http://localhost:11434/api/chat -d '{
  "model": "gemma2-2b-lora",
  "messages": [{"role":"user","content":"こんにちは。自己紹介して"}]
}'
```
