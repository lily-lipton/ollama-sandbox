import os, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer
from peft import LoraConfig

# ---- CPUのスレッド数を絞る（過負荷防止）----
# OMP_NUM_THREADS: PyTorchなどで使われるOpenMPベースの並列計算用スレッド数を指定
os.environ.setdefault("OMP_NUM_THREADS", "2")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

# MKL_NUM_THREADS: Intel MKL (行列演算ライブラリ) で用いるスレッド数を指定
os.environ.setdefault("MKL_NUM_THREADS", "2")

# NUMEXPR_MAX_THREADS: numexprライブラリの最大スレッド数（内部で使われることがある）を指定
os.environ.setdefault("NUMEXPR_MAX_THREADS", "2")

# torch.set_num_interop_threads: 異なる並列バックエンド間での競合を抑えるためのスレッド数（1で厳しくする）
torch.set_num_interop_threads(1)


"""
モデルをローカルから読む
"""
MODEL_DIR = "../models/gemma-2-2b"

# Hub を経由しない
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

# Gemmaはpadトークン未定義なので、SFT時のバッチ化のためにpadをEOSに合わせる
# バッチは以下のような2次元配列となり、最大長に要素数を揃えて上げる必要がある
# 文1: "こんにちは"     => [123, 456, 789]      => [123, 456, 789, <pad>, <pad>]
# 文2: "今日は良い天気" => [10, 20, 30, 40, 50] => [10, 20, 30, 40, 50]
# 文3: "こんばんは"     => [111, 222, 333]      => [111, 222, 333, <pad>, <pad>]
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# モデルの読み込み設定
# 【重要】CPUでは fp16/bf16 は不可。float32 で安定動作
# 【任意】勾配チェックポイントを使うなら後で use_cache=False にする
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    dtype=torch.float32,  # CPUでは float32 のみ対応
)

# 勾配チェックポイント (メモリ消費は抑えられるが、計算時間は増える)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# キャッシュの無効化
# 推論向けの高速化キャッシュをOFFにして、学習時のメモリ使用を抑える
# if hasattr(model.config, "use_cache"):
#     model.config.use_cache = False

"""
データセットの読み込み
"""
data = load_dataset("skouai/Kansai-Obachan")
# このデータは test のみなので train に付け替える or そのまま test を使う
data = DatasetDict({"train": data["test"]})

# ここで tokenizer による map は不要（SFTTrainer に任せる）
# data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# SFTTrainerに渡す生テキストを作る関数
def formatting_func(example):
    # データセット (skouai/Kansai-Obachan) に合わせて修正
    instr = example.get("instruction", "")
    inp   = example.get("input", "")
    out   = example.get("output", "")

    if inp:
        text = f"指示: {instr}\n入力: {inp}\n応答: {out}{tokenizer.eos_token}"
    else:
        text = f"指示: {instr}\n応答: {out}{tokenizer.eos_token}"

    return [text]


"""
LoRA設定
"""
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "o_proj", "k_proj", "v_proj",  # Attention周り
        "gate_proj", "up_proj", "down_proj"      # MLP周り（GemmaはSwiGLU系）
    ],
    task_type="CAUSAL_LM",
)


"""
トレーニング設定
"""
training_args = transformers.TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,   # ←据え置き（学習量維持）
    warmup_steps=2,
    max_steps=20,                    # ←据え置き（学習量維持）
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=20,
    save_total_limit=1,
    fp16=False,
    bf16=False,
    # ↓ メモリ/CPU負荷を抑えるための実運用オプション
    optim="adafactor",               # AdamWよりメモリ消費が小さい
    dataloader_num_workers=0,        # ワーカープロセスを増やさない
    dataloader_pin_memory=False,     # GPUなしならFalseが安定
    disable_tqdm=False,
    report_to="none",
)


"""
SFTTrainerを用いたデータローダ構築
"""
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=data["train"],
    formatting_func=formatting_func,
    args=training_args,
    tokenizer=tokenizer,
    # max_seq_length は “実データの長さ” に合わせるのがコツ
    # データが短いなら 1024を維持しても無駄に重くなるだけ
    # 95パーセンタイル長に合わせて 256～512 へ“調整”しても学習量（ステップ数）は維持されます。
    max_seq_length=512,
)

trainer.train()


"""
adapterの保存
"""
# LoRA差分（adapter）のみ保存
# 出力先は outputs/ 直下
trainer.model.save_pretrained("outputs/lora_adapter")

print("Done. LoRA adapter saved to outputs/lora_adapter")
