import os, torch, transformers, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict, Dataset
from trl import SFTTrainer
from peft import LoraConfig

# ---- M1 Pro環境での最適化設定 ----
# M1 Proの統一メモリアーキテクチャを活用するため、スレッド数を増やす
# OMP_NUM_THREADS: PyTorchなどで使われるOpenMPベースの並列計算用スレッド数を指定
os.environ.setdefault("OMP_NUM_THREADS", "8")  # M1 Proのコア数に合わせて増加
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

# MKL_NUM_THREADS: Intel MKL (行列演算ライブラリ) で用いるスレッド数を指定
os.environ.setdefault("MKL_NUM_THREADS", "8")  # M1 Proのコア数に合わせて増加

# NUMEXPR_MAX_THREADS: numexprライブラリの最大スレッド数（内部で使われることがある）を指定
os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")  # M1 Proのコア数に合わせて増加

# torch.set_num_interop_threads: 異なる並列バックエンド間での競合を抑えるためのスレッド数
torch.set_num_interop_threads(4)  # M1 Proの性能を活用するため増加

# Hugging Face Hubへのアクセスを完全に無効化
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


"""
モデルをローカルから読む
"""
# 使用するモデルを指定（gemma-2-2b または gemma-2-9b）
MODEL_NAME = "gemma-2-2b"  # 必要に応じて "gemma-2-9b" に変更
MODEL_DIR = f"/trainer/models/{MODEL_NAME}"

# Hub を経由しない
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)

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
    trust_remote_code=True,
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
# 社内FAQデータを読み込み
def load_faq_dataset(file_path):
    
    data = {"train": [], "validation": [], "test": []}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            split = item.get("split", "train")
            if split in data:
                data[split].append(item)
    
    # Datasetオブジェクトを作成
    train_dataset = Dataset.from_list(data["train"])
    validation_dataset = Dataset.from_list(data["validation"]) if data["validation"] else Dataset.from_list(data["train"][:5])
    test_dataset = Dataset.from_list(data["test"]) if data["test"] else Dataset.from_list(data["train"][:5])
    
    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

# 社内FAQデータを読み込み
data = load_faq_dataset("/trainer/data/sample/help_list_tool_login.jsonl")

# DatasetDictから個別のDatasetを取得
train_dataset = data["train"]
eval_dataset = data["validation"]
test_dataset = data["test"]

# ここで tokenizer による map は不要（SFTTrainer に任せる）
# data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# SFTTrainerに渡す生テキストを作る関数
def formatting_func(example):
    # 社内FAQデータの構造に合わせて修正
    instr = example.get("instruction", "")
    inp   = example.get("input", "")
    out   = example.get("output", "")

    if inp:
        text = f"質問: {instr}\n詳細: {inp}\n回答: {out}{tokenizer.eos_token}"
    else:
        text = f"質問: {instr}\n回答: {out}{tokenizer.eos_token}"

    return [text]  # リストを返す（SFTTrainerの要求）


"""
LoRA設定
"""
lora_config = LoraConfig(
    r=32,                    # 表現力を向上させるため増加（8→32）
    lora_alpha=64,           # rの2倍に設定（32→64）
    lora_dropout=0.05,       # 過学習を防ぎつつ学習効率を向上（0.1→0.05）
    target_modules=[
        "q_proj", "o_proj", "k_proj", "v_proj",  # Attention周り
        "gate_proj", "up_proj", "down_proj"      # MLP周り（GemmaはSwiGLU系）
    ],
    task_type="CAUSAL_LM",
    bias="none",             # バイアスは学習しない（メモリ効率化）
    use_rslora=True,         # Rank-Stabilized LoRA（学習安定性向上）
)


"""
トレーニング設定 - バリデーションとEarly Stopping対応
"""
training_args = transformers.TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,     # 小規模データセット（79件）のため1に戻す
    per_device_eval_batch_size=1,      # バリデーション用バッチサイズ
    gradient_accumulation_steps=4,     # 実効バッチサイズ4で十分（8→4）
    warmup_steps=5,                    # 小規模データセットに適したウォームアップ（10→5）
    max_steps=100,                     # 小規模データセットに適した学習ステップ数（200→100）
    learning_rate=1e-4,                # より安定した学習率（2e-4→1e-4）
    logging_steps=5,                   # ログ頻度を調整（1→5）
    eval_steps=10,                     # バリデーション実行頻度（10ステップごと）
    save_steps=50,                     # チェックポイント保存頻度調整（20→50）
    save_total_limit=3,                # より多くのチェックポイントを保持（1→3）
    fp16=False,                        # M1 Proではfloat32が安定
    bf16=False,                        # M1 Proではfloat32が安定
    # ↓ M1 Pro環境での最適化オプション
    optim="adamw_torch",               # M1 Proでより効率的（adafactor→adamw_torch）
    dataloader_num_workers=2,          # M1 Proのコア数を活用（0→2）
    dataloader_pin_memory=False,       # CPU環境ではFalseが安定
    disable_tqdm=False,
    report_to="none",
    # 追加の最適化オプション
    dataloader_prefetch_factor=2,      # データローダーの効率化
    label_smoothing_factor=0.1,        # 過学習防止
    max_grad_norm=1.0,                 # 勾配クリッピング
    # Early Stopping設定
    load_best_model_at_end=True,       # 最良モデルを最終的に読み込み
    metric_for_best_model="eval_loss",  # 評価指標（損失）
    greater_is_better=False,           # 損失は小さい方が良い
    eval_strategy="steps",             # ステップごとに評価（新しい引数名）
    save_strategy="steps",              # ステップごとに保存
)

"""
Early Stopping コールバック
"""
from transformers import EarlyStoppingCallback

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # 3回連続で改善しなければ停止
    early_stopping_threshold=0.001,  # 改善の閾値
)

"""
SFTTrainerを用いたデータローダ構築 - バリデーション対応
"""
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,          # バリデーションデータを追加
    formatting_func=formatting_func,
    args=training_args,
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback], # Early Stoppingコールバックを追加
    # max_seq_length は "実データの長さ" に合わせるのがコツ
    # 社内FAQデータの回答は200-500文字程度なので、768で十分
    # GCE VMでのCPU推論も考慮して適度な長さに設定
    max_seq_length=768,
    packing=False,  # packingを無効にして各サンプルを個別処理
)

# 学習実行
trainer.train()

"""
テストデータでの最終評価
"""
print("\n=== テストデータでの最終評価 ===")
test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
print(f"テスト損失: {test_results['test_loss']:.4f}")

"""
adapterの保存
"""
# LoRA差分（adapter）のみ保存
# 出力先は models/ 配下に統一
adapter_path = f"/trainer/models/{MODEL_NAME}-lora"
trainer.model.save_pretrained(adapter_path)

print("\n=== 学習完了 ===")
print(f"LoRA adapter saved to {adapter_path}")
print(f"最終テスト損失: {test_results['test_loss']:.4f}")
print("学習が正常に完了しました。")
