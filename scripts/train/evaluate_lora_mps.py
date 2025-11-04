"""
LoRAファインチューニング済みモデルの性能評価スクリプト

概要:
    学習済みLoRA adapterをロードして、テストデータセットで性能評価を実行します。
    評価専用スクリプトなので、gradient_checkpointingは無効化し、
    通常のTrainerを使用することでSFTTrainerの警告を回避します。
    また、評価時のメモリ逼迫を防ぐために最適化されています。

使用方法:
    1. .envファイルで以下の環境変数を設定:
       - MODEL_NAME: ベースモデル名（デフォルト: gemma-3-4b-it）
       - ADAPTER_PATH: 評価対象のLoRA adapterパス
       - DATASET_PATH: テストデータセットのパス
    
    2. スクリプトを実行:
       python3 scripts/train/evaluate_lora_mps.py

利点:
    - gradient_checkpointingなしで評価（計算効率が良い）
    - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3でメモリ逼迫を防止
    - SFTTrainerのtokenizer非推奨警告を回避
    - 学習スクリプトと分離されてシンプル
    - 複数のadapterを順次評価可能
    - メモリ不足時の自動復旧処理
"""
import os
import torch
import transformers
import json
import gc
from datetime import datetime, timezone, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import PeftModel
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

"""
macOS GPU（Metal Performance Shaders）環境での最適化設定
"""
# MPS (Metal Performance Shaders) が利用可能かの判定
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS (Metal Performance Shaders) が利用できません。このスクリプトはmacOS GPU環境でのみ実行可能です。")

# OS環境変数
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "4")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")  # 評価時は30%に制限してメモリ逼迫を防ぐ

# PyTorch内でのスレッド制御
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(2)

# MPS (Metal Performance Shaders) デバイスの設定
device = torch.device("mps")
print("MPS (Metal Performance Shaders) が利用可能です。GPUを使用して評価を実行します。")
print(f"使用デバイス: {device}")
print("評価用メモリ最適化: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3")


"""
モデルをローカルからロード
"""
# 使用するモデルを指定（.envから取得）
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-4b-it")
MODEL_DIR = f"models/{MODEL_NAME}"

# 評価対象のLoRA adapterパスを指定（.envから取得、デフォルトは最新のものを探す）
ADAPTER_PATH = os.getenv("ADAPTER_PATH", None)

# AutoTokenizer.from_pretrained()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)

# モデルのロード設定（評価時はgradient_checkpointing不要）
model_dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
    dtype=model_dtype,
    device_map=None,
)

# モデルをデバイス (MPS) に配置
model = model.to(device)
print(f"モデルを {device} に配置しました。")

# トークナイザーの基本設定
# padトークンが未定義の場合はeosトークンを使用（transformersライブラリが自動的に同期する）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA adapterをロード
if ADAPTER_PATH:
    try:
        print(f"LoRA adapterを読み込み中: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        
        # メモリクリアしてからマージ
        gc.collect()
        torch.mps.empty_cache()
        
        model = model.merge_and_unload()  # LoRAをマージして通常のモデルに変換
        print("LoRA adapterをマージしました。")
        
        # マージ後のメモリクリア
        gc.collect()
        torch.mps.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            print(f"メモリ不足エラーが発生しました: {e}")
            print("評価を中止します。")
            exit(1)
        else:
            raise e
else:
    print("ADAPTER_PATHが指定されていません。ベースモデルのみで評価します。")

# 評価時はuse_cache=Trueで高速化
model.config.use_cache = True

# 推論モードに設定
model.eval()


"""
データセットの読み込み
"""
def load_dataset(file_path):
    """データセットを読み込む"""
    data = {"test": []}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("split") == "test":
                data["test"].append(item)
    
    # Datasetオブジェクトを作成
    test_dataset = Dataset.from_list(data["test"]) if data["test"] else Dataset.from_list([])
    
    return test_dataset

# データセットを読み込み（.envから取得）
dataset_path = os.getenv("DATASET_PATH", "data/help_list_tool_login.jsonl")
test_dataset = load_dataset(dataset_path)

print(f"テストデータセット: {len(test_dataset)} 件")


"""
SFTTrainer用のデータ整形関数
"""
def formatting_func(batch):
    """データをモデルのchat_templateに合わせて整形"""
    outputs = batch.get("output", [])
    if not outputs:
        return []

    inputs = batch.get("input", [])
    if not inputs:
        inputs = [""] * len(outputs)

    systems = batch.get("system", [])
    if not systems:
        systems = [None] * len(outputs)

    texts = []
    for inp, out, sys in zip(inputs, outputs, systems):
        # メッセージリストを構築
        messages = []

        # systemメッセージがある場合は追加
        if sys and sys.strip():
            messages.append({"role": "system", "content": sys.strip()})

        # userメッセージを追加
        messages.append({"role": "user", "content": inp.strip() if inp else ""})

        # assistantメッセージを追加
        messages.append({"role": "assistant", "content": out.strip() if out else ""})

        # tokenizer.chat_templateを使って整形
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return texts


"""
データセットの前処理
"""
def preprocess_function(examples):
    """データセットをモデルの入力形式に変換"""
    # formatting_funcを使ってバッチ処理
    texts = formatting_func(examples)
    
    # トークナイズ
    model_inputs = tokenizer(
        texts,
        max_length=768,  # 学習時と同じ長さ
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    
    # labelsとして使用
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

# データセットを前処理
print("\nデータセットを前処理中...")
tokenized_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=10,
    remove_columns=test_dataset.column_names,
    desc="Tokenizing dataset"
)


"""
評価設定
"""
training_args = TrainingArguments(
    output_dir="outputs/evaluation",  # 評価結果の保存先
    per_device_eval_batch_size=2,     # 評価用バッチサイズ
    bf16=True,                        # bfloat16で評価
    dataloader_num_workers=0,         # macOS環境では0が安定
    dataloader_pin_memory=False,      # MPSではFalse推奨
    disable_tqdm=False,               # プログレスバーを表示
    report_to="none",                 # ログは出力しない
)


"""
評価実行
"""
if __name__ == '__main__':
    try:
        # Trainerを作成（SFTTrainerではなく通常のTrainerを使用）
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_dataset,
            processing_class=tokenizer,  # tokenizer警告を回避
        )
        
        print("\n=== テストデータでの評価開始 ===")
        
        # 評価実行前のメモリクリア
        gc.collect()
        torch.mps.empty_cache()
        
        # 評価実行
        eval_results = trainer.evaluate()
        
        print("\n=== 評価結果 ===")
        print(f"テスト損失: {eval_results['eval_loss']:.4f}")
        print(f"評価実行時間: {eval_results['eval_runtime']:.2f}秒")
        
        # 結果をファイルに保存
        JST = timezone(timedelta(hours=9))
        timestamp = datetime.now(JST).strftime("%Y%m%d-%H%M%S")
        output_file = f"outputs/logs/evaluation_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\n評価結果を保存しました: {output_file}")
        print("評価が正常に完了しました。")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            print(f"\nメモリ不足エラーが発生しました: {e}")
            print("評価はスキップされました。")
            exit(1)
        else:
            raise e
    except Exception as e:
        print(f"\n評価中にエラーが発生しました: {e}")
        print("評価を中止します。")
        raise e

