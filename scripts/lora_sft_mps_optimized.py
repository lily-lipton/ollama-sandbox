import os, torch, transformers, json, gc
from datetime import datetime, timezone, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import DatasetDict, Dataset
from trl import SFTTrainer
from peft import LoraConfig

"""
macOS GPU（Metal Performance Shaders）環境での最適化設定
"""
# MPS (Metal Performance Shaders) が利用可能かの判定
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS (Metal Performance Shaders) が利用できません。このスクリプトはmacOS GPU環境でのみ実行可能です。")

# OS環境変数
os.environ.setdefault("OMP_NUM_THREADS", "4")                     # PyTorch内部で使われているOpenMP (線形計算ライブラリ) のスレッド数制御
os.environ.setdefault("MKL_NUM_THREADS", "4")                     # PyTorch内部で使われているIntel MKL (行列演算ライブラリ) のスレッド数制御
os.environ.setdefault("NUMEXPR_MAX_THREADS", "4")                 # transformers内部で使われているnumexpr (数式評価高速化ライブラリ) の最大スレッド数制御
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")         # MPSが利用できない場合のCPUフォールバック有効化
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # MPSのメモリ使用量の最適化 (0.0: 最小限のメモリ使用)

# PyTorch内でのスレッド制御
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))         # PyTorchがCPU計算に使うスレッド数制御
torch.set_num_interop_threads(2)                                  # 異なる並列バックエンド間での競合を抑えるためのスレッド数制御

# MPS (Metal Performance Shaders) デバイスの設定
device = torch.device("mps")
print("MPS (Metal Performance Shaders) が利用可能です。GPUを使用してトレーニングを実行します。")
print(f"使用デバイス: {device}")

# MPS環境でのメモリ最適化設定
torch.mps.empty_cache()  # MPSキャッシュをクリア


"""
モデルをローカルからロード
"""
# 使用するモデルを指定
MODEL_NAME = "gemma-2-9b"
MODEL_DIR = f"models/{MODEL_NAME}"

# AutoTokenizer.from_pretrained()
#   - 指定したモデル専用のトークナイザーのインスタンスを生成
#   - local_files_only=True
#       - Hugging Face Hub を経由せずにローカルからモデルをロード
#   - trust_remote_code=True
#       - モデル側で独自に定義されたトークナイザークラスやメソッドを読み込むことを許可
#       - リモートで提供されている Python コードを実行できるようになるため、信頼できるソースであることが前提
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)

# Gemmaはpadトークン未定義なので、SFT時のバッチ化のためにpadをEOSに合わせる
#   - バッチは以下のような2次元配列となり、最大長に要素数を揃えて上げる必要がある
#         文1: "こんにちは"     => [123, 456, 789]      => [123, 456, 789, <pad>, <pad>]
#         文2: "今日は良い天気" => [10, 20, 30, 40, 50] => [10, 20, 30, 40, 50]
#         文3: "こんばんは"     => [111, 222, 333]      => [111, 222, 333, <pad>, <pad>]
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# モデルのロード設定
# macOS GPU (MPS) では bf16 (bfloat16) が推奨
model_dtype = torch.bfloat16

# AutoModelForCausalLM.from_pretrained()
#   - 指定したモデルをロードし、専用のモデルインスタンス (学習可能なモデルオブジェクト) を生成
#   - local_files_only=True
#       - Hugging Face Hub を経由せずにローカルからモデルをロード
#   - trust_remote_code=True
#       - モデル側で独自に定義されたトークナイザークラスやメソッドを読み込むことを許可
#   - dtype=model_dtype
#       - モデルの重みをどの精度 (データ型) で読み込むかを指定
#       - ここでは別途設定した torch.bfloat16 を渡して、省メモリ化や高速化を図っている
#   - device_map=None
#       - 自動で GPU / CPU を割り当てず、後続のコードで手動でデバイス配置を制御
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

# 勾配チェックポイント (gradient_checkpointing) を有効化
# この手法により、トレーニング中のメモリ消費を抑えられるが、計算時間は増加する
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()


"""
データセットの読み込み
"""
# 教師データセットを読み込み、DatasetDictオブジェクトを返す
# 返却値の構造：
#     DatasetDict({
#         train: Dataset({
#             features: ['output', 'input', 'instruction'],
#             num_rows: 15015
#         }),
#         validation: Dataset({
#             features: ['output', 'input', 'instruction'],
#             num_rows: 2000
#         }),
#         test: Dataset({
#             features: ['output', 'input', 'instruction'],
#             num_rows: 2000
#         })
#     })
def load_dataset(file_path):
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

# 教師データセットを読み込み
data = load_dataset("data/sample/help_list_tool_login.jsonl")

# DatasetDictから各Datasetを取得
train_dataset = data["train"]
eval_dataset = data["validation"]
test_dataset = data["test"]

# SFTTrainer用のデータ整形を行う (バッチ処理版)
# なお、ここではバッチ処理の形式を取っているが、単一サンプル処理 + dataset.map() でも同様の処理が可能
# 入力値 (バッチデータ) の構造:
#    batch = {
#        "instruction": ["質問1", "質問2", "質問3", …],
#        "input": ["詳細1", "", "詳細3", …], 
#        "output": ["回答1", "回答2", "回答3", …]
#    }
# 返却値の構造:
#     [
#         "質問: {instruction}\n詳細: {input}\n回答: {output}{EOS}",
#         "質問: {instruction}\n\n回答: {output}{EOS}",
#         :
#     ]
def formatting_func(batch):
    outputs = batch.get("output", [])
    if not outputs:
        return []

    instructions = batch.get("instruction")
    if not instructions:
        instructions = [""] * len(outputs)

    inputs = batch.get("input")
    if not inputs:
        inputs = [""] * len(outputs)

    texts = []
    for instr, inp, out in zip(instructions, inputs, outputs):
        instr = instr or ""
        out = out or ""
        if inp:
            text = f"質問: {instr}\n詳細: {inp}\n回答: {out}{tokenizer.eos_token}"
        else:
            text = f"質問: {instr}\n回答: {out}{tokenizer.eos_token}"
        texts.append(text)

    return texts


"""
LoRA設定
"""
lora_config = LoraConfig(
    r=32,                    # 数値が大きいほど追加の学習パラメータが多くなり表現力が向上する
    lora_alpha=64,           # LoRAのスケーリング係数 (`r`の2倍程度がよく使われる)
    lora_dropout=0.05,       # LoRA層にだけ適用するドロップアウト (一部の重みを一時的に無効化)。値を下げるとより多くの情報を使い、上げるとランダム要素が入り過学習を防ぎやすくなる。0.05は「やや控えめ」な値。
    target_modules=[
        "q_proj", "o_proj", "k_proj", "v_proj",  # Attention (重要単語を判別する仕組み) の主要部分。LoRAで軽量に学習パラメータを加える対象
        "gate_proj", "up_proj", "down_proj"      # MLP (文章をより深く理解する部分) の層。GemmaはSwiGLUという形式なのでこの3つ
    ],
    task_type="CAUSAL_LM",    # モデルの種類
    bias="none",              # 追加のバイアス項を使わない (LoRAの場合は基本不要)
    use_rslora=True,          # Rank-Stabilized LoRA (RS-LoRA) を使用する = 学習が安定しやすくなる
)


"""
トレーニング設定
"""
per_device_train_batch_size = 2      # 1デバイス (GPU) あたりの学習用バッチサイズ
per_device_eval_batch_size = 2       # 1デバイス (GPU) あたりのバリデーション用バッチサイズ
gradient_accumulation_steps = 8      # 勾配の累積回数 (この回数分バッチを積み重ねてから1回パラメータを更新する
fp16_enabled = False                 # float16 (FP16) 半精度演算の有効化フラグ (MPSでは未対応または十分な恩恵がないためFalse)
bf16_enabled = True                  # bfloat16(BF16) 半精度演算の有効化フラグ (MPSではbf16による計算精度向上や安定性・高速化が期待できるためTrue推奨)
dataloader_num_workers = 0           # データローダーで使用するワーカープロセス数 (0にするとシングルプロセスのみ使用。macOSではマルチプロセス時にエラーが発生しやすいため、0が推奨値)
dataloader_pin_memory = False        # PyTorchのdataloaderでメモリをpinned(固定)するかどうか (WindowsやCUDA環境ではTrueが高速化に繋がることが多いが、MPSではむしろメモリ確保失敗など安定しないのでFalseにする)

training_args = transformers.TrainingArguments(
    output_dir="outputs",                  # 学習成果物 (モデルやログなど) を出力するディレクトリ
    per_device_train_batch_size=per_device_train_batch_size,   # 1デバイス (GPU) の訓練用バッチサイズ
    per_device_eval_batch_size=per_device_eval_batch_size,     # 1デバイス (GPU) の検証用バッチサイズ
    gradient_accumulation_steps=gradient_accumulation_steps,   # 勾配累積回数
    learning_rate=1e-4,                    # 基本の学習率 (AdamW最適化器でよく使われる値、モデルやデータに応じて変更推奨)

    warmup_steps=5,                        # 最初の数ステップは小さい学習率でウォームアップ (急な学習率変動による不安定化を防ぐ)
    max_steps=100,                         # 総学習ステップ数。データ量や目的に合わせて調整
    logging_steps=5,                       # ログ (損失や評価値など) を何ステップごとに出力するか
    eval_steps=10,                         # バリデーション (検証) を何ステップごとに行うか
    save_steps=50,                         # モデルをチェックポイントとして何ステップごとに保存するか

    # 以下は検証用
    # warmup_steps=1,
    # max_steps=5,
    # logging_steps=1,
    # eval_steps=2,
    # save_steps=4,

    save_total_limit=3,                    # 保存するチェックポイント (モデルの節目) 数の上限、それを超えた古いものは自動消去
    fp16=fp16_enabled,                     # float16の有効化フラグ
    bf16=bf16_enabled,                     # bfloat16の有効化フラグ
    optim="adamw_torch",                   # パラメータ最適化アルゴリズム (PyTorch標準のAdamW)
    dataloader_num_workers=dataloader_num_workers,    # DataLoaderに使う並列プロセス数 (macOSは0で安定・エラー回避)
    dataloader_pin_memory=dataloader_pin_memory,      # DataLoaderがバッチデータを"pinned memory"に置くか。MPSはFalse推奨
    disable_tqdm=False,                    # プログレスバー (tqdm) 表示を無効化するか。Trueにすると標準出力が静かになる
    report_to="tensorboard",               # TensorBoardでログを可視化
    logging_dir="outputs/logs",            # ログファイルの保存先
    label_smoothing_factor=0.1,            # ラベルスムージングで過学習防止 (0.1は控えめだが効果的)
    max_grad_norm=1.0,                     # 1回あたりの勾配の最大ノルム値 (値を超えたらクリッピングし暴走学習防止)

    # ---- Early Stopping & モデル保存戦略 ----
    load_best_model_at_end=True,           # 学習最後に「評価損失が最も良い」チェックポイントを自動で読み込む
    metric_for_best_model="eval_loss",     # どの指標で「ベスト」を決めるか (ここではバリデーション用損失)
    greater_is_better=False,               # 指標が小さいほど良い (損失系の場合はFalse, accuracyなどはTrue)
    eval_strategy="steps",                 # バリデーション評価のタイミング ("steps"だとNステップごと)
    save_strategy="steps",                 # モデル保存のタイミング ("steps"だとNステップごと)
)


"""
SFTTrainerを用いたデータローダ構築
"""
MAX_SEQ_LENGTH = 768         # 1サンプルの最大シーケンス長 (トークン数)。入出力文を連結した場合の最大トークン長。モデルやリソースに応じて調整する。FAQの回答+質問なら768が適切。
PACKING_ENABLED = False      # 複数の短い文を1シーケンスに "パッキング" して詰めて効率化するかどうか。Falseだと各サンプル1文ずつ扱う。macOS環境や少量データではFalse推奨。
NUM_OF_SEQUENCES = 1024      # SFTTrainerのデータ前処理 (packやchunk) 時に詰め込む最大シーケンス数。大規模データで効率を重視する際に使うが、通常はデフォルト値でOK。
CHARS_PER_TOKEN = 3.6        # 日本語文字列をトークンに換算するための目安。日本語は1トークンあたり平均3.6文字程度 (OpenAI/Gemma系換算値)。長さ見積もりや下処理に利用。

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,       # 3回連続で改善しなければ停止
    early_stopping_threshold=0.001,  # 改善の閾値
)

trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    args=training_args,
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback],  # Early Stoppingコールバック
    max_seq_length=MAX_SEQ_LENGTH,
    packing=PACKING_ENABLED,
)

if __name__ == '__main__':
    """
    SFTTrainerを用いた学習実行
    """
    # 既存のチェックポイントがあるかチェック
    last_ckpt = get_last_checkpoint(training_args.output_dir)
    if last_ckpt:
        print(f"既存のチェックポイントを発見しました: {last_ckpt}")
        print("チェックポイントから学習を再開します...")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("新しい学習を開始します...")
        trainer.train()

    """
    テストデータでの最終評価
    """
    print("\n--- テストデータでの最終評価 ---")
    
    # テスト評価時のメモリ制限を設定
    print("テスト評価用のメモリ制限を設定します...")
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.3"  # MPSメモリ使用量を30%に制限
    os.environ["MALLOC_ARENA_MAX"] = "2"  # メモリアリーナ数を制限
    os.environ["OMP_NUM_THREADS"] = "2"   # スレッド数を削減
    os.environ["MKL_NUM_THREADS"] = "2"   # MKLスレッド数を削減

    try:
        # テストデータセットをSFTTrainerのメソッドで事前トークナイズして評価
        # (SFTTrainerは学習/検証データしか自動処理しないため、テストデータを手動で同じ形式に変換する必要がある)
        test_dataset_prepared = trainer._prepare_dataset(
            test_dataset,
            tokenizer=tokenizer,
            packing=PACKING_ENABLED,
            dataset_text_field=None,
            max_seq_length=384,  # シーケンス長を384に制限
            formatting_func=formatting_func,
            num_of_sequences=NUM_OF_SEQUENCES,
            chars_per_token=CHARS_PER_TOKEN,
            remove_unused_columns=trainer.args.remove_unused_columns,
        )
        
        # メモリクリア
        gc.collect()
        torch.mps.empty_cache()
        
        test_results = trainer.evaluate(eval_dataset=test_dataset_prepared, metric_key_prefix="test")
        print(f"テスト損失: {test_results['test_loss']:.4f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            print(f"メモリ不足エラーが発生しました: {e}")
            print("テストデータの評価をスキップします。")
            test_results = {"test_loss": float('inf')}
        else:
            raise e
    except Exception as e:
        print(f"テストデータ評価中にエラーが発生しました: {e}")
        print("テストデータの評価をスキップします。")
        test_results = {"test_loss": float('inf')}
    finally:
        # テスト評価後に元の設定を復元
        print("テスト評価完了後、元の設定を復元します...")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["MALLOC_ARENA_MAX"] = "4"
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"

    """
    adapterの保存
    """
    # LoRA差分（adapter）のみ保存
    # 出力先は models/ 配下に統一し、末尾に生成日時 (JST) を追加
    JST = timezone(timedelta(hours=9))
    timestamp = datetime.now(JST).strftime("%Y%m%d-%H%M%S")
    adapter_path = f"models/{MODEL_NAME}-lora-{timestamp}"
    trainer.model.save_pretrained(adapter_path)

    print("\n--- 学習完了 ---")
    print(f"LoRA adapter saved to {adapter_path}")
    print(f"最終テスト損失: {test_results['test_loss']:.4f}")
    print("学習が正常に完了しました。")
