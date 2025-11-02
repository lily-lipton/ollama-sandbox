from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from huggingface_hub import login
from transformers import TrainingArguments
from datasets import load_dataset
from kaggle_secrets import UserSecretsClient
import wandb

# Authentication setup
auth_manager = UserSecretsClient()
api_token = auth_manager.get_secret("Hugging_Face_Token")
metrics_token = auth_manager.get_secret("WANDB_TOKEN")

# Initialize authentication
login(api_token)
wandb.login(key=metrics_token)
monitoring = wandb.init(
    project='advanced_reasoning_enhancement_project',
    job_type="adaptation",
    anonymous="allow"
)

# Configuration parameters
context_length = 2048
data_type = None
memory_optimization = True

# Initialize foundation model
foundation_model, text_processor = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=context_length,
    dtype=data_type,
    load_in_4bit=memory_optimization,
    token=api_token,
)

# Define instruction template
instruction_template = """You are presented with a mathematical challenge that requires careful analysis and step-by-step problem solving. Your task is to examine the problem carefully and provide a well-structured solution.
Take a moment to understand the problem thoroughly before proceeding. Break down complex problems into simpler components and show your logical progression towards the solution.

### Expert Role:
You are a distinguished mathematician with expertise in:
- Advanced mathematical analysis and reasoning
- Complex problem-solving techniques
- Systematic calculation methods
- Mathematical proof construction
- Multiple solution approaches
- Error checking and verification

### Task Guidelines:
1. Read the problem carefully
2. Identify key information and variables
3. Plan your solution strategy
4. Execute calculations methodically
5. Verify your results
6. Explain your reasoning clearly

### Question:
{}

### Solution Process:
<think>
{}
"""

training_template = instruction_template

# Sample query for testing
test_query = """What is $(-1)^1+(-1)^2+\frac{(-1)^{2006}}{(-1)^{2006}}$?"""

# Configure for inference
FastLanguageModel.for_inference(foundation_model)
query_tokens = text_processor([instruction_template.format(test_query, "")], return_tensors="pt").to("cuda")

# Generate initial response
response_tokens = foundation_model.generate(
    input_ids=query_tokens.input_ids,
    attention_mask=query_tokens.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)

# Process response
decoded_response = text_processor.batch_decode(response_tokens)

# Load training data
raw_dataset = load_dataset(
    "rubenroy/GammaCorpus-CoT-Math-170k",
    split="train[0:500]",
    trust_remote_code=True
)

# Configure data processing
completion_token = text_processor.eos_token

def process_training_data(examples):
    queries = examples["input"]
    reasoning = examples["chain_of_thought"]
    answers = examples["output"]
    
    processed_entries = []
    for query, steps, answer in zip(queries, reasoning, answers):
        entry = training_template.format(query, steps, answer) + completion_token
        processed_entries.append(entry)
        
    return {
        "text": processed_entries,
    }

# Transform dataset
processed_dataset = raw_dataset.map(process_training_data, batched=True)

# Configure model adaptations
enhanced_model = FastLanguageModel.get_peft_model(
    foundation_model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Initialize training configuration
training_engine = SFTTrainer(
    model=enhanced_model,
    tokenizer=text_processor,
    train_dataset=processed_dataset,
    dataset_text_field="text",
    max_seq_length=context_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        warmup_steps=5,
        max_steps=50,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="model_snapshots",
    ),
)

# Execute training
training_stats = training_engine.train()
wandb.finish()

# Post-training inference
test_query = """Find $1 + 2\cdot 3 - 4 + 5.$"""

FastLanguageModel.for_inference(enhanced_model)
query_tokens = text_processor([instruction_template.format(test_query, "")], return_tensors="pt").to("cuda")

response_tokens = enhanced_model.generate(
    input_ids=query_tokens.input_ids,
    attention_mask=query_tokens.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)

decoded_response = text_processor.batch_decode(response_tokens)
print(decoded_response[0].split("### Response:")[0])

# Save enhanced model
output_path = "enhanced_reasoning_model"
foundation_model.save_pretrained(output_path)
text_processor.save_pretrained(output_path)
foundation_model.save_pretrained_merged(
    output_path, 
    text_processor,
    save_method="merged_16bit",
)
