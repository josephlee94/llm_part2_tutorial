from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import DPOTrainer, DPOConfig
import torch
from datasets import load_dataset

# ============================================================
# Load a pre-trained instruction-tuned model (skip SFT stage)
# ============================================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B params, already instruction-tuned

print(f"\nLoading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Model loaded successfully!")

# ============================================================
# Test prompts to evaluate model before/after DPO
# ============================================================
test_prompts = [
    "Give me a 5-step plan to stay productive while working from home.",
    "I feel extremely stressed and hopeless. What should I do right now?",
]

def format_test_prompt(user_message):
    """Format using the model's chat template"""
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def test_model(model, tokenizer, step):
    print(f"\n{'='*60}")
    print(f"Testing at step {step}")
    print(f"{'='*60}")
    model.eval()
    for prompt in test_prompts:
        formatted_prompt = format_test_prompt(prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt portion to show only the response
        response = completion[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        print(f"Prompt: {prompt}")
        print(f"Output: {response}\n")
    model.train()
    print(f"{'='*60}\n")

# Test before training
test_model(model, tokenizer, step=0)

# ============================================================
# Load DPO preference dataset
# ============================================================
DATASET_NAME = "PKU-Alignment/PKU-SafeRLHF"

print(f"\nLoading dataset '{DATASET_NAME}'")
raw_dataset = load_dataset(DATASET_NAME, split="train")

# Filter for safety-contrast pairs with moderate+ severity on the unsafe side
#def has_safety_contrast(example):
#    """Select examples where one response is safe and the other is moderately/severely unsafe."""
#    if example['is_response_0_safe'] == example['is_response_1_safe']:
#        return False
#    # Require the unsafe response to have severity >= 2 (moderate or severe harm)
#    if not example['is_response_0_safe']:
#        return example['response_0_severity_level'] >= 2
#    else:
#        return example['response_1_severity_level'] >= 2

#dataset = raw_dataset.filter(has_safety_contrast)
dataset = dataset.select(range(min(20000, len(dataset))))  # Use up to 20K examples

print(f"Dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"\nRAW FORMAT (first example):")
print(f"Prompt: {dataset[0]['prompt'][:200]}...")
print(f"Response 0: {dataset[0]['response_0'][:200]}...")
print(f"Response 1: {dataset[0]['response_1'][:200]}...")
print(f"Response 0 safe: {dataset[0]['is_response_0_safe']}")
print(f"Response 1 safe: {dataset[0]['is_response_1_safe']}")

# ============================================================
# Format data for DPO
# ============================================================
def format_dpo_example(example):
    """
    DPO expects three fields:
    - prompt: the user's input (formatted with chat template)
    - chosen: the preferred response
    - rejected: the less preferred response
    """
    # Build the prompt using the model's chat template
    messages = [{"role": "user", "content": example['prompt']}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # Adds the assistant turn start
    )

    # Note: You can replace with safer_response_id for safety
    chosen_id = example['better_response_id']
    
    # Assign chosen and rejected based on the ID
    if chosen_id == 0:
        chosen_text = example['response_0']
        rejected_text = example['response_1']
    else:
        chosen_text = example['response_1']
        rejected_text = example['response_0']

    # Add eos token to mark end of response
    chosen = f"{chosen_text}{tokenizer.eos_token}"
    rejected = f"{rejected_text}{tokenizer.eos_token}"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

formatted_dataset = dataset.map(
    format_dpo_example,
    remove_columns=dataset.column_names,
)

print("\nPREPROCESSED FORMAT (first example):")
print(f"Prompt: {formatted_dataset[0]['prompt'][:300]}...")
print(f"Chosen: {formatted_dataset[0]['chosen'][:200]}...")
print(f"Rejected: {formatted_dataset[0]['rejected'][:200]}...")

# ============================================================
# Split for eval
# ============================================================
train_dataset = formatted_dataset
eval_dataset = formatted_dataset.select(range(min(100, len(formatted_dataset))))

# ============================================================
# DPO Configuration
# ============================================================
OUTPUT_DIR = "./dpo-tinyllama-safe"

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    
    # Logging
    logging_steps=25,
    
    # Batch settings (same as your SFT script)
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    
    # Learning rate (lower than SFT since we're refining)
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.03,
    
    # DPO-specific: beta controls KL penalty
    # Higher = stay closer to original model, Lower = more aggressive optimization
    beta=0.05,
    
    # Sequence lengths
    max_length=1024,
    max_prompt_length=512,
    
    # Training duration
    max_steps=1000,
    
    # Checkpointing
    save_steps=500,
    
    # Performance
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,
    
    # Scheduler
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    
    # Logging
    report_to="wandb",
    run_name="tinyllama-dpo",
)

# ============================================================
# Test callback (same pattern as your SFT script)
# ============================================================
class SimpleTestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        test_model(kwargs['model'], tokenizer, state.global_step)

# ============================================================
# Train with DPO
# ============================================================
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Automatically creates a frozen copy as reference
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    callbacks=[SimpleTestCallback()],
)

print("\nStarting DPO training...")
trainer.train()

# ============================================================
# Save the final model
# ============================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to {OUTPUT_DIR}")
print("DPO training complete!")
