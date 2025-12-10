import torch
from transformers import AutoModelForCausalLM

# 1. Define paths
# base_path = "huggyllama/llama-7b" # Or your local base path
base_path = "Qwen/Qwen3-4B" # Or your local base path
# merged_path = "/scratch/pawsey1151/alexchen5/LongLoRA/models/20251203_100415_llama-7b_train_data_s80"
merged_path = "/scratch/pawsey1151/alexchen5/LongLoRA/models/20251203_135521_Qwen3-4B_add_atom_action_qa_format_s20"

print("Loading models (this consumes 2x VRAM, ensure you have space)...")
# Load small slices if possible, or map to CPU to save GPU memory
base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, device_map="cpu")
merged_model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.float16, device_map="cpu")

print("\n--- CHECKING EMBEDDINGS (trainable_params.bin) ---")
base_emb = base_model.get_input_embeddings().weight
merged_emb = merged_model.get_input_embeddings().weight

print(f"Base Shape: {base_emb.shape}")
print(f"Merged Shape: {merged_emb.shape}")

# Determine the common vocabulary size (usually 32000)
common_vocab_size = min(base_emb.shape[0], merged_emb.shape[0])

# Compare only the overlapping rows
# We slice [:common_vocab_size] to ignore the new 32001th token for the math check
diff = (base_emb[:common_vocab_size] - merged_emb[:common_vocab_size]).abs().sum()

if diff == 0:
    print("❌ FAILURE: The original embeddings are identical. 'trainable_params.bin' was NOT merged.")
else:
    print(f"✅ SUCCESS: Embeddings are different (Diff: {diff.item()}). 'trainable_params.bin' was applied.")

# Optional: Check if the new token exists and has content
if merged_emb.shape[0] > base_emb.shape[0]:
    print("✅ SUCCESS: Vocab size increased. The tokenizer resize worked.")

print("\n--- CHECKING ATTENTION LAYERS (LoRA Weights) ---")
# Check a random layer, e.g., layer 5's query projection
base_weight = base_model.model.layers[5].self_attn.q_proj.weight
merged_weight = merged_model.model.layers[5].self_attn.q_proj.weight

if torch.equal(base_weight, merged_weight):
    print("❌ FAILURE: Attention weights are identical. LoRA adapters were NOT merged.")
else:
    print("✅ SUCCESS: Attention weights are different. LoRA was applied.")

print("\n--- CHECKING VOCAB SIZE ---")
print(f"Base Vocab: {base_model.config.vocab_size}")
print(f"Merged Vocab: {merged_model.config.vocab_size}")