# Written by Yukang Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import argparse
import transformers
from peft import PeftModel
from typing import Dict

# ---------------- CONFIGURATION ---------------- #
# Force CPU execution to avoid ROCm/Bitsandbytes crashes
DEVICE = torch.device("cpu")
# ----------------------------------------------- #

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--peft_model', type=str, required=True)
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    args = parser.parse_args()
    return args

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def main(args):
    print(f"--- Starting Merge on {DEVICE} ---")
    print("Base Model:", args.base_model)
    print("LoRA Adapter:", args.peft_model)

    # 1. Load Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 2. Load Base Model (Strictly CPU, No Device Map)
    print("Loading base model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, # Important for RAM management
        device_map=None,        # DISABLE auto device map to prevent nesting issues
    ).to(DEVICE)

    # 3. Resize Embeddings
    print("Resizing embeddings...")
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # 4. Load Trainable Params (Embeddings/Norms)
    trainable_params_path = os.path.join(args.peft_model, "trainable_params.bin")
    if os.path.isfile(trainable_params_path):
        print(f"Loading trainable_params from {trainable_params_path}")
        model.load_state_dict(torch.load(trainable_params_path, map_location=DEVICE), strict=False)
    else:
        print("WARNING: trainable_params.bin not found!")

    # 5. Load LoRA Adapters
    print("Loading LoRA adapters...")
    # Ensure we are looking for adapter_model.bin
    if not os.path.exists(os.path.join(args.peft_model, "adapter_model.bin")):
        print("CRITICAL ERROR: 'adapter_model.bin' not found. Did you rename pytorch_model.bin?")
        return

    model = PeftModel.from_pretrained(
        model,
        args.peft_model,
        device_map=None, # DISABLE auto device map
        torch_dtype=torch.float16,
    )

    # 6. Merge
    print("Merging weights...")
    model = model.merge_and_unload()

    # 7. Save
    print(f"Saving to {args.save_path}...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("Done saving!")

if __name__ == "__main__":
    args = parse_config()
    main(args)