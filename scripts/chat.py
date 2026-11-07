# Chat with the instruction-tuned LoRA model.
#   python -m scripts.chat
#   python -m scripts.chat --temperature 0 --checkpoint outputs/checkpoints/instruction_lora.pth
#
# load: config -> GPTModel -> add LoRA (same rank as training) -> load checkpoint
# loop: read instruction -> wrap in prompt template -> generate -> print only the response

import argparse
import torch
import tiktoken

from model.GPTModel import GPTModel
from model.lora import replace_linear_with_lora
from utils.GenerateText import generate_text_simple
from data.instruction_dataloader import format_input

# must match the config used in instruction_train.py
CFG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,   # gpt2-medium
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
LORA_ALPHA = 16    # must match training (alpha is not stored in the checkpoint)
EOS_ID = 50256     # <|endoftext|>


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    # read the LoRA rank from the checkpoint itself: A has shape (in_dim, rank)
    rank = state["trf_blocks.0.attn.W_q.lora.A"].shape[1]

    model = GPTModel(CFG)
    replace_linear_with_lora(model, rank=rank, alpha=LORA_ALPHA)
    model.load_state_dict(state)  # full state dict: GPT-2 weights + LoRA weights
    model.to(device).eval()
    print(f"loaded {checkpoint} (LoRA rank={rank}, alpha={LORA_ALPHA}) on {device}")
    return model


def answer(model, tokenizer, device, instruction, user_input, args):
    prompt = format_input({"instruction": instruction, "input": user_input})
    prompt += "\n\n### Response:\n"

    idx = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    out = generate_text_simple(
        model=model, idx=idx,
        max_new_tokens=args.max_new_tokens,
        context_size=CFG["context_length"],
        eos_id=EOS_ID, top_k=args.top_k, temperature=args.temperature,
    )
    new_tokens = out[0, idx.shape[1]:].tolist()  # drop the prompt, keep only the answer
    return tokenizer.decode(new_tokens).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/checkpoints/instruction_lora.pth")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = load_model(args.checkpoint, device)
    tokenizer = tiktoken.get_encoding("gpt2")

    print("\nType an instruction. Optional: add an input after '||'")
    print("  e.g.  Convert to passive || The chef cooks the meal every day.")
    print("Commands: /temp 0.3   /quit\n")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/temp"):
            try:
                args.temperature = float(line.split()[1])
                print(f"temperature = {args.temperature}\n")
            except (IndexError, ValueError):
                print("usage: /temp 0.7\n")
            continue

        instruction, _, user_input = line.partition("||")
        reply = answer(model, tokenizer, device, instruction.strip(), user_input.strip(), args)
        print(f"GPT: {reply}\n")


if __name__ == "__main__":
    main()
