
import os
import torch
import torch.nn.functional as F
import sys
import contextlib

# Force Windows Console to UTF-8 (CP65001)
if sys.platform == "win32":
    os.system("chcp 65001 > nul") 
    # sys.stdin/stdout are already valid if python started after this, 
    # but reconfiguring ensures Python knows about it.
    try:
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'out_chinese')
# out_dir = 'out'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 200 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # Forced due to RTX 5080 compatibility issues
dtype = 'float32'
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # usage of compile not supported on windows yet in this env
# -----------------------------------------------------------------------------

# model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# tokenizer
enc = tiktoken.get_encoding("gpt2")

def generate_streaming(instruction, input_context=None):
    # Format for Alpaca:
    # User: {instruction}
    # Input: {input} (optional)
    # Assistant:
    if input_context and input_context.strip():
        formatted_prompt = f"User: {instruction}\nInput: {input_context}\nAssistant:"
    else:
        formatted_prompt = f"User: {instruction}\nAssistant:"
    
    start_ids = enc.encode(formatted_prompt, allowed_special={"<|endoftext|>"})
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    print("Assistant: ", end='', flush=True)
    
    # run generation
    generated_ids = []
    print_len = 0
    with torch.no_grad():
        # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        with contextlib.nullcontext():
            for k in range(max_new_tokens):
                logits, _ = model(x)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # append to the sequence
                x = torch.cat((x, idx_next), dim=1)
                
                # accumulate tokens for correct decoding of multi-byte chars
                token = idx_next.item()
                generated_ids.append(token)
                
                # Use decode_bytes to get raw bytes, then decode manually to handle partials
                # errors='ignore' ensures we don't see replacement chars for incomplete bytes
                current_bytes = enc.decode_bytes(generated_ids)
                current_text = current_bytes.decode('utf-8', errors='ignore')
                
                # Stop if <|endoftext|>
                if token == 50256: # <|endoftext|>
                    break
                
                # Heuristic stop: If model tries to start a new "User:" turn
                if "User:" in current_text:
                    # check if it's a real stop or just random noise, but for now simple check
                    break

                # print only the *new* valid characters
                if len(current_text) > print_len:
                    new_text = current_text[print_len:]
                    print(new_text, end='', flush=True)
                    print_len += len(new_text)
                
                # stop if we see <|endoftext|> (though simple gpt2 might not predict it perfectly trained on raw text)
                # Alpaca data uses <|endoftext|> as separator
                # gpt2 tokenizer endoftext id is 50256
                if token == 50256: 
                    break
    print("\n")

def safe_input(prompt):
    print(prompt, end='', flush=True)
    if sys.platform == "win32":
        try:
            # Read raw bytes from stdin buffer to bypass console encoding issues
            line_bytes = sys.stdin.buffer.readline()
            if not line_bytes:
                raise EOFError
            # Decode using utf-8, ignoring errors to prevent crashes
            return line_bytes.decode('utf-8', errors='ignore').strip()
        except AttributeError:
            # Fallback if buffer is not available
            return input().strip()
    else:
        return input().strip()

if __name__ == "__main__":
    print(f"Model loaded from {ckpt_path}")
    print("Note: To ensure Chinese input works, please make sure your terminal font supports Chinese.")
    
    if len(sys.argv) > 1:
        # Non-interactive mode for quick testing
        # e.g. python generate.py "Translate this" "Hello World"
        instruction = sys.argv[1]
        input_context = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Instruction: {instruction}")
        if input_context:
            print(f"Input: {input_context}")
            
        generate_streaming(instruction, input_context)
    else:
        # Interactive mode
        print("Type 'q' to quit.")
        print("Tip: To use 'Input' context, type command first, press Enter, then type input context when asked.")
        while True:
            try:
                # Use safe_input instead of built-in input()
                instruction = safe_input("\nUser (Instruction): ")
                if instruction.lower() == 'q':
                    break
                if instruction == "":
                    continue
                
                # Ask for optional input
                input_context = safe_input("Input (Context/Optional): ")
                
            except EOFError:
                break
            except KeyboardInterrupt:
                break
                
            generate_streaming(instruction, input_context)
