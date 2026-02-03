
import os
import sys

def prepare_repo():
    # Configuration
    source_file = os.path.join("out_chinese", "ckpt.pt")
    parts_dir = "model_parts"
    chunk_size = 90 * 1024 * 1024 # 90MB (keep under 100MB limit safely)

    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found.")
        return

    os.makedirs(parts_dir, exist_ok=True)
    
    print(f"Splitting {source_file} into {parts_dir}...")
    part_num = 0
    with open(source_file, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            part_num += 1
            part_name = f"part_{part_num:03d}.bin"
            part_path = os.path.join(parts_dir, part_name)
            with open(part_path, 'wb') as p:
                p.write(chunk)
            print(f"Created {part_name}")
    
    print("Creating join_model.py...")
    join_script = f'''
import os
import sys

def join_model():
    parts_dir = "model_parts"
    output_dir = "out_chinese"
    output_file = os.path.join(output_dir, "ckpt.pt")
    
    os.makedirs(output_dir, exist_ok=True)
    
    parts = sorted([f for f in os.listdir(parts_dir) if f.startswith("part_")])
    
    if not parts:
        print("No parts found in model_parts!")
        input("Press Enter to exit...")
        return

    print(f"Joining {{len(parts)}} parts into {{output_file}}...")
    
    with open(output_file, 'wb') as outfile:
        for part in parts:
            part_path = os.path.join(parts_dir, part)
            print(f"Processing {{part}}...")
            with open(part_path, 'rb') as infile:
                outfile.write(infile.read())
                
    print("Done! Model restored.")

if __name__ == "__main__":
    join_model()
'''
    with open("join_model.py", "w", encoding='utf-8') as f:
        f.write(join_script)

    print("Creating .gitignore...")
    with open(".gitignore", "w") as f:
        f.write("out_chinese/\n") # Ignore the original large file and folder
        f.write("*.zip\n")
        f.write("__pycache__/\n")

    print("Creating README_GITHUB.md...")
    readme = """# My LLM

This repository contains my trained LLM.

## How to use

1. Clone this repository.
2. Run `python join_model.py` to restore the model weights.
3. Run `python generate.py` to start the chat.

## Files
- `model.py`: Model definition
- `generate.py`: Inference script
- `model_parts/`: Split model weights (due to GitHub limits)
"""
    with open("README_GITHUB.md", "w", encoding='utf-8') as f:
        f.write(readme)

    print("Preparation complete.")

if __name__ == "__main__":
    prepare_repo()
