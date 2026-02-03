
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

    print(f"Joining {len(parts)} parts into {output_file}...")
    
    with open(output_file, 'wb') as outfile:
        for part in parts:
            part_path = os.path.join(parts_dir, part)
            print(f"Processing {part}...")
            with open(part_path, 'rb') as infile:
                outfile.write(infile.read())
                
    print("Done! Model restored.")

if __name__ == "__main__":
    join_model()
