import os


def process_gyafc_data(input_dir, output_file):
    informal_path = os.path.join(input_dir, "test.0")
    formal_path = os.path.join(input_dir, "test.1")
    
    with open(informal_path, "r", encoding="utf-8") as f:
        informal_texts = f.readlines()
    
    with open(formal_path, "r", encoding="utf-8") as f:
        formal_texts = f.readlines()
    
    informal_texts = [line.strip() for line in informal_texts if line.strip()]
    formal_texts = [line.strip() for line in formal_texts if line.strip()]
    
    assert len(informal_texts) == len(formal_texts), "Mismatch between informal and formal text counts"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for informal, formal in zip(informal_texts, formal_texts):
            f.write(f"{formal}\t1\n")  # 1 - formal text
            f.write(f"{informal}\t0\n")
    
    print(f"File saved: {output_file}")


if __name__ == "__main__":
    input_directory = "GYAFC/"
    output_file = "data/gyafc.txt"

    process_gyafc_data(input_directory, output_file)
