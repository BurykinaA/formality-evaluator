import os
import openai
import random
from tqdm import tqdm

def generate_examples(num_examples=500):
    formal_texts = []
    informal_texts = []
    
    topics = [
        "work and career", "education", "technology", "travel", 
        "food and cooking", "health", "relationships", "hobbies", "sports",
        "movies and TV shows", "music", "books", "social media",
        "shopping", "finance", "weather", "news", "holidays"
    ]
    
    formal_prompt = """Generate one sentence of formal speech on the topic "{topic}". 
    The sentence should be written in formal language, as in business correspondence or academic work.
    Return only the sentence without additional comments."""
    
    informal_prompt = """Generate one sentence of informal speech on the topic "{topic}". 
    The sentence should be written in conversational language, as in correspondence with friends.
    Use slang, abbreviations, and informal expressions.
    Return only the sentence without additional comments."""
    
    print(f"Generating {num_examples} examples of formal and informal speech...")
    
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    for _ in tqdm(range(num_examples)):
        topic = random.choice(topics)
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that generates text examples."},
                    {"role": "user", "content": formal_prompt.format(topic=topic)}
                ],
                max_tokens=100,
                temperature=0.7
            )
            formal_text = response.choices[0].message.content.strip()
            formal_texts.append(formal_text)
        except Exception as e:
            print(f"Error generating formal text: {e}")
            continue
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that generates text examples."},
                    {"role": "user", "content": informal_prompt.format(topic=topic)}
                ],
                max_tokens=100,
                temperature=0.7
            )
            informal_text = response.choices[0].message.content.strip()
            informal_texts.append(informal_text)
        except Exception as e:
            print(f"Error generating informal text: {e}")
            continue
        
    
    return formal_texts, informal_texts

def save_to_file(formal_texts, informal_texts, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for formal, informal in zip(formal_texts, informal_texts):
            f.write(f"{formal}\t1\n")
            f.write(f"{informal}\t0\n")
    
    print(f"File saved: {output_file}")

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OpenAI API key via the OPENAI_API_KEY environment variable")
        exit(1)
    
    output_file = "data/gpt_generated.txt"
    num_examples = 500
    
    formal_texts, informal_texts = generate_examples(num_examples)
    
    if len(formal_texts) != len(informal_texts):
        print(f"Warning: number of formal ({len(formal_texts)}) and informal ({len(informal_texts)}) texts does not match")
        min_len = min(len(formal_texts), len(informal_texts))
        formal_texts = formal_texts[:min_len]
        informal_texts = informal_texts[:min_len]
    
    save_to_file(formal_texts, informal_texts, output_file)
    print(f"Generated and saved {len(formal_texts)} pairs of examples")
