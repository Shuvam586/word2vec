import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def read_corpus(file_path: str) -> str:
    DATA_PATH = PROJECT_ROOT / "data" / (file_path+'.txt')
    
    with open(DATA_PATH, "r") as f:
        text = f.read()

    return text

def tokenize(text: str, lowercase = True) -> list:
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    if lowercase:
        text = text.lower()
    
    tokens = list(text.split(" "))
    
    return tokens

def load_and_tokenize(file_path: str, lowercase = True) -> list:
    return tokenize(read_corpus(file_path), lowercase)