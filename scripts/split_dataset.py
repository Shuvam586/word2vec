import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def split():
    file_path = input("Name of txt file you wish to split: ")
    DATA_PATH = PROJECT_ROOT / "data" / (file_path)
    
    with open(DATA_PATH, "r") as f:
        text = f.read().split(" ")
    
    len_text = len(text)
    tiny = text[:int((0.01*0.01)*len_text)]
    small = text[:int((2*0.01)*len_text)]
    medium = text[:int((30*0.01)*len_text)]

    with open(PROJECT_ROOT / "data" / "tiny.txt", "w") as f:
        f.write(" ".join(tiny))

    with open(PROJECT_ROOT / "data" / "small.txt", "w") as f:
        f.write(" ".join(small))

    with open(PROJECT_ROOT / "data" / "medium.txt", "w") as f:
        f.write(" ".join(medium))

    if file_path!='raw.txt':
        os.rename(file_path, 'raw.txt')

split()