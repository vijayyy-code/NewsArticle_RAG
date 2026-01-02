import os
import re
from pathlib import Path

# Emoji-only Unicode ranges (SAFE)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE
)

TARGET_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".txt", ".md"}

def clean_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)

def process_file(path: Path):
    try:
        original = path.read_text(encoding="utf-8")
        cleaned = clean_emojis(original)

        if original != cleaned:
            path.write_text(cleaned, encoding="utf-8")
            print(f" Cleaned emojis safely: {path}")
    except Exception as e:
        print(f" Skipped {path}: {e}")

def clean_project(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in TARGET_EXTENSIONS:
                process_file(file_path)

if __name__ == "__main__":
    PROJECT_FOLDER = "."  # current directory
    clean_project(PROJECT_FOLDER)
