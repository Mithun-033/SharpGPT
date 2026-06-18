from tokenizers import Tokenizer
from datasets import load_dataset
import os
from tqdm import tqdm
import json

DATA_DIR="data"
TOK_DIR="tokenizers_dir"
path="mansaripo/ClimbMix_shuffled"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"

def download_unseen_climbmix():
    ''' Downloads the unseen portion of the climbmix dataset for tokenizer evaluation.
    
    This function downloads the remaining 5 billion words of the climbmix dataset from HuggingFace. 
    The data is stored in a text file named "climbmix_unseen.txt" within the "data" directory. 
    The download process is similar to the initial download, but it continues from where the previous download left off.
    
    The function:
    - Loads the dataset with streaming=True for memory efficiency
    - Counts words and updates progress bar in real-time
    - Writes each text row to the output file
    - Stops when the target word count (50mil) is reached
    
    '''
    
    data=load_dataset(
        path,
        split="train",
        streaming=True 
    )
    count=0
    target_words=50_000_000
    with open(os.path.join(DATA_DIR,"climbmix_unseen.txt"),"w",encoding="utf-8") as f:
        with tqdm(total=target_words,desc="Climbmix Unseen",unit="words",mininterval=0.1,miniters=5) as pbar:
            for row in data:

                text=row["text"]
                words=len(text.split())
                count+=words
                pbar.update(words)

                f.write(text+"\n")

                if count>=target_words:
                    break

    print("Count of words :",count)

    
def check_compression():
    ''' Checks the compression ratio of the trained tokenizers on the unseen portion of the climbmix dataset.
    
    This function evaluates the compression efficiency of the trained tokenizers (e.g., tokenizer_49k.json and tokenizer_32k.json) by encoding the unseen climbmix dataset and calculating the average number of tokens per word. The results are printed to the console for comparison.
    
    The function:
    - Loads each trained tokenizer
    - Reads the unseen climbmix dataset line by line
    - Encodes each line using the tokenizer and counts tokens and words
    - Calculates and prints the average tokens per word for each tokenizer

    '''
    tokenizers=[
        os.path.join(TOK_DIR,"tokenizer_32k.json"),
        os.path.join(TOK_DIR,"tokenizer_32k_whitespace.json"),
        os.path.join(TOK_DIR,"tokenizer_49k.json"),
        os.path.join(TOK_DIR,"tokenizer_49k_whitespace.json")
    ]

    for tok_path in tokenizers:
        tok=Tokenizer.from_file(tok_path)
        total_tokens=0
        total_words=0
        total_chars=0

        with open(os.path.join(DATA_DIR,"climbmix_unseen.txt"),"r",encoding="utf-8") as f:
            with tqdm(total=50_000_000, desc=f"Evaluating {os.path.basename(tok_path)}", unit="words", mininterval=0.1, miniters=5) as pbar:
                for line in f:
                    total_chars+=len(line)

                    words=line.split()
                    curr_length=len(words)
                    total_words+=curr_length

                    pbar.update(curr_length)

                    tokens=tok.encode(line).ids
                    total_tokens+=len(tokens)

        tokens_per_word=total_tokens/total_words
        chars_per_token=total_chars/total_tokens

        print(
            f"Tokenizer: {os.path.basename(tok_path)}, "
            f"Tokens/Word: {tokens_per_word:.4f}, "
            f"Chars/Token: {chars_per_token:.4f}"
        )

        with open(os.path.join(TOK_DIR,"Compression_ratios.json"),"a") as f:
            json.dump({
                "tokenizer": os.path.basename(tok_path),
                "total_tokens": total_tokens,
                "total_words": total_words,
                "total_chars": total_chars,
                "tokens_per_word": tokens_per_word,
                "chars_per_token": chars_per_token
            },f,indent=4)
            f.write("\n")

if __name__=="__main__":
    os.makedirs(DATA_DIR,exist_ok=True)
    
    # print("Downloading unseen portion of ClimbMix dataset...")
    # download_unseen_climbmix()
    # print("Climbix Unseen downloaded...\n")

    print("Checking compression ratios...")
    check_compression()
    