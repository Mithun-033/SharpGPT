from datasets import load_dataset
import os 
from tqdm import tqdm

pre_train_path="karpathy/climbmix-400b-shuffle"
minecraft_path="lparkourer10/minecraft-wiki"
minecraft_path2="minhaozhang/minecraft-question-answer-630k"

ROOT_DIR="data"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"


def download_climbmix():
    ''' Downloads the climbmix dataset from HuggingFace to train tokenizer.
    
    This function downloads 1 billion words of pre-training data from the 
    "karpathy/climbmix-400b-shuffle" dataset on HuggingFace. The data is stored 
    in a text file named "climbmix.txt" within the "data" directory.
    
    The download process:
    - Loads the dataset with streaming=True for memory efficiency
    - Counts words and updates progress bar in real-time
    - Writes each text row to the output file
    - Stops when the target word count (1B) is reached
    
    '''
    
    data=load_dataset(
        pre_train_path,
        split="train",
        streaming=True 
    )
    count=0
    target_words=1_000_000_000
    with open(os.path.join(ROOT_DIR,"climbmix.txt"),"w",encoding="utf-8") as f:
        with tqdm(total=target_words,desc="Climbmix",unit="words",mininterval=0.1,miniters=5) as pbar:
            for row in data:
                text=row["text"]
                words=len(text.split())
                count+=words
                pbar.update(words)

                f.write(text+"\n")

                if count>=target_words:
                    break

    print("Count of words :",count)


if __name__=="__main__":
    os.makedirs(ROOT_DIR,exist_ok=True)
    print("Downloading files...")

    download_climbmix()
    print("Climbix downloaded...\n")
