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
    
    This function downloads 75 million words of pre-training data from the 
    "karpathy/climbmix-400b-shuffle" dataset on HuggingFace. The data is stored 
    in a text file named "climbmix.txt" within the "data" directory.
    
    The download process:
    - Loads the dataset with streaming=True for memory efficiency
    - Counts words and updates progress bar in real-time
    - Writes each text row to the output file
    - Stops when the target word count (75M) is reached
    
    '''
    
    data=load_dataset(
        pre_train_path,
        split="train",
        streaming=True 
    )
    count=0
    target_words=200_000_000
    with open(os.path.join(ROOT_DIR,"climbmix_200.txt"),"w",encoding="utf-8") as f:
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


def download_mine_q_a():
    ''' Downloads the Minecraft Q&A dataset from HuggingFace to train tokenizer.
    
    This function downloads 20 million words of Q&A data from the 
    "minhaozhang/minecraft-question-answer-630k" dataset on HuggingFace. The data 
    is stored in a text file named "mine_q_a.txt" within the "data" directory.
    
    The download process:
    - Loads the dataset with streaming=True for memory efficiency
    - Counts words (question + answer) and updates progress bar in real-time
    - Writes question and answer pairs to the output file
    - Stops when the target word count (20M) is reached
    

    '''

    data=load_dataset(
        minecraft_path2,
        split="train",
        streaming=True
    )

    count=0
    target_words=20_000_000
    with open(os.path.join(ROOT_DIR,"mine_q_a.txt"),"w",encoding="utf-8") as f:
        with tqdm(total=target_words,desc="Mine_Q&A",unit="words",mininterval=0.1,miniters=5) as pbar:
            for row in data:
                question=row["question"]
                answer=row["answer"]
                words=len(question.split())+len(answer.split())
                count+=words
                pbar.update(words)
    
                f.write(question+"\n")
                f.write(answer+"\n\n")
    
                if count>=target_words:
                    break

    print("Count of words :",count)


def download_mine_wiki():
    ''' Downloads the MineWiki dataset from HuggingFace to train tokenizer.
    
    This function downloads 5 million words of training data from the 
    "lparkourer10/minecraft-wiki" dataset on HuggingFace. The data is stored 
    in a text file named "mine_wiki.txt" within the "data" directory.
    
    The download process:
    - Loads the dataset with streaming=True for memory efficiency
    - Counts words (question + answer) and updates progress bar in real-time
    - Writes question and answer pairs to the output file
    - Stops when the target word count (5M) is reached
    
    '''

    data=load_dataset(
        minecraft_path,
        split="train",
        streaming=True
    )
    count=0
    target_words=5_000_000
    with open(os.path.join(ROOT_DIR,"mine_wiki.txt"),"w",encoding="utf-8") as f:
        with tqdm(total=target_words,desc="Mine_Wiki",unit="words",mininterval=0.1,miniters=5) as pbar:
            for row in data:
                question=row["question"]
                answer=row["answer"]
                words=len(question.split())+len(answer.split())
                count+=words
                pbar.update(words)

                f.write(question+"\n")
                f.write(answer+"\n\n")

                if count>=target_words:
                    break

    print("Count of words :",count)



if __name__=="__main__":
    os.makedirs(ROOT_DIR,exist_ok=True)
    print("Downling files...")

    download_climbmix()
    print("Climbix downloaded...\n")

    # download_mine_q_a()
    # print("Q&A downloaded...\n")

    # download_mine_wiki()
    # print("Mine-Wiki downloaded...\n")

    # print("Downloading completed.")