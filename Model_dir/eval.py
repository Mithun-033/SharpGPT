import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import argparse
from Model_Classes import GPT
from HyperParam_Classes import Config
from tokenizers import Tokenizer
import os

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
state_dict=torch.load("C:\\Users\\mithu\\Desktop\\Mithun\\MineGPT\\fine_tuned_model_0.pt",map_location=DEVICE)

new_state_dict={}

for k,v in state_dict.items():
    if k.startswith("_orig_mod."):
        k=k[len("_orig_mod."):]
    new_state_dict[k]=v

model=GPT(Config()).to(DEVICE)
model.load_state_dict(new_state_dict)

model.eval()
tokenizer=Tokenizer.from_file("C:\\Users\\mithu\\Desktop\\Mithun\\MineGPT\\tokenizers_dir\\tokenizer_49k_whitespace.json")


#####################################################################
# MODEL INTERFACE
#####################################################################


def encode(text):
    return tokenizer.encode(text).ids


@torch.no_grad()
def get_logits(ids):
    x=torch.tensor(ids,dtype=torch.long,device=DEVICE).unsqueeze(0)
    logits=model(x)
    return logits[0]


@torch.no_grad()
def logprob(context,continuation):
    full=context+continuation

    full_ids=encode(full)
    ctx_ids=encode(context)

    logits=get_logits(full_ids)

    log_probs=F.log_softmax(logits[:-1],dim=-1)

    score=0.0

    for i in range(len(ctx_ids),len(full_ids)):
        token=full_ids[i]
        score+=log_probs[i-1,token].item()

    return score


#####################################################################
# GENERIC MULTIPLE CHOICE
#####################################################################


def mcq_score(prompt,choices):
    scores=[]

    for choice in choices:
        scores.append(logprob(prompt," "+choice))

    return max(range(len(scores)),key=lambda i:scores[i])


#####################################################################
# HELLASWAG
#####################################################################


def eval_hellaswag(limit=None):
    ds=load_dataset("hellaswag","default",split="validation")

    correct=0
    total=0

    for row in tqdm(ds):
        pred=mcq_score(row["ctx"],row["endings"])

        if pred==int(row["label"]):
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# PIQA
#####################################################################


def eval_piqa(limit=None):
    ds=load_dataset("piqa",split="validation")

    correct=0
    total=0

    for row in tqdm(ds):
        pred=mcq_score(
            row["goal"],
            [row["sol1"],row["sol2"]]
        )

        if pred==row["label"]:
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# WINOGRANDE
#####################################################################


def eval_winogrande(limit=None):
    ds=load_dataset(
        "winogrande",
        "winogrande_xl",
        split="validation"
    )

    correct=0
    total=0

    for row in tqdm(ds):
        sentence=row["sentence"]

        prompt=sentence.replace("_","")

        pred=mcq_score(
            prompt,
            [row["option1"],row["option2"]]
        )

        if pred+1==int(row["answer"]):
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# ARC
#####################################################################


def eval_arc(config,limit=None):
    ds=load_dataset(
        "ai2_arc",
        config,
        split="test"
    )

    correct=0
    total=0

    for row in tqdm(ds):
        choices=row["choices"]["text"]
        labels=row["choices"]["label"]

        pred=mcq_score(row["question"],choices)

        pred_label=labels[pred]

        if pred_label==row["answerKey"]:
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# OPENBOOKQA
#####################################################################


def eval_openbookqa(limit=None):
    ds=load_dataset(
        "openbookqa",
        "main",
        split="test"
    )

    correct=0
    total=0

    for row in tqdm(ds):
        pred=mcq_score(
            row["question_stem"],
            row["choices"]["text"]
        )

        label=row["choices"]["label"][pred]

        if label==row["answerKey"]:
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# LAMBADA
#####################################################################


def eval_lambada(limit=None):
    ds=load_dataset(
        "lambada",
        split="test"
    )

    correct=0
    total=0

    for row in tqdm(ds):
        text=row["text"]

        words=text.split()

        context=" ".join(words[:-1])
        target=words[-1]

        pred_score=logprob(context," "+target)

        random_score=logprob(context," the")

        if pred_score>random_score:
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# MMLU
#####################################################################


def eval_mmlu(limit=None):
    ds=load_dataset(
        "cais/mmlu",
        "all",
        split="test"
    )

    correct=0
    total=0

    for row in tqdm(ds):
        pred=mcq_score(
            row["question"],
            row["choices"]
        )

        if pred==row["answer"]:
            correct+=1

        total+=1

        if limit and total>=limit:
            break

    return 100*correct/total


#####################################################################
# RUNNER
#####################################################################


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args=parser.parse_args()

    results={}

    results["hellaswag"]=eval_hellaswag(args.limit)
    results["piqa"]=eval_piqa(args.limit)
    results["winogrande"]=eval_winogrande(args.limit)
    results["arc_easy"]=eval_arc("ARC-Easy",args.limit)
    results["arc_challenge"]=eval_arc("ARC-Challenge",args.limit)
    results["openbookqa"]=eval_openbookqa(args.limit)
    results["lambada"]=eval_lambada(args.limit)
    results["mmlu"]=eval_mmlu(args.limit)

    print("\n"+"="*50)

    for k,v in results.items():
        print(f"{k:20s} {v:.2f}")

    print("="*50)


if __name__=="__main__":
    main()