"""Evaluate the saved recognition model on a labeled held-out test JSONL."""
import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from claim.data import ImageryDataset
from claim.model import ImageryRecognizer
from train import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--test',default='data/test.jsonl')
    p.add_argument('--checkpoint',default='outputs/best.pt')
    p.add_argument('--output',default='outputs/test_metrics.json')
    args=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    tokenizer=AutoTokenizer.from_pretrained('outputs/tokenizer',use_fast=True)
    dataset=ImageryDataset(args.test,tokenizer)
    model=ImageryRecognizer(checkpoint['backbone']).to(device)
    model.load_state_dict(checkpoint['model'])
    result=evaluate(model,DataLoader(dataset,batch_size=16),dataset,device)
    from pathlib import Path
    Path(args.output).parent.mkdir(exist_ok=True,parents=True)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
