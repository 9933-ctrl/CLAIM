"""Fine-tune BERT-BiLSTM-CRF with auxiliary KG-fusion span classification."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from claim.config import Config
from claim.data import ImageryDataset
from claim.model import ImageryRecognizer
from claim.kg import KGStore
from claim.fusion import KGFusionClassifier, collect_gold_spans
from claim.spans import decode_spans
from claim.metrics import span_metrics


def evaluate(model, dataloader, dataset, device):
    model.eval()
    golds, predictions = [], []
    with torch.no_grad():
        for batch in dataloader:
            indices = batch.pop('record_index').tolist()
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(batch['input_ids'],batch['attention_mask'],batch['label_mask'])
            for i, idx in enumerate(indices):
                record=dataset.records[idx]
                char_positions=batch['char_indices'][i][batch['label_mask'][i].bool()].tolist()
                predictions.append(decode_spans(record['text'],outputs['tags'][i],char_positions))
                golds.append(record['spans'])
    return span_metrics(golds,predictions)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',default='data/train.jsonl')
    parser.add_argument('--dev',default='data/dev.jsonl')
    parser.add_argument('--kg',default='data/triples.jsonl')
    parser.add_argument('--kg-embeddings',default=None)
    parser.add_argument('--model',default=Config.backbone)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--no-kg',action='store_true')
    parser.add_argument('--no-crf',action='store_true',help='Reserved: softmax ablation not implemented')
    args=parser.parse_args()
    if args.no_crf:
        parser.error('The no-CRF ablation requires a separate softmax head; not implemented.')
    seed=42
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer=AutoTokenizer.from_pretrained(args.model,use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('Use a Fast tokenizer for character-aligned labels')
    train=ImageryDataset(args.train,tokenizer)
    dev=ImageryDataset(args.dev,tokenizer)
    training=DataLoader(train,batch_size=args.batch_size,shuffle=True)
    validation=DataLoader(dev,batch_size=args.batch_size,shuffle=False)
    model=ImageryRecognizer(args.model).to(device)
    kg = KGStore(args.kg,args.kg_embeddings) if not args.no_kg else None
    fusion=KGFusionClassifier(kg_dim=kg.dim).to(device) if kg else None
    parameters=list(model.parameters())+(list(fusion.parameters()) if fusion else [])
    optimizer=torch.optim.AdamW(parameters,lr=args.lr)
    total_steps=max(1,len(training)*args.epochs)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*total_steps),total_steps)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    best=-1
    for epoch in range(args.epochs):
        model.train()
        if fusion:fusion.train()
        cumulative=0.0
        for batch in training:
            record_indices=batch.pop('record_index').tolist()
            records=[train.records[i] for i in record_indices]
            batch={k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad()
            result=model(batch['input_ids'],batch['attention_mask'],batch['label_mask'],batch['labels'])
            loss=result['loss']
            if fusion:
                compact_indices=[]
                for i in range(len(records)):
                    compact_indices.append(batch['char_indices'][i][batch['label_mask'][i].bool()].tolist())
                collected=collect_gold_spans(result['features'],records,compact_indices,kg,device)
                if collected:
                    contexts,kg_vectors,classes=collected
                    logits=fusion(contexts,kg_vectors)
                    loss=loss+0.3*nn.functional.cross_entropy(logits,classes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.0)
            optimizer.step();scheduler.step()
            cumulative+=float(loss.detach())
        metrics=evaluate(model,validation,dev,device)
        print(f"epoch {epoch+1}: loss={cumulative/max(1,len(training)):.4f} dev_macro_f1={metrics['macro_f1']:.4f}")
        if metrics['macro_f1']>best:
            best=metrics['macro_f1']
            checkpoint={'model':model.state_dict(),'fusion':fusion.state_dict() if fusion else None,
                        'backbone':args.model,'kg_dim':kg.dim if kg else 256,'epoch':epoch+1}
            torch.save(checkpoint,output/'best.pt')
            tokenizer.save_pretrained(output/'tokenizer')
            (output/'dev_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
    print('Best dev macro-F1:',best,'checkpoint:',output/'best.pt')

if __name__=='__main__':main()
