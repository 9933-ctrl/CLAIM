"""Recognition + optional KG-based reranking + cultural meaning generation."""
import argparse
import json
import torch
from transformers import AutoTokenizer
from claim.data import encode_record
from claim.model import ImageryRecognizer
from claim.fusion import KGFusionClassifier
from claim.kg import KGStore
from claim.rag import CulturalGenerator
from claim.spans import decode_spans
from claim.config import CATEGORIES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--text',required=True)
    p.add_argument('--dynasty',default='Tang')
    p.add_argument('--checkpoint',default='outputs/best.pt')
    p.add_argument('--kg',default='data/triples.jsonl')
    p.add_argument('--kg-embeddings',default=None)
    p.add_argument('--generator',default=None,help='HF causal LLM, e.g. Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--no-kg-rerank',action='store_true')
    args=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    tokenizer=AutoTokenizer.from_pretrained('outputs/tokenizer',use_fast=True)
    model=ImageryRecognizer(checkpoint['backbone']).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    kg=KGStore(args.kg,args.kg_embeddings)
    generator=CulturalGenerator(kg,model_name=args.generator)
    fusion=None
    if checkpoint.get('fusion') is not None:
        fusion=KGFusionClassifier(kg_dim=checkpoint['kg_dim']).to(device)
        fusion.load_state_dict(checkpoint['fusion']);fusion.eval()
    encoded=encode_record({'text':args.text,'spans':[]},tokenizer,128)
    tensors={k:torch.tensor(v,dtype=torch.long,device=device).unsqueeze(0)
             for k,v in encoded.items() if k in ('input_ids','attention_mask','label_mask')}
    with torch.no_grad():
        out=model(**tensors)
        positions=[i for i,m in zip(encoded['char_indices'],encoded['label_mask']) if m]
        spans=decode_spans(args.text,out['tags'][0],positions)
        for span in spans:
            if fusion and args.kg_embeddings and not args.no_kg_rerank:
                indices=[idx for idx,pos in enumerate(positions) if span['start']<=pos<span['end']]
                if indices:
                    context=out['features'][0,indices].mean(0).unsqueeze(0)
                    kg_vector=kg.embedding(span['text']).to(device).unsqueeze(0)
                    score=torch.softmax(fusion(context,kg_vector),dim=-1)[0]
                    category=int(score.argmax())
                    span['label']=CATEGORIES[category]
                    span['category_confidence']=float(score[category])
            span['interpretation']=generator.generate(span['text'],args.dynasty,span['label'])
    print(json.dumps({'text':args.text,'spans':spans},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
