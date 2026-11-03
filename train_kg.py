import argparse
from claim.kg import train_transe

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--triples',default='data/triples.jsonl')
    p.add_argument('--output',default='outputs/transe.pt')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--dim',type=int,default=256)
    args=p.parse_args()
    train_transe(args.triples,args.output,dim=args.dim,epochs=args.epochs)
