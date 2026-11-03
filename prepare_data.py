"""Create deterministic poem-level splits from an annotated JSONL source."""
import argparse
from claim.data import read_jsonl,split_by_poem,write_jsonl

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output-dir',default='data')
    args=parser.parse_args()
    sets=split_by_poem(read_jsonl(args.input))
    for name,records in zip(('train','dev','test'),sets):
        write_jsonl(records,f'{args.output_dir}/{name}.jsonl')
        print(name,len(records))
