"""TransE knowledge-graph embedding model and lexical/fuzzy triple retriever."""
import json
import random
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def load_triples(path):
    triples = []
    with open(path, encoding='utf8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                triples.append((item['head'], item['relation'], item['tail']))
    return triples


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=256, margin=1.0):
        super().__init__()
        self.entities = nn.Embedding(num_entities, dim)
        self.relations = nn.Embedding(num_relations, dim)
        self.margin = margin
        nn.init.xavier_uniform_(self.entities.weight)
        nn.init.xavier_uniform_(self.relations.weight)

    def distance(self, triples):
        h, r, t = triples.unbind(-1)
        return (F.normalize(self.entities(h), dim=-1) + self.relations(r)
                - F.normalize(self.entities(t), dim=-1)).norm(p=1, dim=-1)

    def forward(self, positives, negatives):
        return F.relu(self.margin + self.distance(positives) - self.distance(negatives)).mean()


def train_transe(triples_path, output_path, dim=256, epochs=100, batch_size=512,
                 learning_rate=0.001, seed=42, device='cpu'):
    torch.manual_seed(seed)
    triples = load_triples(triples_path)
    if not triples:
        raise ValueError('KG contains no triples')
    entity_names = sorted({v for h, _, t in triples for v in (h, t)})
    relation_names = sorted({r for _, r, _ in triples})
    e2id = {v:i for i,v in enumerate(entity_names)}
    r2id = {v:i for i,v in enumerate(relation_names)}
    data = torch.tensor([(e2id[h], r2id[r], e2id[t]) for h,r,t in triples],dtype=torch.long)
    model = TransE(len(entity_names), len(relation_names), dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        order = torch.randperm(len(data))
        total = 0.0
        for indexes in order.split(batch_size):
            positives = data[indexes].to(device)
            negatives = positives.clone()
            replace_head = torch.rand(len(indexes), device=device) < 0.5
            random_entities = torch.randint(len(entity_names), (len(indexes),), device=device)
            negatives[replace_head,0] = random_entities[replace_head]
            negatives[~replace_head,2] = random_entities[~replace_head]
            optimizer.zero_grad()
            loss = model(positives, negatives)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indexes)
        print(f'TransE epoch {epoch+1}/{epochs} loss={total/len(data):.5f}')
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.cpu().state_dict(), 'entities':entity_names,
                'relations':relation_names,'dim':dim},output_path)
    return output_path


class KGStore:
    def __init__(self, triples_path, embedding_path=None):
        self.triples = load_triples(triples_path)
        self.by_head = defaultdict(list)
        for triple in self.triples:
            self.by_head[triple[0]].append(triple)
        self.names = list(self.by_head)
        self.embeddings = {}
        self.dim = 256
        if embedding_path:
            checkpoint = torch.load(embedding_path,map_location='cpu',weights_only=True)
            self.dim = int(checkpoint['dim'])
            weight = checkpoint['state_dict']['entities.weight']
            self.embeddings = dict(zip(checkpoint['entities'],weight))

    def match(self, term, threshold=.55):
        if term in self.by_head:
            return term
        if not self.names:
            return None
        candidate = max(self.names,key=lambda x: SequenceMatcher(None,term,x).ratio())
        if SequenceMatcher(None,term,candidate).ratio() >= threshold:
            return candidate
        return None

    def retrieve(self, term, top_k=5):
        match = self.match(term)
        return self.by_head.get(match,[])[:top_k] if match is not None else []

    def embedding(self, term):
        match = self.match(term)
        if match is None or match not in self.embeddings:
            return torch.zeros(self.dim)
        return self.embeddings[match].detach().clone()
