import torch
from claim.crf import LinearCRF
from claim.spans import decode_spans
from claim.config import TAG2ID
from claim.data import character_tags
from claim.kg import KGStore
from claim.metrics import span_metrics


def test_crf_gradients_and_viterbi():
    model=LinearCRF(3)
    emissions=torch.randn(2,4,3,requires_grad=True)
    gold=torch.tensor([[1,2,0,0],[0,1,2,0]])
    mask=torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
    loss=model(emissions,gold,mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert emissions.grad is not None
    assert list(map(len,model.decode(emissions,mask)))==[3,4]


def test_offsets_and_spans():
    rec={'text':'明月照人','spans':[{'start':0,'end':2,'label':'Moon'}]}
    tags=character_tags(rec)
    assert tags==['B-Moon','I-Moon','O','O']
    pred=decode_spans(rec['text'],[TAG2ID[t] for t in tags])
    assert pred[0]['text']=='明月'
    assert span_metrics([rec['spans']],[pred])['micro_f1']==1


def test_kg():
    store=KGStore('data/triples.jsonl')
    assert store.retrieve('明月',5)[0][0]=='明月'
