"""Exact-match labeled span precision/recall/F1 and generation metrics."""
from collections import defaultdict
from .config import CATEGORIES


def span_metrics(golds, predictions):
    assert len(golds)==len(predictions)
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for gold,pred in zip(golds,predictions):
        gt = {(int(s['start']),int(s['end']),s['label']) for s in gold}
        pt = {(int(s['start']),int(s['end']),s['label']) for s in pred}
        for s in gt & pt: tp[s[2]] += 1
        for s in pt - gt: fp[s[2]] += 1
        for s in gt - pt: fn[s[2]] += 1
    result = {}; all_tp=all_fp=all_fn=0
    for category in CATEGORIES:
        t,f,n = tp[category],fp[category],fn[category]
        p=t/(t+f) if t+f else 0.0
        r=t/(t+n) if t+n else 0.0
        result[category]={'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.0,
                          'support':t+n}
        all_tp+=t; all_fp+=f; all_fn+=n
    p=all_tp/(all_tp+all_fp) if all_tp+all_fp else 0
    r=all_tp/(all_tp+all_fn) if all_tp+all_fn else 0
    return {'macro_f1':sum(x['f1'] for x in result.values())/len(CATEGORIES),
            'micro_precision':p,'micro_recall':r,'micro_f1':2*p*r/(p+r) if p+r else 0,
            'categories':result}


def generation_metrics(references, hypotheses, use_bertscore=False):
    import sacrebleu
    from rouge_score import rouge_scorer
    # Chinese character tokenization for consistent BLEU/ROUGE on unsegmented Chinese.
    bleu = sacrebleu.corpus_bleu(hypotheses,[references],tokenize='char').score/100
    scorer = rouge_scorer.RougeScorer(['rougeL'],use_stemmer=False)
    scores = [scorer.score(' '.join(r),' '.join(h))['rougeL'].fmeasure
              for r,h in zip(references,hypotheses)]
    result = {'bleu4':bleu,'rougeL':sum(scores)/len(scores) if scores else 0.0}
    if use_bertscore:
        from bert_score import score
        _,_,f1 = score(hypotheses,references,lang='zh',verbose=False)
        result['bertscore_f1'] = float(f1.mean())
    return result
