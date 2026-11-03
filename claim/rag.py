"""Structured cultural prompts, configurable local HF causal LLM, and KG evidence gate.

The supplied evidence checker checks lexical evidence coverage, not factual entailment.
Do NOT interpret it as a validated hallucination detector.
"""
import re
from .kg import KGStore


def build_prompt(span, dynasty, category, triples):
    evidence = '\n'.join(f'- {h} --{r}--> {t}' for h,r,t in triples) or '(No verified cultural triples available)'
    return (f'You are a classical Chinese literature scholar. Given the imagery span "{span}" '
            f'from a {dynasty}-dynasty poem, category {category}, and cultural triples:\n'
            f'{evidence}\nExplain the cultural symbolism and emotional connotation in 3–5 sentences. '
            'Do not invent historical facts not provided in the evidence. '
            'If the evidence is insufficient, explicitly state the uncertainty.')


def evidence_coverage(text, triples):
    """Fraction of distinct KG tail entities that are literally mentioned in the output.

    An inexpensive transparent heuristic, NOT semantic factual-claim precision/recall.
    """
    terms = {t for _,_,t in triples if t}
    if not terms:
        return 0.0
    return sum(term in text for term in terms)/len(terms)


class CulturalGenerator:
    def __init__(self, kg_store, model_name=None, device_map='auto'):
        self.kg = kg_store
        self.generator = None
        if model_name:
            from transformers import pipeline
            self.generator = pipeline('text-generation',model=model_name,device_map=device_map)

    def generate(self, span, dynasty, category, max_new_tokens=180, threshold=.7):
        triples = self.kg.retrieve(span,top_k=5)
        prompt = build_prompt(span,dynasty,category,triples)
        if self.generator is None:
            # explicitly identified extractive fallback for CPU demonstration
            content = ('；'.join(f'{h}的{r}为{t}' for h,r,t in triples)
                       if triples else '知识图谱中没有足够证据，无法可靠解释该意象。')
            mode = 'extractive_fallback'
        else:
            output = self.generator(prompt,max_new_tokens=max_new_tokens,do_sample=False)
            generated = output[0]['generated_text']
            content = generated[len(prompt):] if generated.startswith(prompt) else generated
            mode = 'causal_lm'
            if triples and evidence_coverage(content,triples) < threshold:
                refined_prompt = prompt + '\nVerified facts: ' + '; '.join(f'{h}-{r}-{t}' for h,r,t in triples)
                output = self.generator(refined_prompt,max_new_tokens=max_new_tokens,do_sample=False)
                generated = output[0]['generated_text']
                content = generated[len(refined_prompt):] if generated.startswith(refined_prompt) else generated
        return {'span':span,'category':category,'dynasty':dynasty,'triples':triples,
                'prompt':prompt,'generated_text':content,'mode':mode,
                'evidence_coverage':evidence_coverage(content,triples)}
