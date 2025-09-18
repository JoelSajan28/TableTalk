from __future__ import annotations
from typing import Dict, List, Set, Tuple
import re, difflib

_GENERIC_TAILS = (
    "implementation", "implementations", "process", "processes", "procedure", "procedures",
    "details", "overview", "report", "reports", "plan", "plans", "policy", "policies",
    "manual", "system", "deployment", "deployments", "status", "update", "updates",
)
_WORD = r"[A-Za-z0-9_\-/]+"

def build_vocab_from_profiles(profiles: List[Dict]) -> Set[str]:
    vocab: Set[str] = set()
    for p in profiles:
        vocab.add(p["name"].lower())
        for c in p.get("columns", []) or []:
            vocab.add(str(c).lower())
        for ex in p.get("example_values", []) or []:
            exl = str(ex).lower()
            vocab.add(exl)
            for tok in re.findall(_WORD, exl):
                vocab.add(tok)
    return {v for v in vocab if v and v != "-"}

def strip_generic_tails(q: str) -> str:
    pattern = re.compile(rf"\b({_WORD})\s+({'|'.join(_GENERIC_TAILS)})\b", flags=re.I)
    prev = None
    cur = q
    while prev != cur:
        prev = cur
        cur = pattern.sub(r"\1", cur)
    return cur

def focus_terms(q: str, vocab: Set[str]) -> List[str]:
    tokens = re.findall(_WORD, q.lower())
    uniq: List[str] = []
    for t in tokens:
        if t in uniq:
            continue
        if t in vocab:
            uniq.append(t)
            continue
        near = difflib.get_close_matches(t, vocab, n=1, cutoff=0.86)
        if near:
            uniq.append(near[0])
    return uniq[:6]

def refine_question(original_q: str, profiles: List[Dict]) -> Tuple[str, List[str]]:
    q = (original_q or "").strip()
    q1 = strip_generic_tails(q)
    vocab = build_vocab_from_profiles(profiles)
    focuses = focus_terms(q1, vocab)
    refined = q1 if q1 != q else q
    return refined, focuses
