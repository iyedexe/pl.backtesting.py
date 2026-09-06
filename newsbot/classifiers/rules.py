"""
Keyword / regex classifier. No external dependencies, deterministic, fast, and
easy to extend: add a `(pattern, category, weight)` row.

Weights of every matched rule are summed and clamped to [-1, 1]. Bearish rules
carry negative weights so that "beats estimates but cuts guidance" nets out
negative instead of triggering a long.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from ..models import Classification, NewsItem
from .base import Classifier

Rule = Tuple[str, str, float]

_EST = r'(?:estimates?|expectations?|consensus|forecasts?|street|views?|projections?)'

DEFAULT_RULES: List[Rule] = [
    # --- earnings -------------------------------------------------------
    (rf'\b(?:beats?|tops?|exceeds?|surpass(?:es|ed)?|crushes|blows? past|ahead of)\b.{{0,60}}\b{_EST}\b',
     'earnings_beat', 0.7),
    (rf'\b{_EST}\b.{{0,20}}\b(?:beat|topped)\b', 'earnings_beat', 0.6),
    (r'\brecord\b.{0,30}\b(?:revenue|sales|profit|earnings|quarter|results?|bookings)\b', 'earnings_beat', 0.5),
    (r'\b(?:revenue|sales|profit|eps|earnings|income)\b.{0,40}\b(?:rose|grew|jump(?:ed|s)?|surg(?:ed|es)?|'
     r'soar(?:ed|s)?|climb(?:ed|s)?|up)\b.{0,10}\d+(?:\.\d+)?\s?%', 'earnings_beat', 0.4),
    (r'\bstrong(?:er)?\b.{0,20}\b(?:quarter|results|demand|growth)\b', 'earnings_beat', 0.3),
    (rf'\b(?:misses?|missed|fall(?:s)? short|short of|below|lags?|trails?)\b.{{0,60}}\b{_EST}\b',
     'earnings_miss', -0.8),
    (r'\b(?:disappoint(?:s|ing|ed)?|weak(?:er)?(?:-than-expected)?)\b.{0,30}'
     r'\b(?:results?|quarter|sales|revenue|demand)\b', 'earnings_miss', -0.5),
    (r'\b(?:loss widens|wider loss|swings? to (?:a )?loss|net loss)\b', 'earnings_miss', -0.4),
    (r'\b(?:revenue|sales|profit|earnings)\b.{0,40}\b(?:fell|drop(?:ped|s)?|declin(?:ed|es)|plung(?:ed|es)|'
     r'slump(?:ed|s)?|down)\b.{0,10}\d+(?:\.\d+)?\s?%', 'earnings_miss', -0.4),
    # --- guidance -------------------------------------------------------
    (r'\b(?:raises?|raised|lifts?|lifted|boosts?|boosted|hikes?|ups)\b.{0,50}\b(?:guidance|outlook|forecast|'
     r'full[- ]year|fy ?\d{2,4}|targets?)\b', 'guidance_raise', 0.8),
    (r'\b(?:guidance|outlook|forecast)\b.{0,30}\b(?:above|ahead of|tops?|exceeds?)\b', 'guidance_raise', 0.6),
    (r'\b(?:cuts?|lowers?|lowered|slashes?|slashed|trims?|reduces?|withdraws?|suspends?|pulls?)\b.{0,50}'
     r'\b(?:guidance|outlook|forecast|full[- ]year|targets?)\b', 'guidance_cut', -0.9),
    (r'\b(?:guidance|outlook|forecast)\b.{0,30}\b(?:below|short of|misses?|disappoints?|cut|lowered)\b',
     'guidance_cut', -0.7),
    # --- analysts -------------------------------------------------------
    (r'\bupgrad(?:e|ed|es|ing)\b', 'analyst_upgrade', 0.5),
    (r'\b(?:price target|target price|\bpt\b)\b.{0,30}\b(?:raised|lifted|hiked|boosted|increased)\b',
     'analyst_upgrade', 0.3),
    (r'\b(?:initiat(?:es|ed|ion)|coverage)\b.{0,40}\b(?:buy|outperform|overweight|strong buy)\b',
     'analyst_upgrade', 0.4),
    (r'\bdowngrad(?:e|ed|es|ing)\b', 'analyst_downgrade', -0.6),
    (r'\b(?:price target|target price)\b.{0,30}\b(?:cut|lowered|slashed|reduced|trimmed)\b', 'analyst_downgrade', -0.3),
    # --- biotech / regulatory -------------------------------------------
    (r'\bfda\b.{0,60}\b(?:approv(?:al|es|ed)|clear(?:s|ed|ance)|grants?|accepts?|authoriz(?:es|ed|ation))\b',
     'fda_approval', 0.9),
    (r'\b(?:approv(?:al|es|ed))\b.{0,40}\b(?:fda|ema|regulators?)\b', 'fda_approval', 0.7),
    (r'\bbreakthrough (?:therapy )?designation\b', 'fda_approval', 0.5),
    (r'\bpositive\b.{0,30}\b(?:topline|top-line|phase ?[123]|pivotal|data|results)\b', 'clinical_positive', 0.6),
    (r'\b(?:meets?|met|achiev(?:es|ed))\b.{0,30}\bprimary endpoint\b', 'clinical_positive', 0.8),
    (r'\b(?:fails?|failed|missed|did not meet)\b.{0,30}\b(?:primary )?endpoint\b', 'clinical_negative', -0.9),
    (r'\bfda\b.{0,60}\b(?:rejects?|declines?|complete response letter|\bcrl\b|refus(?:es|al)|warning letter)\b',
     'clinical_negative', -0.9),
    (r'\bclinical hold\b', 'clinical_negative', -0.8),
    (r'\b(?:product )?recall\b', 'clinical_negative', -0.4),
    # --- capital allocation ---------------------------------------------
    (r'\b(?:buyback|share repurchase|repurchase program|repurchase authorization)\b', 'buyback', 0.4),
    (r'\b(?:raises?|increases?|hikes?|boosts?)\b.{0,20}\bdividend\b', 'dividend_raise', 0.4),
    (r'\b(?:special dividend)\b', 'dividend_raise', 0.3),
    (r'\b(?:suspends?|cuts?|eliminates?|slashes?)\b.{0,20}\bdividend\b', 'dividend_cut', -0.6),
    (r'\b(?:public offering|secondary offering|stock offering|share offering|equity offering|'
     r'convertible (?:senior )?notes?|at-the-market|registered direct offering|prices? (?:its |an? )?offering)\b',
     'dilution', -0.7),
    # --- deals ----------------------------------------------------------
    (r'\bto be acquired\b', 'acquisition_target', 0.9),
    (r'\b(?:agrees?|agreed|enters?|entered)\b.{0,30}\b(?:to be acquired|to merge|definitive (?:merger )?agreement)\b',
     'acquisition_target', 0.8),
    (r'\b(?:receives?|received|rejects?)\b.{0,40}\b(?:takeover|buyout|acquisition|unsolicited)\b.{0,20}'
     r'\b(?:offer|proposal|bid)\b', 'acquisition_target', 0.6),
    (r'\b(?:awarded|wins?|won|secures?|secured|lands?|receives?)\b.{0,50}\b(?:contract|order|award|deal)\b',
     'contract_win', 0.4),
    (r'\$\s?\d[\d,.]*\s?(?:million|billion|m|b)\b.{0,40}\b(?:contract|order|deal|agreement)\b', 'contract_win', 0.2),
    (r'\b(?:added to|to join|joins|inclusion in|included in)\b.{0,30}\b(?:s&p ?500|nasdaq[- ]?100|russell ?\d+|'
     r'dow jones)\b', 'index_inclusion', 0.5),
    (r'\b(?:partnership|collaboration|strategic alliance|teams? up)\b.{0,40}\bwith\b', 'partnership', 0.2),
    # --- governance / legal / distress ----------------------------------
    (r'\b(?:sec|doj|ftc|regulators?)\b.{0,30}\b(?:investigat(?:es|ion|ing)|probe|subpoena|charges?|sues?|lawsuit)\b',
     'legal', -0.6),
    (r'\b(?:class action|securities fraud|indict(?:ed|ment)|accounting (?:irregularities|errors)|restat(?:es|ement))\b',
     'legal', -0.6),
    (r'\b(?:ceo|cfo|chief executive|chief financial)\b.{0,30}\b(?:resigns?|steps? down|departs?|fired|ousted|'
     r'terminated)\b', 'executive_departure', -0.4),
    (r'\b(?:chapter 11|chapter 7|bankruptcy|going concern|insolven(?:t|cy)|default(?:s|ed)? on)\b', 'distress', -1.0),
    (r'\b(?:data breach|cyber ?attack|ransomware)\b', 'incident', -0.4),
    (r'\bshort(?:-| )seller\b.{0,20}\breport\b', 'short_report', -0.5),
]


class RuleClassifier(Classifier):
    def __init__(self, rules: Optional[Iterable[Rule]] = None):
        self._rules = [(re.compile(p, re.IGNORECASE | re.DOTALL), cat, w)
                       for p, cat, w in (rules if rules is not None else DEFAULT_RULES)]

    def classify(self, item: NewsItem) -> Classification:
        return self.classify_text(item.text)

    def classify_text(self, text: str) -> Classification:
        text = ' '.join(text.split())
        matched = []
        for rx, cat, weight in self._rules:
            if rx.search(text):
                matched.append((cat, weight))
        if not matched:
            return Classification(score=0.0, category='none', confidence=0.0)
        score = max(-1.0, min(1.0, sum(w for _, w in matched)))
        # Category: the strongest single rule on the winning side.
        side = 1 if score > 0 else -1
        same_side = [(c, w) for c, w in matched if (w > 0) == (side > 0)] or matched
        category = max(same_side, key=lambda cw: abs(cw[1]))[0]
        confidence = min(1.0, 0.4 + 0.2 * len(matched))
        reasons = [f'{c}({w:+.1f})' for c, w in matched]
        return Classification(score=round(score, 3), category=category, confidence=confidence, reasons=reasons)
