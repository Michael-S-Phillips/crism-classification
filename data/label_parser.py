# crism_classification/data/label_parser.py
import re
import numpy as np

CLASSES = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other',
           'alteration']
N_CLASSES = len(CLASSES)
_CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

_CONFIDENCE_WEIGHTS = {'high': 1.0, 'moderate': 0.5, 'low': 0.25}
_CONF_RE = re.compile(r'\((\w+)\)')

# Maps token strings to class contributions.
# Keys are lowercase substrings to match against tokens.
# Longer keys must be tried first (see sorted() call in parse_category).
# Empty-dict entries explicitly mark tokens that are known but not target classes.
_TOKEN_MAP = {
    'type 1 olivine': {'olivine_t1': 1.0},
    'type 2 olivine': {'olivine_t2': 1.0},
    'olivine':        {'olivine_t1': 0.5, 'olivine_t2': 0.5},
    'lcp':            {'lcp': 1.0},
    'hcp':            {'hcp': 1.0},
    'plagioclase':    {'plagioclase': 1.0},
    'other':          {'other': 1.0},
    # Alteration minerals (clays, sulfates, opal, prehnite, chlorite, etc.).
    # Was {} (silently ignored) before 2026-06-10; now mapped to its own
    # column so the 6-class classifier can learn it as a positive output.
    'alteration':     {'alteration': 1.0},
    # Known non-target tokens — explicitly ignored so they don't fall through
    # as unrecognised. New non-target minerals should be added here.
    'red slope':      {},
    'spinel':         {},
    'pyroxene':       {},
}


def parse_category(category: str) -> tuple[np.ndarray, float]:
    """
    Parse a CRISM geopackage Category string into a multi-hot label vector
    and a confidence sample weight.

    Parameters
    ----------
    category : str
        e.g. "Type 1 olivine (High)", "hcp + olivine (Moderate)".
        Must not be None or empty — raises ValueError if so.

    Returns
    -------
    label : np.ndarray, shape (6,), dtype float32
        Multi-hot vector for [olivine_t1, olivine_t2, lcp, hcp, plagioclase, other].
        Untyped "olivine" in mixed labels contributes 0.5 to both t1 and t2.
        Returns all-zeros if no recognised mineral token is found.
    weight : float
        Confidence sample weight: High=1.0, Moderate=0.5, Low=0.25.
        Defaults to 0.25 if confidence tier is absent or unrecognised.
    """
    if category is None:
        raise ValueError("category must not be None")
    if not category.strip():
        raise ValueError(f"category must not be empty, got: {category!r}")

    label = np.zeros(N_CLASSES, dtype=np.float32)

    # Extract confidence tier — case-insensitive lookup
    conf_match = _CONF_RE.search(category)
    confidence = conf_match.group(1).lower() if conf_match else 'low'
    weight = _CONFIDENCE_WEIGHTS.get(confidence, 0.25)

    # Remove confidence parenthetical, split on '+', process each token
    mineral_part = _CONF_RE.sub('', category).strip()
    tokens = [t.strip().lower() for t in mineral_part.split('+')]

    for token in tokens:
        # Try longest key first to prevent 'olivine' matching before 'type 1 olivine'
        for key in sorted(_TOKEN_MAP.keys(), key=len, reverse=True):
            if key in token:
                for cls, val in _TOKEN_MAP[key].items():
                    label[_CLASS_IDX[cls]] = max(label[_CLASS_IDX[cls]], val)
                break  # each token matches at most one map entry

    return label, weight


def get_confidence_tier(category: str) -> str:
    """
    Extract the confidence tier string from a category label.

    Returns the original-case confidence string (e.g. 'High', 'Moderate', 'Low'),
    or 'Low' if no parenthesised term is found.
    """
    match = _CONF_RE.search(category)
    return match.group(1) if match else 'Low'
