# crism_classification/data/label_parser.py
import re
import numpy as np

CLASSES = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
N_CLASSES = len(CLASSES)

_CONFIDENCE_WEIGHTS = {'High': 1.0, 'Moderate': 0.5, 'Low': 0.25}

# Maps token strings found in category to class indices and values.
# For untyped "olivine", we assign 0.5 to both t1 and t2.
_TOKEN_MAP = {
    'type 1 olivine': {'olivine_t1': 1.0},
    'type 2 olivine': {'olivine_t2': 1.0},
    'olivine':        {'olivine_t1': 0.5, 'olivine_t2': 0.5},
    'lcp':            {'lcp': 1.0},
    'hcp':            {'hcp': 1.0},
    'plagioclase':    {'plagioclase': 1.0},
    'other':          {'other': 1.0},
    # ignored tokens (produce no label contribution)
    'alteration':     {},
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
        e.g. "Type 1 olivine (High)", "hcp + olivine (Moderate)"

    Returns
    -------
    label : np.ndarray, shape (6,), dtype float32
        Multi-hot vector for [olivine_t1, olivine_t2, lcp, hcp, plagioclase, other].
        Untyped "olivine" in mixed labels contributes 0.5 to both t1 and t2.
    weight : float
        Confidence sample weight: High=1.0, Moderate=0.5, Low=0.25.
    """
    label = np.zeros(N_CLASSES, dtype=np.float32)
    class_idx = {c: i for i, c in enumerate(CLASSES)}

    # Extract confidence tier from parentheses, e.g. "(High)"
    conf_match = re.search(r'\((\w+)\)', category)
    confidence = conf_match.group(1) if conf_match else 'Low'
    weight = _CONFIDENCE_WEIGHTS.get(confidence, 0.25)

    # Remove the confidence part and split on '+'
    mineral_part = re.sub(r'\s*\([^)]*\)', '', category).strip()
    tokens = [t.strip().lower() for t in mineral_part.split('+')]

    for token in tokens:
        # Try longest match first (so "type 1 olivine" matches before "olivine")
        for key in sorted(_TOKEN_MAP.keys(), key=len, reverse=True):
            if key in token:
                for cls, val in _TOKEN_MAP[key].items():
                    if cls in class_idx:
                        label[class_idx[cls]] = max(label[class_idx[cls]], val)
                break

    return label, weight


def get_confidence_tier(category: str) -> str:
    """Extract the confidence tier string from a category label."""
    match = re.search(r'\((\w+)\)', category)
    return match.group(1) if match else 'Low'
