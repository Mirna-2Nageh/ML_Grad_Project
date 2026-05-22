"""
Arabic legal text preprocessing and document classification.
"""
import re
from typing import List
import config


_ARABIC_INDIC_DIGITS = str.maketrans(
    # Standard Arabic-Indic (Egypt/Saudi/Gulf) U+0660-U+0669 + Eastern Arabic-Indic /
    # Persian U+06F0-U+06F9. Egyptian legal text mixes both freely; missing the eastern
    # range was breaking article-citation validation against real cassation rulings.
    "٠١٢٣٤٥٦٧٨٩" + "۰۱۲۳۴۵۶۷۸۹",
    "0123456789" + "0123456789",
)


def normalize_arabic_indic_digits(text: str) -> str:
    """Map Arabic-Indic and Eastern Arabic-Indic digits -> Western 0-9."""
    return text.translate(_ARABIC_INDIC_DIGITS)


def clean_arabic_legal_text(text: str) -> str:
    """Comprehensive preprocessing for Arabic legal documents."""
    # 1. Remove diacritics (tashkeel)
    text = re.sub(r'[ً-ٰٟ]', '', text)

    # 1b. Unify digit forms - Egyptian legal text mixes Arabic-Indic and Western digits freely.
    text = normalize_arabic_indic_digits(text)

    # 2. Normalize Arabic characters
    text = re.sub(r'[أإآ]', 'ا', text)
    if config.NORMALIZE_TA_MARBUTA:
        text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)

    # 3. Remove common legal document boilerplate
    text = re.sub(r'بسم الله الرحمن الرحيم', '', text)
    text = re.sub(r'باسم الشعب', '', text)
    text = re.sub(r'محكمة\s+\S+\s+الابتدائية', '', text)

    # 4. Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)

    # 5. Remove page numbers and artifacts
    text = re.sub(r'- \d+ -', '', text)
    text = re.sub(r'\(\s*\d+\s*\)', '', text)

    # 6. Normalize Arabic punctuation spacing
    text = text.replace('؛', '؛ ')
    text = text.replace('،', '، ')

    return text.strip()


def preprocess_arabic_for_bm25(text: str) -> List[str]:
    """Normalize and tokenize Arabic text for BM25 indexing/search."""
    text = re.sub(r'[ً-ٰٟ]', '', text)
    text = normalize_arabic_indic_digits(text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'[ى]', 'ي', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def get_document_type(source_path: str) -> str:
    """Classify document by its directory location."""
    if 'جنايات' in source_path:
        return 'criminal_case'
    elif 'محكمه النقض' in source_path:
        return 'cassation_ruling'
    elif 'موسوعة' in source_path:
        return 'cassation_encyclopedia'
    elif 'قانون العقوبات' in source_path or 'عقوبات' in source_path:
        return 'penal_code'
    elif 'إجراءات' in source_path or 'اجراءات' in source_path or 'الاجرءات' in source_path:
        return 'criminal_procedure'
    elif 'الطب الشرعي' in source_path:
        return 'forensic_medicine'
    elif 'القواعد' in source_path or 'مجموعة' in source_path:
        return 'legal_rules_collection'
    elif 'قانون الجنائي' in source_path or 'الجنائي' in source_path:
        return 'criminal_law_reference'
    else:
        return 'legal_reference'


def get_legal_category(source_path: str) -> str:
    """Extract the legal topic category from the path (for encyclopedia subdirs)."""
    parts = source_path.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if 'موسوعة' in part and i + 1 < len(parts):
            next_part = parts[i + 1]
            if not next_part.endswith('.txt') and not next_part.endswith('.pdf'):
                return next_part
    return 'general'


def get_legal_topic(source_path: str) -> str:
    """Encyclopedia subdir name as the topic - empty for non-encyclopedia paths.

    The encyclopedia is the only part of the dataset where folder taxonomy is hand-curated
    by legal topic (تزوير, قتل عمد, ...). Surface as first-class metadata for
    retrieval filtering and the confidence-score's topic-match factor (graph-lite signal).
    """
    parts = source_path.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if 'موسوعة' in part and i + 1 < len(parts):
            next_part = parts[i + 1]
            if not (next_part.endswith('.txt') or next_part.endswith('.pdf')):
                return next_part
    return ''


# Matches an article keyword (singular/plural/dual) + optional colon, then the
# first number AND any following comma/'و'-separated numbers in the same list.
# Handles: "المادة 316", "المواد: 315", "المواد 211، 212، 213", "المادتين 230 و232".
_ARTICLE_BLOCK_PATTERN = re.compile(
    r'(?:المادة|المادتين|المادتان|المواد|مادة|مادتين|مواد)\s*:?\s*'
    r'(\d+(?:\s*[،,و]\s*\d+)*)'
)
_NUM_PATTERN = re.compile(r'\d+')


def extract_article_references(text: str) -> List[str]:
    """Return deduplicated Egyptian legal article numbers cited in `text`, as Western-digit strings.

    Recognizes singular/plural/dual article keywords and number lists, so
    "المواد 211، 212، 213" yields ['211','212','213'] (not just the first)."""
    normalized = normalize_arabic_indic_digits(text)
    matches: List[str] = []
    for block in _ARTICLE_BLOCK_PATTERN.findall(normalized):
        matches.extend(_NUM_PATTERN.findall(block))
    unique = list(set(matches))
    try:
        unique.sort(key=int)
    except ValueError:
        unique.sort()
    return unique
