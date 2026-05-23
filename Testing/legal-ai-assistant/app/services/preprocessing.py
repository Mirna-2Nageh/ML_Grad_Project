"""
Arabic legal text preprocessing and document classification.
"""
import re
from typing import List, Dict
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


# ─────────────────────────────────────────────────────────────────────────────
# Query domain classification + synonym expansion
#
# WHY: The eval(3) data showed retrieval missed procedural-law chunks for
# questions like "شروط التوقيف الاحتياطي" because user phrasing didn't lexically
# match the chunks (which say "الحبس الاحتياطي"). Domain detection + a small
# Arabic synonym table fixes the most common variants without an LLM call.
# ─────────────────────────────────────────────────────────────────────────────

# Procedural keywords — these belong in قانون الإجراءات الجنائية.
_PROCEDURAL_TERMS = {
    'التوقيف', 'الحبس الاحتياطي', 'الحبس', 'التحقيق', 'التفتيش',
    'الضبط', 'محضر', 'محاضر', 'النيابة', 'النيابة العامة',
    'قاضي التحقيق', 'الاستئناف', 'الطعن', 'النقض', 'حكم النقض',
    'الإجراءات', 'الاجراءات', 'إجراءات', 'اجراءات',
    'الاستجواب', 'القبض', 'إعادة المحاكمة', 'المعارضة',
    'الإدعاء', 'الادعاء', 'المحاكمة',
}

# Substantive keywords — these belong in قانون العقوبات.
_SUBSTANTIVE_TERMS = {
    'العقوبة', 'عقوبة', 'الجريمة', 'جريمة', 'الأركان', 'أركان',
    'تعريف', 'القصد الجنائي', 'الركن المادي', 'الركن المعنوي',
    'الشروع', 'القتل العمد', 'القتل', 'السرقة', 'التزوير', 'الرشوة',
    'الاختلاس', 'الإباحة', 'موانع المسؤولية', 'الدفاع الشرعي',
    'الجنحة', 'الجناية', 'المخالفة', 'المسؤولية الجنائية',
}

# Synonym sets — each entry triggers an additional retrieval pass. The same
# concept is referenced under multiple terms in the index, so the original
# query + 1-2 synonyms substantially widens recall without ballooning latency.
_QUERY_SYNONYMS: Dict[str, List[str]] = {
    'التوقيف الاحتياطي':    ['الحبس الاحتياطي', 'احتجاز المتهم على ذمة التحقيق'],
    'الحبس الاحتياطي':      ['التوقيف الاحتياطي', 'احتجاز المتهم على ذمة التحقيق'],
    'الدفاع الشرعي':        ['حق الدفاع عن النفس', 'الدفاع عن النفس والمال', 'دفع الصائل'],
    'القتل العمد':          ['قتل عمد مع سبق الإصرار', 'إزهاق الروح عمداً'],
    'القتل الخطأ':          ['القتل غير العمد', 'القتل بإهمال'],
    'الجريمة التامة':       ['اكتمال الجريمة', 'الجريمة المكتملة'],
    'الشروع':               ['البدء في التنفيذ', 'الشروع في الجريمة', 'الشروع غير الموقوف'],
    'موانع المسؤولية':      ['أسباب امتناع المسؤولية', 'عدم المسؤولية الجنائية', 'الإكراه والاضطرار'],
    'أسباب الإباحة':        ['أسباب الإباحة العامة', 'إباحة الفعل', 'انتفاء الجريمة'],
    'الفاعل الأصلي':        ['المساهمة الجنائية', 'الفاعل والشريك', 'فاعل الجريمة'],
    'الشريك':               ['الاشتراك في الجريمة', 'المساهمة التبعية', 'شريك بالاتفاق'],
    'القصد الجنائي':        ['القصد العام', 'القصد الخاص', 'الركن المعنوي'],
    'الركن المادي':         ['السلوك الإجرامي', 'الفعل المادي للجريمة'],
    'الركن المعنوي':        ['القصد الجنائي', 'الإرادة الإجرامية'],
    'التزوير':              ['تزوير المحررات', 'اصطناع محرر', 'تزوير المحررات الرسمية'],
    'السرقة':               ['الاختلاس', 'سرقة المنقولات', 'أخذ مال الغير'],
    'السرقة بالإكراه':      ['سرقة باستخدام القوة', 'الإكراه في السرقة'],
    'الجهل بالقانون':       ['عدم العلم بالقانون', 'الغلط في القانون'],
    'علاقة السببية':        ['رابطة السببية', 'الرابطة السببية', 'إسناد النتيجة'],
    'النية الإجرامية':      ['القصد الجنائي', 'الإرادة الإجرامية'],
}


def classify_query_domain(query: str) -> str:
    """Return 'procedural', 'substantive', or 'unknown' based on Arabic keywords.

    Used by the retrieval layer to boost chunks whose doc_type matches the
    inferred domain. Conservative on purpose — a missed classification just
    falls back to undifferentiated hybrid retrieval (current behavior).
    """
    if not query:
        return 'unknown'
    q = query
    proc_hits = sum(1 for t in _PROCEDURAL_TERMS if t in q)
    subst_hits = sum(1 for t in _SUBSTANTIVE_TERMS if t in q)
    # Require a clear majority; ties stay 'unknown' so we don't push the wrong way.
    if proc_hits > subst_hits and proc_hits > 0:
        return 'procedural'
    if subst_hits > proc_hits and subst_hits > 0:
        return 'substantive'
    return 'unknown'


def expand_query_synonyms(query: str, max_extra: int = 2) -> List[str]:
    """Return up to `max_extra` synonym variants of `query` for multi-pass retrieval.

    Replaces the first matched key in the query with each of its synonyms.
    Deterministic, no LLM call, ~microseconds. Returns [] when no key matches —
    in which case the caller should just use the original query.
    """
    if not query:
        return []
    variants: List[str] = []
    seen: set = {query.strip()}
    for key, syns in _QUERY_SYNONYMS.items():
        if key in query:
            for syn in syns:
                v = query.replace(key, syn).strip()
                if v and v not in seen:
                    variants.append(v)
                    seen.add(v)
                if len(variants) >= max_extra:
                    return variants
    return variants


# Map a query domain to the chunk doc_types that should be boosted for that domain.
DOMAIN_TO_DOC_TYPES: Dict[str, set] = {
    'procedural':  {'criminal_procedure', 'cassation_ruling'},
    'substantive': {'penal_code', 'cassation_encyclopedia', 'criminal_law_reference', 'legal_reference'},
    'unknown':     set(),
}


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
