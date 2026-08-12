"""
Пошук по внутрішній базі знань — реалізація інструмента `search_knowledge_base`
**всередині цього проекту**, без звернення до сусіднього сервісу.

Чому лексичний пошук, а не embeddings. Інструмент агента має відповідати за
частки секунди й не тягнути за собою півтора гігабайта ваг: агент викликає його
всередині циклу, під таймаутом усієї задачі. На корпусі з десятка регламентів
BM25 з нормалізацією дає те, що потрібно, і залишається повністю
детермінованим — а отже, тестованим без мережі.

Токенізація і стемер перенесені з мого ж kb-rag-assistant, де ефект стемера
**заміряний**, а не постульований: на підмножині питань зі словоформами він дає
+18.9 п.п. MRR. Українська сильно флективна, і без нормалізації питання «скільки
днів відпустки» не перетинається з документом, де написано «24 календарні дні»,
жодним токеном.

Стемер навмисно примітивний — відкидання найдовшого відомого закінчення. Це не
морфологічний аналіз: `погодження` → `погодж`, а `погодити` → `погод`, і вони не
злипаються. Повноцінна альтернатива — pymorphy/stanza, але це +200 МБ і на
порядок повільніше, чого інструмент у циклі агента собі дозволити не може.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-zа-яґєії0-9]+")
_APOSTROPHE = str.maketrans({"'": "", "’": "", "ʼ": "", "`": ""})

_SUFFIXES = [
    "ування", "овану", "ованого", "ованим",
    "ання", "ення", "іння", "ості", "істю", "ість", "ства",
    "ями", "ами", "ові", "еві", "єві", "ому", "ему", "ими", "іми", "ого",
    "ють", "ать", "ять", "ати", "ити", "ути", "имо", "ємо", "ете",
    "их", "ий", "ій", "ою", "ею", "ії", "ям", "ах", "ях", "ам", "ом", "ем",
    "ів", "ей", "ої", "им", "ім", "ти", "ла", "ло", "ли",
    "а", "е", "и", "і", "о", "у", "ю", "я", "й", "ь",
]
_SUFFIXES.sort(key=len, reverse=True)

MIN_STEM = 4

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().translate(_APOSTROPHE))


def stem(token: str) -> str:
    if token.isdigit():
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM:
            return token[: -len(suffix)]
    return token


def normalize(text: str, stemming: bool = True) -> list[str]:
    tokens = tokenize(text)
    return [stem(t) for t in tokens] if stemming else tokens


class BM25Index:
    """
    BM25 (Okapi) власною реалізацією — алгоритм на сорок рядків, а бібліотека
    тягне свою токенізацію, яка для української не годиться.
    """

    def __init__(self, texts: list[str], stemming: bool = True):
        self.stemming = stemming
        self.docs: list[Counter[str]] = []
        self.lengths: list[int] = []
        df: Counter[str] = Counter()

        for text in texts:
            counts = Counter(normalize(text, stemming))
            self.docs.append(counts)
            self.lengths.append(sum(counts.values()))
            df.update(counts.keys())

        self.n = len(texts)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        # IDF зі згладжуванням: не дає від'ємних ваг токенам, які є більш ніж
        # у половині документів (у корпусі з десяти регламентів це половина слів).
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        terms = normalize(query, self.stemming)
        if not terms or self.n == 0:
            return []

        scores = [0.0] * self.n
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.docs):
                tf = counts.get(term)
                if not tf:
                    continue
                norm = 1 - B + B * (self.lengths[i] / self.avgdl if self.avgdl else 0.0)
                scores[i] += idf * (tf * (K1 + 1)) / (tf + K1 * norm)

        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0.0), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]


# Розширення запиту синонімами.
#
# Головна вада лексичного пошуку: він шукає слова, а не зміст. Запит «зламали
# мій акаунт» не перетинається з регламентом, де написано «компрометація
# облікового запису», ЖОДНИМ токеном — і BM25 чесно повертає порожньо.
# Embeddings закрили б цей клас повністю, але ціна — півтора гігабайта ваг
# усередині циклу агента, який працює під таймаутом.
#
# Тому — словник на десяток пар для найчастіших розбіжностей «як пише людина»
# проти «як написано в регламенті». Це латка, а не розв'язання: слово, якого
# тут немає, так само не знайдеться. Межа підходу описана в README.
QUERY_SYNONYMS: dict[str, list[str]] = {
    "акаунт": ["обліковий запис"],
    "аккаунт": ["обліковий запис"],
    "зламали": ["компрометація скомпрометований інцидент безпеки"],
    "зламав": ["компрометація"],
    "хакнули": ["компрометація інцидент безпеки"],
    "вкрали": ["компрометація"],
    "ноут": ["ноутбук обладнання"],
    "лаптоп": ["ноутбук обладнання"],
    "звільняється": ["звільнення"],
    "новачок": ["новий співробітник"],
    "мітинг": ["зустріч"],
    "созвон": ["зустріч"],
    "чек": ["фіскальний чек відшкодування"],
    "рахунок": ["оплата закупівля"],
    "хворію": ["лікарняний"],
    "захворів": ["лікарняний"],
    "відгул": ["робота у вихідний"],
}


def expand_query(query: str) -> str:
    """Додає до запиту синоніми знайдених слів. Не замінює — саме додає."""
    lowered = query.lower()
    extras = [syn for word, syns in QUERY_SYNONYMS.items() if word in lowered for syn in syns]
    return f"{query} {' '.join(extras)}" if extras else query


@dataclass
class Chunk:
    doc_title: str
    section: str
    text: str
    source: str

    def citation(self) -> str:
        return f"{self.doc_title} → {self.section}"


def split_markdown(text: str, doc_path: Path) -> list[Chunk]:
    """
    Структурне чанкування: межа — заголовок, а не N символів.

    Регламент так і написаний: розділ «Ліміти погодження» — це закінчена думка.
    Різати його посередині означає віддати агентові півтаблиці, і він відповість
    упевнено й неправильно.
    """
    lines = text.splitlines()
    doc_title = doc_path.stem
    section = "Загальне"
    buffer: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(
                Chunk(doc_title=doc_title, section=section, text=body, source=str(doc_path))
            )

    for line in lines:
        if line.startswith("# "):
            flush()
            buffer = []
            doc_title = line[2:].strip()
            section = "Загальне"
        elif line.startswith(("## ", "### ")):
            flush()
            buffer = []
            section = line.lstrip("#").strip()
        else:
            buffer.append(line)
    flush()
    return chunks


# Поріг релевантності.
#
# BM25 майже завжди повертає «щось»: у запиті «хто президент Марса» слово «хто»
# є в половині регламентів, і формально це влучання з ненульовим балом. Агент,
# який отримує такий результат, чесно вважає, що відповідь знайдена, і будує на
# ній лист.
#
# Значення НЕ вгадане — воно з кривої (`main.py eval --retrieval-only`,
# розділ «Поріг релевантності»):
#
#   поріг 0–1  → hit@3 100%, але поза базою мовчить лише 28.6%
#   поріг 2    → hit@3 100%, мовчить 71.4%
#   поріг 3    → hit@3  92%, мовчить 100%   ← коліно, робоче значення
#   поріг 4    → hit@3  84%, мовчить 100%   ← мінус 8 п.п. повноти задарма
#
# Перша спроба була 4.0 — за вимірюванням на семи питаннях. На повному наборі
# з 25 виявилося, що вона коштує 16% правильних відповідей і нічого не додає.
#
# Запас тонкий і це варто знати: найвищий бал серед питань поза базою —
# 2.995 («як приготувати борщ»), тобто до порогу лишається 0.005. На більшому
# корпусі поріг доведеться переміряти, а не успадкувати.
MIN_SCORE = 3.0


class KnowledgeBase:
    """Корпус markdown-регламентів у пам'яті. Будується за мілісекунди."""

    def __init__(self, chunks: list[Chunk], stemming: bool = True):
        self.chunks = chunks
        self.stemming = stemming
        # Заголовок додається до тіла чанка навмисно: назва розділу — найточніші
        # ключові слова, які в тексті часто не повторюються.
        self.index = BM25Index(
            [f"{c.doc_title} {c.section}\n{c.text}" for c in chunks], stemming=stemming
        )

    @classmethod
    def from_dir(cls, folder: Path, stemming: bool = True) -> "KnowledgeBase":
        chunks: list[Chunk] = []
        for path in sorted(folder.glob("*.md")):
            chunks.extend(split_markdown(path.read_text(encoding="utf-8"), path))
        return cls(chunks, stemming=stemming)

    def search(self, query: str, top_k: int = 3, min_score: float = MIN_SCORE) -> list[dict]:
        """
        Порожній список — це відповідь «у базі цього немає», а не збій.
        Саме він має підштовхнути агента до ескалації замість вигадування.
        """
        hits = self.index.search(expand_query(query), top_k)
        return [
            {
                "citation": self.chunks[i].citation(),
                "text": self.chunks[i].text[:700],
                "score": round(score, 3),
            }
            for i, score in hits
            if score >= min_score
        ]
