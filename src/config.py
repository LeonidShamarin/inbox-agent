"""
Конфігурація і — головне — **бюджети**.

Агент без бюджету не «іноді зациклюється», а зациклюється рано чи пізно
обов'язково: достатньо, щоб інструмент повертав щось, що модель вважає
недостатнім, і вона проситиме його знову. Тому обмежень тут три, і вони
незалежні, бо ловлять різні збої:

* `max_steps` — модель ходить по колу («пошукай ще раз»);
* `max_tokens` — кроків небагато, але контекст росте і кожен коштує дорожче;
* `timeout_s` — жоден із перших двох не спрацював, бо один виклик просто висить.

Перевищення будь-якого — це не виняток, а нормальний кінець сесії зі статусом
`aborted` і причиною в трасі. Мовчазне продовження було б інцидентом.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

# Оркестрація за готовим набором інструментів — не задача на глибоке міркування.
# flash-lite вистачає, а денна квота безкоштовного тіру в неї щедріша за flash.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Ліміт безкоштовного тіру — 5 запитів/хв на модель.
DEFAULT_RPM = 5

# Ціни flash-lite за мільйон токенів, USD — лише для оцінки вартості у звіті.
DEFAULT_INPUT_PRICE = 0.10
DEFAULT_OUTPUT_PRICE = 0.40


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    rpm: int = DEFAULT_RPM

    # --- бюджети однієї задачі ---
    # 6 кроків — це чотири виклики інструментів плюс план і відповідь. Жоден
    # сценарій із eval-набору не потребує більше; якщо агент упирається в цю
    # стелю, він майже напевно ходить по колу.
    max_steps: int = 6
    max_tokens: int = 20_000
    timeout_s: float = 60.0
    # Окремо від max_steps: модель може викликати той самий інструмент
    # знову і знову в межах дозволених кроків.
    max_same_tool_calls: int = 2

    # --- поведінка ---
    # Вимикач human-in-the-loop. Потрібен рівно для одного рядка в eval:
    # «скільки незворотних дій виконалося б без людини».
    require_confirmation: bool = True

    input_price_per_mtok: float = DEFAULT_INPUT_PRICE
    output_price_per_mtok: float = DEFAULT_OUTPUT_PRICE

    def to_dict(self) -> dict:
        return asdict(self)

    def short_name(self) -> str:
        parts = [self.model.split("-")[-1], f"steps{self.max_steps}"]
        if not self.require_confirmation:
            parts.append("noconfirm")
        return "+".join(parts)


def config_from_env() -> AgentConfig:
    """Конфіг для сервісу: у Docker прапорців CLI немає, є оточення."""
    return AgentConfig(
        model=os.environ.get("AGENT_MODEL", DEFAULT_MODEL),
        max_steps=int(os.environ.get("AGENT_MAX_STEPS", "6")),
        max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "20000")),
        timeout_s=float(os.environ.get("AGENT_TIMEOUT_S", "60")),
        rpm=int(os.environ.get("AGENT_RPM", str(DEFAULT_RPM))),
    )
