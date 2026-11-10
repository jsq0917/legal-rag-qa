"""法条时效字段：从正文提取施行日期，并与 Milvus 的整型日期互转。

约定
----
* MySQL ``DATE`` 用 ``YYYY-MM-DD`` 字符串，未知为 ``NULL``；
* Milvus 没有可空标量，用 ``INT64`` 存 ``YYYYMMDD``，``0`` 表示未知/长期有效；
* 只从正文里读**明确的**"自 XXXX 年 X 月 X 日起施行"，读不到就留空，
  不按常识补日期（例如劳动合同法的修正日期不从推断得到）。
"""

from __future__ import annotations

import re
import unicodedata

_DATE_RE = re.compile(r"自\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*起\s*施\s*行")


def extract_effective_from(text: str) -> str:
    """返回 ``YYYY-MM-DD``；没有明确施行日期返回空串。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    matches = _DATE_RE.findall(normalized)
    if not matches:
        return ""
    year, month, day = matches[-1]
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def normalize_date(value: str | None) -> str:
    """宽松归一日期字符串；空值/非法值返回空串。"""
    value = (value or "").strip()
    if not value:
        return ""
    match = re.fullmatch(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", value)
    if not match:
        return ""
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def date_to_int(value: str | None) -> int:
    """``YYYY-MM-DD`` → ``YYYYMMDD``；空/非法 → 0。"""
    normalized = normalize_date(value)
    return int(normalized.replace("-", "")) if normalized else 0


def int_to_date(value: int | None) -> str:
    """``YYYYMMDD`` → ``YYYY-MM-DD``；0/非法 → 空串。"""
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    text = str(number)
    if len(text) != 8:
        return ""
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"
