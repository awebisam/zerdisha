"""Shared JSON parsing utilities for LLM outputs."""

from typing import Optional, Tuple, Dict, Any
import json
import re


def strip_code_fences(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def clean_json_text(text: str) -> str:
    if not text:
        return text
    # Replace smart quotes with standard quotes
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*(}\s*)", r"\1", text)
    text = re.sub(r",\s*(]\s*)", r"\1", text)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_json_substring(text: str) -> Optional[str]:
    if not text:
        return None
    start_candidates = [i for i, ch in enumerate(text) if ch in '{[']
    for start in start_candidates:
        stack = []
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch in '{[':
                    stack.append(ch)
                elif ch in '}]':
                    if not stack:
                        break
                    open_ch = stack.pop()
                    if (open_ch == '{' and ch != '}') or (open_ch == '[' and ch != ']'):
                        break
                    if not stack:
                        return text[start:i + 1]
    return None


def safe_load_json(raw_content: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if raw_content is None:
        return None, "No content"
    content = strip_code_fences(raw_content)
    content = clean_json_text(content)
    try:
        return json.loads(content), None
    except Exception as e1:
        substring = extract_json_substring(content)
        if substring:
            try:
                return json.loads(clean_json_text(substring)), None
            except Exception as e2:
                naive = re.sub(r"'([^'\\]*)'", r'"\1"', substring)
                try:
                    return json.loads(clean_json_text(naive)), None
                except Exception as e3:
                    snippet = (
                        raw_content[:200] + '...') if len(raw_content) > 200 else raw_content
                    return None, f"JSON parse failed: {e1} | substring failed: {e2} | naive failed: {e3} | content snippet: {snippet}"
        else:
            snippet = (
                raw_content[:200] + '...') if len(raw_content) > 200 else raw_content
            return None, f"JSON parse failed: {e1} | no balanced substring | content snippet: {snippet}"
