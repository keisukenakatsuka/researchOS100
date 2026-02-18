# src/notion/properties.py
"""Notion property value extraction utilities.

Pure-parsing helpers that convert a raw Notion page property object into
a plain Python value.  No API calls, no side effects — safe to use
anywhere.

Extracted from notebook 040_weekly_papers_review Cell 05 and promoted to
a shared module so that 047+ scripts can reuse the same logic.

Usage::

    from src.notion.properties import extract_property_value

    name = extract_property_value(page, "Name")          # str
    score = extract_property_value(page, "Importance")    # float | None
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def extract_property_value(page: Mapping[str, Any], prop_name: str) -> Any:
    """Extract a single property value from a Notion page object.

    Supports all common Notion property types and returns a plain Python
    value (``str``, ``float``, ``None``, …).  Unknown types return
    ``None``.

    Parameters
    ----------
    page:
        A raw Notion page dict (as returned by the API).
    prop_name:
        The property name to extract (must match the DB column name exactly).

    Returns
    -------
    str | float | None
        The extracted value.
    """
    properties = page.get("properties", {})
    if prop_name not in properties:
        return None

    prop_data = properties[prop_name]
    prop_type = prop_data.get("type")

    if prop_type == "title":
        titles = prop_data.get("title", [])
        return titles[0].get("plain_text", "") if titles else ""

    if prop_type == "rich_text":
        texts = prop_data.get("rich_text", [])
        return texts[0].get("plain_text", "") if texts else ""

    if prop_type == "url":
        return prop_data.get("url", "")

    if prop_type == "date":
        date_obj = prop_data.get("date")
        if date_obj:
            return date_obj.get("start", "")
        return None

    if prop_type == "number":
        return prop_data.get("number")

    if prop_type == "select":
        select_obj = prop_data.get("select")
        return select_obj.get("name", "") if select_obj else ""

    if prop_type == "multi_select":
        multi = prop_data.get("multi_select", [])
        return ", ".join(item.get("name", "") for item in multi)

    if prop_type == "created_time":
        return prop_data.get("created_time", "")

    if prop_type == "relation":
        relations = prop_data.get("relation", [])
        return ", ".join(rel.get("id", "") for rel in relations)

    if prop_type == "checkbox":
        return prop_data.get("checkbox", False)

    if prop_type == "email":
        return prop_data.get("email", "")

    if prop_type == "phone_number":
        return prop_data.get("phone_number", "")

    if prop_type == "formula":
        formula = prop_data.get("formula", {})
        f_type = formula.get("type")
        return formula.get(f_type) if f_type else None

    # Unknown type
    return None


def page_to_record(
    page: Mapping[str, Any],
    property_names: List[str],
) -> Dict[str, Any]:
    """Convert a Notion page into a flat dict of extracted values.

    Always includes ``notion_page_id`` and ``notion_url`` alongside the
    requested properties.

    Parameters
    ----------
    page:
        Raw Notion page dict.
    property_names:
        List of property names to extract.

    Returns
    -------
    dict
        ``{"notion_page_id": ..., "notion_url": ..., <prop>: ..., ...}``
    """
    record: Dict[str, Any] = {
        "notion_page_id": page.get("id", ""),
        "notion_url": page.get("url", ""),
    }
    for name in property_names:
        record[name] = extract_property_value(page, name)
    return record
