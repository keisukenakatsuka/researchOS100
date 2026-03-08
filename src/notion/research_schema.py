# src/notion/research_schema.py
"""Schema definitions for Knowledge Memory Layer Notion databases.

Covers five databases used by the Deep Research pipeline (072):
- Sources
- Evidence
- Claims
- Memos
- Research Runs

Each DB has its own section of PROP_* constants and an
EXPECTED_PROPERTIES dict for optional schema validation.
"""

from __future__ import annotations

from typing import Dict

# ================================================================
# Sources DB
# ================================================================

SRC_PROP_TITLE = "Name"                       # title
SRC_PROP_URL = "URL"                          # url
SRC_PROP_SOURCE_ID = "Source ID"              # rich_text (local ID)
SRC_PROP_SOURCE_TYPE = "Source Type"          # select
SRC_PROP_DOMAIN = "Domain"                   # rich_text
SRC_PROP_FETCH_STATUS = "Fetch Status"       # select
SRC_PROP_FETCHED_CHARS = "Fetched Chars"     # number
SRC_PROP_RETRIEVED_AT = "Retrieved At"       # date

SRC_EXPECTED: Dict[str, str] = {
    SRC_PROP_TITLE: "title",
    SRC_PROP_URL: "url",
    SRC_PROP_SOURCE_ID: "rich_text",
    SRC_PROP_SOURCE_TYPE: "select",
    SRC_PROP_DOMAIN: "rich_text",
    SRC_PROP_FETCH_STATUS: "select",
    SRC_PROP_FETCHED_CHARS: "number",
    SRC_PROP_RETRIEVED_AT: "date",
}

# ================================================================
# Evidence DB
# ================================================================

EV_PROP_TITLE = "Name"                        # title (= statement prefix)
EV_PROP_EVIDENCE_ID = "Evidence ID"           # rich_text (local ID)
EV_PROP_STATEMENT = "Statement"               # rich_text
EV_PROP_CONFIDENCE = "Confidence"             # select
EV_PROP_CONFIDENCE_REASON = "Confidence Reason"  # rich_text
EV_PROP_TAGS = "Tags"                         # multi_select
EV_PROP_SOURCE = "Source"                     # relation -> Sources DB
EV_PROP_EXTRACTED_AT = "Extracted At"         # date

EV_EXPECTED: Dict[str, str] = {
    EV_PROP_TITLE: "title",
    EV_PROP_EVIDENCE_ID: "rich_text",
    EV_PROP_STATEMENT: "rich_text",
    EV_PROP_CONFIDENCE: "select",
    EV_PROP_CONFIDENCE_REASON: "rich_text",
    EV_PROP_TAGS: "multi_select",
    EV_PROP_SOURCE: "relation",
    EV_PROP_EXTRACTED_AT: "date",
}

# ================================================================
# Claims DB
# ================================================================

CL_PROP_TITLE = "Name"                        # title (= statement prefix)
CL_PROP_CLAIM_ID = "Claim ID"                # rich_text (local ID)
CL_PROP_STATEMENT = "Statement"               # rich_text
CL_PROP_CONFIDENCE = "Confidence"             # select
CL_PROP_CONFIDENCE_REASON = "Confidence Reason"  # rich_text
CL_PROP_TAGS = "Tags"                         # multi_select
CL_PROP_EVIDENCE = "Evidence"                 # relation -> Evidence DB
CL_PROP_SOURCES = "Sources"                   # relation -> Sources DB
CL_PROP_CREATED_AT = "Created At"             # date

CL_EXPECTED: Dict[str, str] = {
    CL_PROP_TITLE: "title",
    CL_PROP_CLAIM_ID: "rich_text",
    CL_PROP_STATEMENT: "rich_text",
    CL_PROP_CONFIDENCE: "select",
    CL_PROP_CONFIDENCE_REASON: "rich_text",
    CL_PROP_TAGS: "multi_select",
    CL_PROP_EVIDENCE: "relation",
    CL_PROP_SOURCES: "relation",
    CL_PROP_CREATED_AT: "date",
}

# ================================================================
# Memos DB
# ================================================================

MEMO_PROP_TITLE = "Name"                      # title
MEMO_PROP_MEMO_ID = "Memo ID"                # rich_text (local ID)
MEMO_PROP_SUMMARY = "Summary"                 # rich_text
MEMO_PROP_TYPE = "Type"                       # select
MEMO_PROP_CLAIMS = "Claims"                   # relation -> Claims DB
MEMO_PROP_EVIDENCE = "Evidence"               # relation -> Evidence DB
MEMO_PROP_SOURCES = "Sources"                 # relation -> Sources DB
MEMO_PROP_RESEARCH_RUN = "Research Run"       # relation -> Research Runs DB
MEMO_PROP_CREATED_AT = "Created At"           # date

MEMO_EXPECTED: Dict[str, str] = {
    MEMO_PROP_TITLE: "title",
    MEMO_PROP_MEMO_ID: "rich_text",
    MEMO_PROP_SUMMARY: "rich_text",
    MEMO_PROP_TYPE: "select",
    MEMO_PROP_CLAIMS: "relation",
    MEMO_PROP_EVIDENCE: "relation",
    MEMO_PROP_SOURCES: "relation",
    MEMO_PROP_RESEARCH_RUN: "relation",
    MEMO_PROP_CREATED_AT: "date",
}

# ================================================================
# Research Runs DB
# ================================================================

RR_PROP_TITLE = "Name"                        # title
RR_PROP_RUN_ID = "Run ID"                    # rich_text (local ID)
RR_PROP_REQUEST = "Research Request"          # rich_text
RR_PROP_RUN_TYPE = "Run Type"                # select
RR_PROP_STATUS = "Status"                    # select
RR_PROP_STARTED_AT = "Started At"            # date
RR_PROP_COMPLETED_AT = "Completed At"        # date
RR_PROP_SOURCES = "Sources"                   # relation -> Sources DB
RR_PROP_EVIDENCE = "Evidence"                 # relation -> Evidence DB
RR_PROP_CLAIMS = "Claims"                     # relation -> Claims DB
RR_PROP_MEMOS = "Memos"                       # relation -> Memos DB

RR_EXPECTED: Dict[str, str] = {
    RR_PROP_TITLE: "title",
    RR_PROP_RUN_ID: "rich_text",
    RR_PROP_REQUEST: "rich_text",
    RR_PROP_RUN_TYPE: "select",
    RR_PROP_STATUS: "select",
    RR_PROP_STARTED_AT: "date",
    RR_PROP_COMPLETED_AT: "date",
    RR_PROP_SOURCES: "relation",
    RR_PROP_EVIDENCE: "relation",
    RR_PROP_CLAIMS: "relation",
    RR_PROP_MEMOS: "relation",
}

# ================================================================
# Env var names for DB IDs
# ================================================================

ENV_SOURCES_DB_ID = "NOTION_RESEARCH_SOURCES_DB_ID"
ENV_EVIDENCE_DB_ID = "NOTION_RESEARCH_EVIDENCE_DB_ID"
ENV_CLAIMS_DB_ID = "NOTION_RESEARCH_CLAIMS_DB_ID"
ENV_MEMOS_DB_ID = "NOTION_RESEARCH_MEMOS_DB_ID"
ENV_RESEARCH_RUNS_DB_ID = "NOTION_RESEARCH_RUNS_DB_ID"
