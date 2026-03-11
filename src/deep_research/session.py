"""073 User Research Session — service logic.

Orchestrates the Deep Research pipeline (067–072) for user-facing
research sessions. Decomposes a user question into research questions,
runs each through the pipeline, aggregates results, and generates
a unified final answer.
"""

from __future__ import annotations

import json
import logging
import string
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.deep_research import (
    DATA_DIR,
    generate_run_id,
    save_step_output,
)

logger = logging.getLogger("073_session")

# -- constants ---------------------------------------------------------------

SESSION_DIR = DATA_DIR.parent / "deep_research_sessions"

_LLM_MODEL = "claude-sonnet-4-20250514"

_DECOMPOSE_SYSTEM = """\
You are a research planning assistant.
Given a user question, decompose it into 1–5 independent research questions
that can each be investigated separately.

Rules:
- Each question must be self-contained and independently researchable
- If the original question is already focused, return it as a single question
- Questions should cover different aspects of the topic
- Keep each question concise (one sentence)

Return a JSON object with a single key "questions" containing a list of strings.
Return ONLY a JSON object. No markdown fences, no explanation."""

_ANSWER_WRITING_RULES = """\

## Writing Rules

- Write entirely in Japanese.
- Each section: write 2-5 substantive paragraphs, not just bullet points.
  Bullet lists are fine for enumerating items (products, clients) but MUST be
  wrapped with explanatory paragraphs giving context, background, or analysis.
- Be evidence-based: cite specific facts, numbers, dates, names from the
  research. After key facts, note the source in parentheses
  (e.g., (eurekarobotics.com)).
- Distinguish fact from interpretation: mark speculation with
  「〜と推測される」「〜の可能性がある」.
- If information for a section is limited, explicitly state
  「この領域については公開情報が限定的である」 and briefly note what is known.
- Aim for a report length of 3000-5000 Japanese characters.

## Prohibitions (anti-summarization)

- Do NOT compress multiple distinct facts into a single sentence.
- Do NOT use vague quantifiers (「様々な」「多くの」「いくつかの」) when
  specific items are available in the evidence.
- Do NOT skip details that are present in the evidence.
- Do NOT produce a short summary. This is a detailed report, not an abstract.
- Do NOT omit numbers, dates, or proper nouns that appear in the evidence.

Return the report as plain text with Markdown headings (not JSON)."""

_ANSWER_SYSTEM_COMPANY = """\
You are a research synthesis assistant producing detailed research reports.
Given multiple research run results (claims, evidence, memos, sources),
generate a detailed research report answering the user's original question
in Japanese.

## Output Structure

Use the following sections as a template. Include every section that has
relevant information. Skip a section ONLY if the research contains absolutely
no material for it. For skipped sections, do NOT add a placeholder.

1. **エグゼクティブサマリー** — 3-5 sentences covering the most critical findings.
2. **企業・対象の概要** — Background, history, founding, mission, positioning.
3. **製品・技術・サービス** — Offerings, technical differentiators, strengths.
4. **顧客・導入事例・パートナーシップ** — Key clients, use cases, partnerships.
5. **資金調達・収益・成長指標** — Funding rounds, revenue signals, growth.
6. **地域展開・組織体制** — Geographic presence, leadership, team.
7. **競合・市場ポジション** — Competitive landscape, market positioning.
8. **最近の動向とその意味合い** — Recent news with analysis of implications.
9. **リスク・不確実性・未確認事項** — What is uncertain or unverified.
10. **総合評価と示唆** — Overall assessment and actionable implications.
11. **主な情報源** — List of source domains used.
""" + _ANSWER_WRITING_RULES

_ANSWER_SYSTEM_PERSON = """\
You are a research synthesis assistant producing detailed person research reports.
Given multiple research run results (claims, evidence, memos, sources),
generate a detailed person research report answering the user's original question
in Japanese.

## Output Structure

Use the following sections as a template. Include every section that has
relevant information. Skip a section ONLY if the research contains absolutely
no material for it. For skipped sections, do NOT add a placeholder.

1. **エグゼクティブサマリー** — 3-5 sentences summarizing who this person is and why they matter.
2. **基本プロフィール** — Full name, birth year, nationality, birthplace, and other basic facts.
3. **学歴・教育背景** — Universities, degrees, majors, study abroad, academic honors.
4. **職歴・経歴** — Chronological career history, key career transitions, notable roles.
5. **所属組織・役職歴** — Board memberships, advisory roles, affiliations with organizations.
6. **専門領域・研究分野・関心領域** — Areas of expertise, research themes, intellectual interests.
7. **主要な発言・主張・思想** — Published statements, positions, op-eds, notable quotes.
8. **メディア露出・講演・執筆** — Media appearances, conference talks, books, articles, SNS presence.
9. **人脈・関係性** — Co-founders, mentors, key collaborators, professional network.
10. **人物像の総合評価と示唆** — Overall assessment of the person and actionable implications.
11. **リスク・不確実性・未確認事項** — What is uncertain or unverified about this person.
12. **主な情報源** — List of source domains used.
""" + _ANSWER_WRITING_RULES

_ANSWER_SYSTEM_GENERAL = """\
You are a research synthesis assistant producing detailed research reports.
Given multiple research run results (claims, evidence, memos, sources),
generate a detailed research report answering the user's original question
in Japanese.

## Output Structure

Use the following sections as a template. Include every section that has
relevant information. Skip a section ONLY if the research contains absolutely
no material for it. For skipped sections, do NOT add a placeholder.

1. **エグゼクティブサマリー** — 3-5 sentences covering the most critical findings.
2. **背景・概要** — Background context, definitions, scope of the topic.
3. **主要な調査結果** — Key findings organized by sub-topic or theme.
4. **分析・考察** — Analysis, implications, and interpretation of findings.
5. **最近の動向とその意味合い** — Recent developments and their significance.
6. **リスク・不確実性・未確認事項** — What is uncertain or unverified.
7. **総合評価と示唆** — Overall assessment and actionable implications.
8. **主な情報源** — List of source domains used.
""" + _ANSWER_WRITING_RULES

# intent → answer system prompt mapping
_ANSWER_SYSTEMS: Dict[str, str] = {
    "company_research": _ANSWER_SYSTEM_COMPANY,
    "person_research":  _ANSWER_SYSTEM_PERSON,
    "interview_prep":   _ANSWER_SYSTEM_PERSON,
    "tech_review":      _ANSWER_SYSTEM_GENERAL,
    "policy_analysis":  _ANSWER_SYSTEM_GENERAL,
    "issue_analysis":   _ANSWER_SYSTEM_GENERAL,
}
_ANSWER_SYSTEM_DEFAULT = _ANSWER_SYSTEM_GENERAL

# -- progress labels (Japanese) -----------------------------------------------

_STEP_LABELS = {
    "067": "調査計画を作成しています",
    "068": "情報源を収集しています",
    "069": "エビデンスを抽出しています",
    "070": "信頼度を評価しています",
    "071": "クレームを生成しています",
    "072": "メモを生成しています",
}


# -- session ID ---------------------------------------------------------------


def generate_session_id() -> str:
    """Generate unique session_id in format sess_YYYYMMDD_xxxxx."""
    today = datetime.now().strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"sess_{today}_{suffix}"


# -- question decomposition ---------------------------------------------------


def decompose_question(
    question: str,
    llm_client: Any,
) -> List[str]:
    """Decompose a user question into 1–5 research questions using LLM.

    Falls back to using the original question as-is if LLM fails.
    """
    body = {
        "model": _LLM_MODEL,
        "max_tokens": 1024,
        "system": _DECOMPOSE_SYSTEM,
        "messages": [{"role": "user", "content": question}],
    }

    for attempt in range(3):
        try:
            resp = llm_client.messages_create(body=body)
            resp_text = ""
            for block in resp.get("content", []):
                if block.get("type") == "text":
                    resp_text = block.get("text", "")
                    break

            if not resp_text:
                continue

            parsed = _parse_json(resp_text)
            if parsed and "questions" in parsed:
                questions = parsed["questions"]
                if isinstance(questions, list) and 1 <= len(questions) <= 5:
                    return [str(q) for q in questions]

        except Exception as e:
            logger.warning("Decomposition attempt %d failed: %s", attempt + 1, e)

    # Fallback: use original question as single research question
    logger.info("Decomposition fallback: using original question")
    return [question]


# -- user confirmation --------------------------------------------------------


def confirm_questions(questions: List[str]) -> str:
    """Display decomposed questions and get user confirmation.

    Returns 'y', 'e', or 'n'.
    """
    print()
    print("以下の調査を実行します:")
    print()
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        choice = input("この内容で調査を開始しますか？ [y/e/n]: ").strip().lower()
        if choice in ("y", "e", "n", ""):
            return choice if choice else "n"
        print("  y=実行, e=編集, n=キャンセル")


# -- pipeline execution -------------------------------------------------------


def _print_progress(question_idx: int, total_questions: int, question: str) -> None:
    """Print which question is being processed."""
    print()
    print(f"質問 {question_idx}/{total_questions} を処理しています: {question}")


def _print_step(step_idx: int, total_steps: int, step_key: str) -> None:
    """Print pipeline step progress."""
    label = _STEP_LABELS.get(step_key, step_key)
    print(f"  [{step_idx}/{total_steps}] {label}...")


def run_single_pipeline(
    question: str,
    run_id: str,
    *,
    llm_client: Any,
    search_client: Any,
    news_client: Any = None,
    notion_client: Any = None,
    enable_writeback: bool = False,
    question_idx: int = 1,
    total_questions: int = 1,
) -> Dict[str, Any]:
    """Execute the full 067–072 pipeline for a single research question.

    Returns a result dict with run_id, status, and summary data.
    """
    from src.deep_research.planner import run as run_planner
    from src.deep_research.collector import run as run_collector
    from src.deep_research.extractor import run as run_extractor
    from src.deep_research.credibility import run as run_credibility
    from src.deep_research.synthesizer import run as run_synthesizer
    from src.deep_research.publisher import run as run_publisher

    _print_progress(question_idx, total_questions, question)

    result: Dict[str, Any] = {
        "question": question,
        "run_id": run_id,
        "status": "failed",
        "intent": "general_research",
        "error": None,
        "sources_count": 0,
        "evidence_count": 0,
        "claims_count": 0,
        "memo_title": "",
        "memo_summary": "",
        "source_domains": [],
        "claims": [],
    }

    try:
        # Step 067: Planner
        _print_step(1, 6, "067")
        plan = run_planner(question, llm_client, run_id=run_id, notion_client=notion_client)
        save_step_output(run_id, "067", plan.to_dict())
        result["intent"] = plan.intent

        # Step 068: Collector
        _print_step(2, 6, "068")
        sources_data = run_collector(
            run_id=run_id,
            search_client=search_client,
            news_client=news_client,
        )
        save_step_output(run_id, "068", sources_data)

        # Step 069: Extractor
        _print_step(3, 6, "069")
        evidence_data = run_extractor(run_id=run_id, llm_client=llm_client)
        save_step_output(run_id, "069", evidence_data)

        # Step 070: Credibility
        _print_step(4, 6, "070")
        credibility_data = run_credibility(run_id=run_id)
        save_step_output(run_id, "070", credibility_data)

        # Step 071: Synthesizer
        _print_step(5, 6, "071")
        claims_data = run_synthesizer(run_id=run_id, llm_client=llm_client)
        save_step_output(run_id, "071", claims_data)

        # Step 072: Publisher
        _print_step(6, 6, "072")
        pub_result = run_publisher(
            run_id=run_id,
            notion_client=notion_client,
            enable_writeback=enable_writeback,
        )

        # Collect summary
        memo = pub_result.get("memo", {})
        sources = sources_data.get("sources", [])
        claims = claims_data.get("claims", [])

        domains = list(dict.fromkeys(
            s.get("domain", "") for s in sources if s.get("domain")
        ))

        # Top evidence for final answer context (high first, then medium)
        annotated = credibility_data.get("annotated_evidence", [])
        high_ev = [e.get("statement", "") for e in annotated
                   if e.get("confidence") == "high" and e.get("statement")]
        medium_ev = [e.get("statement", "") for e in annotated
                     if e.get("confidence") == "medium" and e.get("statement")]
        top_evidence = (high_ev + medium_ev)[:10]

        result.update({
            "status": "completed",
            "sources_count": len(sources),
            "evidence_count": len(evidence_data.get("evidence", [])),
            "claims_count": len(claims),
            "memo_title": memo.get("title", ""),
            "memo_summary": memo.get("summary", ""),
            "source_domains": domains,
            "claims": [
                {"statement": c.get("statement", ""), "confidence": c.get("confidence", "")}
                for c in claims
            ],
            "top_evidence": top_evidence,
        })

        print(f"  ✓ 完了 (sources: {result['sources_count']}, "
              f"evidence: {result['evidence_count']}, "
              f"claims: {result['claims_count']})")

    except Exception as e:
        result["error"] = str(e)
        logger.error("Pipeline failed for run_id=%s: %s", run_id, e)
        print(f"  ✗ 失敗: {e}")

    return result


# -- result aggregation -------------------------------------------------------


def aggregate_results(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple pipeline runs."""
    all_claims: List[Dict[str, Any]] = []
    all_domains: List[str] = []
    summaries: List[str] = []
    seen_domains: set = set()

    total_sources = 0
    total_evidence = 0

    for r in run_results:
        if r["status"] != "completed":
            continue
        all_claims.extend(r.get("claims", []))
        total_sources += r.get("sources_count", 0)
        total_evidence += r.get("evidence_count", 0)

        for d in r.get("source_domains", []):
            if d not in seen_domains:
                seen_domains.add(d)
                all_domains.append(d)

        if r.get("memo_summary"):
            summaries.append(r["memo_summary"])

    return {
        "total_sources": total_sources,
        "total_evidence": total_evidence,
        "total_claims": len(all_claims),
        "claims": all_claims,
        "source_domains": all_domains,
        "summaries": summaries,
        "completed_runs": sum(1 for r in run_results if r["status"] == "completed"),
        "failed_runs": sum(1 for r in run_results if r["status"] == "failed"),
    }


# -- final answer generation --------------------------------------------------


def generate_final_answer(
    original_question: str,
    aggregated: Dict[str, Any],
    run_results: List[Dict[str, Any]],
    llm_client: Any,
    intent: str = "general_research",
) -> str:
    """Generate a unified final answer from aggregated results."""
    # Select system prompt based on intent
    system_prompt = _ANSWER_SYSTEMS.get(intent, _ANSWER_SYSTEM_DEFAULT)
    logger.info("Final answer: using %s framework (intent=%s)", intent, intent)

    # Build context for LLM
    context_parts: List[str] = []
    context_parts.append(f"User question: {original_question}")
    context_parts.append("")

    for i, r in enumerate(run_results, 1):
        if r["status"] != "completed":
            context_parts.append(f"Research {i} ({r['question']}): FAILED — {r.get('error', 'unknown')}")
            continue
        context_parts.append(f"Research {i}: {r['question']}")
        if r.get("memo_summary"):
            context_parts.append(f"  Summary: {r['memo_summary']}")
        for ev in r.get("top_evidence", []):
            context_parts.append(f"  Evidence: {ev}")
        for c in r.get("claims", []):
            context_parts.append(f"  Claim [{c.get('confidence', '?')}]: {c.get('statement', '')}")
        context_parts.append("")

    context_parts.append(f"Source domains: {', '.join(aggregated.get('source_domains', []))}")

    user_msg = "\n".join(context_parts)

    body = {
        "model": _LLM_MODEL,
        "max_tokens": 8192,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_msg}],
    }

    for attempt in range(3):
        try:
            resp = llm_client.messages_create(body=body)
            for block in resp.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        return text
        except Exception as e:
            logger.warning("Final answer generation attempt %d failed: %s", attempt + 1, e)

    # Fallback: concatenate per-run summaries
    logger.info("Final answer fallback: concatenating summaries")
    return _build_fallback_answer(original_question, aggregated, run_results)


def _build_fallback_answer(
    question: str,
    aggregated: Dict[str, Any],
    run_results: List[Dict[str, Any]],
) -> str:
    """Build fallback answer when LLM is unavailable."""
    lines: List[str] = []
    lines.append(f"# {question}")
    lines.append("")

    lines.append("## エグゼクティブサマリー")
    lines.append("")
    for r in run_results:
        if r["status"] == "completed" and r.get("memo_summary"):
            lines.append(r["memo_summary"])
            lines.append("")

    lines.append("## 主な調査結果")
    lines.append("")
    for i, c in enumerate(aggregated.get("claims", [])[:15], 1):
        lines.append(f"{i}. {c.get('statement', '')}")
    lines.append("")

    lines.append("## 主なエビデンス")
    lines.append("")
    all_evidence: List[str] = []
    for r in run_results:
        if r["status"] == "completed":
            all_evidence.extend(r.get("top_evidence", []))
    for ev in all_evidence[:20]:
        lines.append(f"- {ev}")
    lines.append("")

    lines.append("## リスク・不確実性")
    lines.append("")
    lines.append("※ LLM による統合分析が利用できなかったため、上記は個別調査結果の機械的な結合です。")
    lines.append("")

    lines.append("## 主な情報源")
    lines.append("")
    for d in aggregated.get("source_domains", [])[:15]:
        lines.append(f"- {d}")

    return "\n".join(lines)


# -- local storage ------------------------------------------------------------


def _ensure_session_dir(session_id: str) -> Path:
    """Create and return session output directory."""
    session_dir = SESSION_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_session(
    session_id: str,
    user_question: str,
    decomposed_questions: List[str],
    run_results: List[Dict[str, Any]],
    final_answer: str,
    status: str,
    created_at: str,
) -> Path:
    """Save session.json and session_answer.md to local storage."""
    session_dir = _ensure_session_dir(session_id)

    # session.json
    session_data = {
        "session_id": session_id,
        "user_question": user_question,
        "decomposed_questions": decomposed_questions,
        "runs": [
            {
                "question": r["question"],
                "run_id": r["run_id"],
                "status": r["status"],
                "sources_count": r.get("sources_count", 0),
                "evidence_count": r.get("evidence_count", 0),
                "claims_count": r.get("claims_count", 0),
                "error": r.get("error"),
            }
            for r in run_results
        ],
        "final_answer": final_answer,
        "status": status,
        "created_at": created_at,
        "completed_at": datetime.now().isoformat(),
    }

    session_path = session_dir / "session.json"
    session_path.write_text(json.dumps(session_data, ensure_ascii=False, indent=2))
    logger.info("session.json saved: %s", session_path)

    # session_answer.md
    answer_path = session_dir / "session_answer.md"
    answer_path.write_text(final_answer, encoding="utf-8")
    logger.info("session_answer.md saved: %s", answer_path)

    return session_dir


# -- display ------------------------------------------------------------------


def display_final_answer(
    final_answer: str,
    aggregated: Dict[str, Any],
    session_dir: Path,
) -> None:
    """Display the final answer on terminal."""
    print()
    print("=" * 60)
    print("調査が完了しました。")
    print("=" * 60)
    print()
    print(final_answer)
    print()
    print(f"詳細: {session_dir / 'session_answer.md'}")
    print()


# -- main orchestrator --------------------------------------------------------


def run_session(
    question: str,
    *,
    llm_client: Any,
    search_client: Any,
    news_client: Any = None,
    notion_client: Any = None,
    enable_writeback: bool = False,
) -> Dict[str, Any]:
    """Execute a full research session.

    1. Decompose the user question
    2. Get user confirmation
    3. Run pipeline for each question
    4. Aggregate results
    5. Generate final answer
    6. Save and display

    Returns session metadata dict.
    """
    session_id = generate_session_id()
    created_at = datetime.now().isoformat()
    logger.info("Session %s started", session_id)

    # 1. Decompose
    print()
    print("質問を分解しています...")
    questions = decompose_question(question, llm_client)

    # 2. Confirm
    choice = confirm_questions(questions)

    if choice == "n":
        print("セッションを終了しました。")
        session_dir = save_session(
            session_id, question, questions, [], "", "cancelled", created_at,
        )
        return {
            "session_id": session_id,
            "status": "cancelled",
            "output_path": str(session_dir),
        }

    if choice == "e":
        # Re-input question
        new_question = input("質問を再入力してください: ").strip()
        if not new_question:
            print("セッションを終了しました。")
            return {
                "session_id": session_id,
                "status": "cancelled",
                "output_path": "",
            }
        question = new_question
        print()
        print("質問を分解しています...")
        questions = decompose_question(question, llm_client)

        # Re-confirm
        choice2 = confirm_questions(questions)
        if choice2 != "y":
            print("セッションを終了しました。")
            session_dir = save_session(
                session_id, question, questions, [], "", "cancelled", created_at,
            )
            return {
                "session_id": session_id,
                "status": "cancelled",
                "output_path": str(session_dir),
            }

    # 3. Execute pipeline for each question
    print()
    print("調査を開始します...")

    run_results: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        run_id = generate_run_id()
        result = run_single_pipeline(
            question=q,
            run_id=run_id,
            llm_client=llm_client,
            search_client=search_client,
            news_client=news_client,
            notion_client=notion_client,
            enable_writeback=enable_writeback,
            question_idx=i,
            total_questions=len(questions),
        )
        run_results.append(result)

    # 4. Aggregate
    aggregated = aggregate_results(run_results)

    # 5. Generate final answer
    completed = aggregated["completed_runs"]
    failed = aggregated["failed_runs"]

    if completed == 0:
        final_answer = "すべての調査が失敗しました。しばらく時間をおいて再度お試しください。"
        status = "failed"
    else:
        # Determine dominant intent from completed runs
        session_intent = "general_research"
        for r in run_results:
            if r["status"] == "completed":
                session_intent = r.get("intent", "general_research")
                break

        print()
        print("統合回答を生成中...")
        final_answer = generate_final_answer(
            question, aggregated, run_results, llm_client,
            intent=session_intent,
        )
        if failed > 0:
            final_answer += (
                f"\n\n※ 一部の調査（{failed}件）は完了しませんでした。"
                "結果に含まれていない情報がある可能性があります。"
            )
        status = "completed" if failed == 0 else "partial"

    # 6. Save
    session_dir = save_session(
        session_id, question, questions, run_results, final_answer, status,
        created_at,
    )

    # 7. Display
    display_final_answer(final_answer, aggregated, session_dir)

    logger.info(
        "Session %s done: status=%s, runs=%d/%d, claims=%d, sources=%d",
        session_id, status, completed, len(questions),
        aggregated["total_claims"], aggregated["total_sources"],
    )

    return {
        "session_id": session_id,
        "user_question": question,
        "decomposed_questions": questions,
        "run_ids": [r["run_id"] for r in run_results],
        "status": status,
        "final_answer": final_answer,
        "output_path": str(session_dir),
    }


# -- helpers -------------------------------------------------------------------


def _parse_json(text: str) -> Optional[dict]:
    """Parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    return None
