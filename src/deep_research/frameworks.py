"""Framework registry for topic-adaptive research.

Centralises ResearchFramework definitions and answer system prompts.
Phase 1: migrates existing 3 prompt templates (company, person, general)
from session.py as manual frameworks.

Usage:
    from src.deep_research.frameworks import get_framework, get_framework_by_intent

    fw = get_framework("company", "general")
    fw = get_framework_by_intent("company_research")
    prompt = fw.answer_system_prompt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# -- dataclasses --------------------------------------------------------------


@dataclass
class SectionDef:
    """Definition of a single report section within a framework."""

    title: str              # e.g. "創業者・経営チーム"
    guidance: str           # what to investigate for this section
    priority: str = "normal"  # "critical" | "normal" | "optional"


@dataclass
class ResearchFramework:
    """A research framework that determines report structure and prompts."""

    framework_id: str           # e.g. "company.general"
    topic: str                  # e.g. "company"
    subtype: str                # e.g. "general"
    display_name: str           # e.g. "企業調査（汎用）"
    section_outline: List[SectionDef]
    search_hints: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    answer_system_prompt: str = ""
    _prompt_source: str = "generated"  # "manual" | "generated"


# -- shared writing rules (appended to all generated prompts) -----------------

ANSWER_WRITING_RULES = """

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


# -- build_answer_prompt helper -----------------------------------------------


def build_answer_prompt(fw: ResearchFramework) -> str:
    """Generate answer_system_prompt from SectionDef list.

    Used for generated frameworks (Phase 2+). Appends ANSWER_WRITING_RULES
    automatically. Manual frameworks already embed writing rules in their
    answer_system_prompt and should NOT use this function.
    """
    header = (
        "You are a research synthesis assistant producing detailed research reports.\n"
        "Given multiple research run results (claims, evidence, memos, sources),\n"
        "generate a detailed research report answering the user's original question\n"
        "in Japanese.\n\n"
        "## Output Structure\n\n"
        "Use the following sections as a template. Include every section that has\n"
        "relevant information. Skip a section ONLY if the research contains absolutely\n"
        "no material for it. For skipped sections, do NOT add a placeholder.\n"
    )
    sections_text = "\n".join(
        f"{i}. **{s.title}** — {s.guidance}"
        for i, s in enumerate(fw.section_outline, 1)
    )
    return header + "\n" + sections_text + ANSWER_WRITING_RULES


# -- manual prompt templates (migrated from session.py) -----------------------

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
""" + ANSWER_WRITING_RULES

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
""" + ANSWER_WRITING_RULES

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
""" + ANSWER_WRITING_RULES


# -- framework registry -------------------------------------------------------

FRAMEWORKS: Dict[str, ResearchFramework] = {
    "company.general": ResearchFramework(
        framework_id="company.general",
        topic="company",
        subtype="general",
        display_name="企業調査（汎用）",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences covering the most critical findings."),
            SectionDef("企業・対象の概要", "Background, history, founding, mission, positioning."),
            SectionDef("製品・技術・サービス", "Offerings, technical differentiators, strengths."),
            SectionDef("顧客・導入事例・パートナーシップ", "Key clients, use cases, partnerships."),
            SectionDef("資金調達・収益・成長指標", "Funding rounds, revenue signals, growth."),
            SectionDef("地域展開・組織体制", "Geographic presence, leadership, team."),
            SectionDef("競合・市場ポジション", "Competitive landscape, market positioning."),
            SectionDef("最近の動向とその意味合い", "Recent news with analysis of implications."),
            SectionDef("リスク・不確実性・未確認事項", "What is uncertain or unverified."),
            SectionDef("総合評価と示唆", "Overall assessment and actionable implications."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "company_brief"],
        answer_system_prompt=_ANSWER_SYSTEM_COMPANY,
        _prompt_source="manual",
    ),
    "company.startup": ResearchFramework(
        framework_id="company.startup",
        topic="company",
        subtype="startup",
        display_name="スタートアップ調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: what the startup does, stage, traction, and why it matters.", "critical"),
            SectionDef("企業概要・沿革", "Founding year, founders, mission, legal entity, HQ location, brief history."),
            SectionDef("創業者・経営チーム", "Founder backgrounds, prior exits, domain expertise, key hires, team size.", "critical"),
            SectionDef("プロダクト・技術", "Core product, technical differentiators, tech stack, IP/patents, roadmap.", "critical"),
            SectionDef("PMF・トラクション", "Key metrics (ARR, MRR, users, growth rate), product-market fit signals, retention.", "critical"),
            SectionDef("資金調達・バリュエーション", "Funding rounds, investors, valuation, cap table highlights, runway.", "critical"),
            SectionDef("ビジネスモデル・収益構造", "Revenue model, pricing, unit economics, path to profitability."),
            SectionDef("顧客・導入事例", "Key customers, use cases, logos, case studies, NPS/satisfaction signals."),
            SectionDef("競合・市場ポジション", "Direct/indirect competitors, differentiation, market share estimate."),
            SectionDef("最近の動向とその意味合い", "Recent news, launches, partnerships, pivots — with analysis of implications."),
            SectionDef("リスク・不確実性・未確認事項", "Key risks: market, execution, funding, regulatory. What is unverified."),
            SectionDef("総合評価と示唆", "Overall assessment and actionable implications for the reader."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "company_brief"],
        _prompt_source="generated",
    ),
    "company.enterprise": ResearchFramework(
        framework_id="company.enterprise",
        topic="company",
        subtype="enterprise",
        display_name="大企業・上場企業調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: company position, scale, recent trajectory, and significance.", "critical"),
            SectionDef("企業概要・沿革", "Founding, history, corporate structure, HQ, employee count, global presence."),
            SectionDef("事業セグメント・製品ポートフォリオ", "Business segments, revenue breakdown by segment, key products/services.", "critical"),
            SectionDef("財務実績・成長指標", "Revenue, profit, margins, YoY growth, stock performance, market cap.", "critical"),
            SectionDef("顧客基盤・導入事例・パートナーシップ", "Major clients, strategic partnerships, ecosystem, channel strategy.", "critical"),
            SectionDef("経営陣・ガバナンス", "CEO/C-suite, board composition, governance structure, leadership changes."),
            SectionDef("技術・R&D・イノベーション", "R&D spend, patents, technology strategy, digital transformation initiatives."),
            SectionDef("地域展開・組織体制", "Geographic footprint, subsidiary structure, regional strategies."),
            SectionDef("競合・市場ポジション", "Market share, competitive landscape, positioning vs peers."),
            SectionDef("最近の動向とその意味合い", "M&A, restructuring, strategic shifts, earnings surprises — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Regulatory, competitive, macroeconomic risks. What is uncertain."),
            SectionDef("総合評価と示唆", "Overall assessment and actionable implications."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "company_brief"],
        _prompt_source="generated",
    ),
    "company.lp": ResearchFramework(
        framework_id="company.lp",
        topic="company",
        subtype="lp",
        display_name="機関投資家（LP）調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: who this LP is, AUM, investment approach, and significance.", "critical"),
            SectionDef("組織概要・沿革", "Founding, type (pension/SWF/endowment/insurance), governance, beneficiaries."),
            SectionDef("運用資産・規模", "AUM, AUM growth trajectory, fund inflows/outflows.", "critical"),
            SectionDef("運用方針・アセットアロケーション", "Target allocation across asset classes, PE/VC allocation %, policy benchmarks.", "critical"),
            SectionDef("GP選定基準・投資プロセス", "How they select GPs, due diligence process, co-investment appetite, mandate criteria.", "critical"),
            SectionDef("ポートフォリオ構成・主要GP関係", "Key GP relationships, fund commitments, vintage diversification."),
            SectionDef("パフォーマンス・実績", "Historical returns, benchmark comparison, risk-adjusted performance."),
            SectionDef("ESG・責任投資方針", "ESG integration, exclusion policies, stewardship activities."),
            SectionDef("組織体制・意思決定構造", "CIO, investment committee, internal vs outsourced management, staffing."),
            SectionDef("最近の動向とその意味合い", "Recent mandate changes, new GP commitments, policy shifts — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Liability mismatch, political risk, liquidity constraints. What is unverified."),
            SectionDef("総合評価と示唆", "Overall assessment and implications for GP fundraising or market analysis."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "company_brief"],
        _prompt_source="generated",
    ),
    "company.vc": ResearchFramework(
        framework_id="company.vc",
        topic="company",
        subtype="vc",
        display_name="VC・PE ファンド調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: fund identity, strategy, scale, and significance.", "critical"),
            SectionDef("ファンド概要・沿革", "Founding, GP team history, fund lineage (Fund I/II/III…), HQ, offices."),
            SectionDef("ファンド規模・資金調達", "Latest fund size, total AUM, LP base composition, fundraising track record.", "critical"),
            SectionDef("投資テーマ・戦略", "Stage focus, sector focus, geographic focus, thesis, check size range.", "critical"),
            SectionDef("ポートフォリオ構成・代表投資先", "Notable portfolio companies, exits (IPO/M&A), current portfolio breakdown.", "critical"),
            SectionDef("投資実績・パフォーマンス", "IRR/MOIC (if public), notable exits with returns, DPI/TVPI signals."),
            SectionDef("GP チーム・意思決定構造", "Key partners, investment committee, decision process, team background.", "critical"),
            SectionDef("バリューアド・支援体制", "Platform team, operational support, network, board involvement."),
            SectionDef("LP基盤・ファンドレイズ動向", "Key LP relationships, recent fundraising, oversubscription signals."),
            SectionDef("競合・市場ポジション", "Competitive positioning vs peer funds, differentiation."),
            SectionDef("最近の動向とその意味合い", "Recent investments, exits, fund launches, strategy shifts — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Key person risk, fund performance uncertainty, market cycle exposure. What is unverified."),
            SectionDef("総合評価と示唆", "Overall assessment and implications for founders or co-investors."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "company_brief"],
        _prompt_source="generated",
    ),
    "company.academic": ResearchFramework(
        framework_id="company.academic",
        topic="company",
        subtype="academic",
        display_name="大学・研究機関調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: institution identity, research strengths, and significance.", "critical"),
            SectionDef("機関概要・沿革", "Founding, type (university/institute/think tank), governance, location, scale."),
            SectionDef("研究分野・主要成果", "Core research areas, landmark papers/discoveries, citation metrics, rankings.", "critical"),
            SectionDef("主要研究者・ラボ", "Key faculty/PIs, notable labs, research group structure.", "critical"),
            SectionDef("産学連携・技術移転", "Industry partnerships, spin-offs, licensing, joint research programs, TLO activity.", "critical"),
            SectionDef("資金・予算構造", "Research funding sources, government grants, endowment, corporate funding."),
            SectionDef("教育・人材輩出", "Notable programs, alumni network, talent pipeline to industry."),
            SectionDef("国際連携・ネットワーク", "International partnerships, exchange programs, global rankings context."),
            SectionDef("施設・インフラ", "Research facilities, computing resources, special equipment or datasets."),
            SectionDef("最近の動向とその意味合い", "Recent breakthroughs, new centers, policy changes — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Funding risks, regulatory changes, brain drain. What is unverified."),
            SectionDef("総合評価と示唆", "Overall assessment and implications for collaboration or investment."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "company_brief"],
        _prompt_source="generated",
    ),
    "person.executive": ResearchFramework(
        framework_id="person.executive",
        topic="person",
        subtype="executive",
        display_name="経営者・役員調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: who this executive is, current role, career highlights, and significance.", "critical"),
            SectionDef("基本プロフィール", "Full name, birth year, nationality, education background, degrees."),
            SectionDef("経歴・キャリアパス", "Chronological career history, key transitions, promotions, cross-industry moves.", "critical"),
            SectionDef("現職・役職", "Current title, responsibilities, reporting structure, tenure in role.", "critical"),
            SectionDef("取締役・アドバイザリー歴", "Board seats (current and past), advisory roles, committee memberships.", "critical"),
            SectionDef("経営実績・戦略的意思決定", "Major strategic decisions, M&A, turnarounds, product launches, growth initiatives.", "critical"),
            SectionDef("リーダーシップスタイル・評判", "Management philosophy, leadership reputation, employee/market perception."),
            SectionDef("報酬・持株・インセンティブ", "Compensation structure, equity holdings, stock options (if public)."),
            SectionDef("業界での影響力・人脈", "Industry associations, conference keynotes, peer network, mentorship."),
            SectionDef("主要な発言・主張", "Published statements, interviews, op-eds, notable public positions."),
            SectionDef("最近の動向とその意味合い", "Recent role changes, strategic moves, public appearances — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Controversies, litigation, governance concerns. What is unverified."),
            SectionDef("総合評価と示唆", "Overall assessment and implications for business dealings or partnerships."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "person_brief"],
        _prompt_source="generated",
    ),
    "person.academic_person": ResearchFramework(
        framework_id="person.academic_person",
        topic="person",
        subtype="academic_person",
        display_name="研究者・学術人物調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: who this researcher is, field, key contributions, and significance.", "critical"),
            SectionDef("基本プロフィール", "Full name, birth year, nationality, current affiliation, title/position."),
            SectionDef("学歴・学位", "Universities, PhD topic/advisor, postdoc positions, academic lineage."),
            SectionDef("研究分野・専門領域", "Core research areas, sub-fields, methodological expertise.", "critical"),
            SectionDef("主要業績・代表論文", "Landmark publications, citation counts, h-index, patents, awards.", "critical"),
            SectionDef("研究室・チーム", "Lab/group composition, current students/postdocs, notable alumni.", "critical"),
            SectionDef("研究資金・グラント", "Major grants (NSF, JSPS, ERC, etc.), funding amounts, PI vs co-PI.", "critical"),
            SectionDef("産学連携・技術移転", "Industry collaborations, consulting, spin-off companies, advisory roles."),
            SectionDef("学会活動・編集委員", "Conference organizing, journal editorial boards, society memberships."),
            SectionDef("講演・メディア・執筆", "Invited talks, media appearances, books, popular science writing."),
            SectionDef("人脈・共同研究者ネットワーク", "Key co-authors, international collaborators, mentorship relationships."),
            SectionDef("最近の動向とその意味合い", "Recent publications, grants, lab moves, new projects — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Retraction concerns, funding gaps, career uncertainty. What is unverified."),
            SectionDef("総合評価と示唆", "Overall assessment and implications for collaboration or recruitment."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "person_brief"],
        _prompt_source="generated",
    ),
    "person.investor_person": ResearchFramework(
        framework_id="person.investor_person",
        topic="person",
        subtype="investor_person",
        display_name="投資家個人調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: who this investor is, current fund/role, track record, and significance.", "critical"),
            SectionDef("基本プロフィール", "Full name, birth year, nationality, education, early career background."),
            SectionDef("経歴・キャリアパス", "Career history from operator/analyst to investor, key transitions."),
            SectionDef("現職・所属ファンド", "Current fund, title, role scope, fund size context.", "critical"),
            SectionDef("投資トラックレコード", "Notable investments, exits (IPO/M&A), returns where public, hit rate.", "critical"),
            SectionDef("投資哲学・テーゼ", "Investment thesis, stage/sector preferences, decision-making framework.", "critical"),
            SectionDef("ポートフォリオ・代表投資先", "Current portfolio companies, sector distribution, geographic focus."),
            SectionDef("取締役参画・ハンズオン支援", "Board seats at portfolio companies, value-add approach, operational involvement.", "critical"),
            SectionDef("エンジェル・個人投資活動", "Angel investments, SPVs, personal fund, scout programs (if applicable)."),
            SectionDef("業界での影響力・人脈", "LP relationships, co-investor network, industry reputation, media presence."),
            SectionDef("主要な発言・主張", "Published views on markets, blog posts, podcast appearances, social media."),
            SectionDef("最近の動向とその意味合い", "Recent investments, fund launches, role changes — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Performance uncertainty, conflicts of interest, reputation risks. What is unverified."),
            SectionDef("総合評価と示唆", "Overall assessment and implications for co-investment or fundraising."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "person_brief"],
        _prompt_source="generated",
    ),
    "person.general": ResearchFramework(
        framework_id="person.general",
        topic="person",
        subtype="general",
        display_name="人物調査（汎用）",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences summarizing who this person is and why they matter."),
            SectionDef("基本プロフィール", "Full name, birth year, nationality, birthplace, and other basic facts."),
            SectionDef("学歴・教育背景", "Universities, degrees, majors, study abroad, academic honors."),
            SectionDef("職歴・経歴", "Chronological career history, key career transitions, notable roles."),
            SectionDef("所属組織・役職歴", "Board memberships, advisory roles, affiliations with organizations."),
            SectionDef("専門領域・研究分野・関心領域", "Areas of expertise, research themes, intellectual interests."),
            SectionDef("主要な発言・主張・思想", "Published statements, positions, op-eds, notable quotes."),
            SectionDef("メディア露出・講演・執筆", "Media appearances, conference talks, books, articles, SNS presence."),
            SectionDef("人脈・関係性", "Co-founders, mentors, key collaborators, professional network."),
            SectionDef("人物像の総合評価と示唆", "Overall assessment of the person and actionable implications."),
            SectionDef("リスク・不確実性・未確認事項", "What is uncertain or unverified about this person."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo", "person_brief"],
        answer_system_prompt=_ANSWER_SYSTEM_PERSON,
        _prompt_source="manual",
    ),
    "market.general": ResearchFramework(
        framework_id="market.general",
        topic="market",
        subtype="general",
        display_name="市場・業界調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: market identity, size, growth trajectory, and key dynamics.", "critical"),
            SectionDef("市場定義・スコープ", "Market definition, boundaries, included/excluded segments, geographic scope.", "critical"),
            SectionDef("市場規模・成長率", "TAM/SAM/SOM, historical and projected growth rates, CAGR, revenue pools.", "critical"),
            SectionDef("市場構造・バリューチェーン", "Industry value chain, key stages, margin distribution, vertical integration trends."),
            SectionDef("主要プレイヤー・競合環境", "Market leaders, challengers, market share estimates, competitive dynamics.", "critical"),
            SectionDef("需要ドライバー・成長要因", "Demand drivers, adoption catalysts, tailwinds, customer trends."),
            SectionDef("技術トレンド・イノベーション", "Technology shifts affecting the market, emerging solutions, disruption vectors."),
            SectionDef("規制・政策環境", "Regulatory framework, compliance requirements, policy trends affecting the market."),
            SectionDef("地域別動向", "Regional differences, geographic hotspots, cross-border dynamics."),
            SectionDef("最近の動向とその意味合い", "Recent M&A, funding, product launches, policy changes — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Market risks: cyclicality, substitution, regulatory, macroeconomic. What is unverified."),
            SectionDef("総合評価と示唆", "Overall market assessment, opportunities, and strategic implications."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo"],
        _prompt_source="generated",
    ),
    "technology.general": ResearchFramework(
        framework_id="technology.general",
        topic="technology",
        subtype="general",
        display_name="技術・テクノロジー調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: what this technology is, maturity, adoption, and significance.", "critical"),
            SectionDef("技術概要・原理", "Core principles, how it works, key components, technical architecture.", "critical"),
            SectionDef("発展経緯・技術史", "Historical evolution, key milestones, breakthrough moments, versioning."),
            SectionDef("技術的優位性・差別化", "Performance characteristics, benchmarks, advantages over alternatives.", "critical"),
            SectionDef("成熟度・採用段階", "TRL level, adoption curve position, production readiness, scalability.", "critical"),
            SectionDef("主要プレイヤー・開発者", "Companies, research groups, open-source communities driving development."),
            SectionDef("ユースケース・応用領域", "Primary and emerging use cases, industry applications, deployment examples.", "critical"),
            SectionDef("エコシステム・関連技術", "Supporting technologies, toolchains, standards, interoperability."),
            SectionDef("研究動向・最新論文", "Recent academic papers, preprints, patents, research directions."),
            SectionDef("課題・技術的制約", "Current limitations, unsolved problems, engineering challenges."),
            SectionDef("最近の動向とその意味合い", "Recent breakthroughs, releases, funding, partnerships — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Technical risks, hype vs reality, ethical concerns. What is unverified."),
            SectionDef("総合評価と示唆", "Overall technology assessment and implications for adoption or investment."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo"],
        _prompt_source="generated",
    ),
    "policy.general": ResearchFramework(
        framework_id="policy.general",
        topic="policy",
        subtype="general",
        display_name="政策・規制調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: what this policy/regulation is, scope, status, and significance.", "critical"),
            SectionDef("政策・規制の概要", "Policy name, issuing body, legal basis, scope of application, effective dates.", "critical"),
            SectionDef("背景・制定経緯", "Problem being addressed, legislative history, political context, prior versions."),
            SectionDef("主要条項・要件", "Key provisions, obligations, thresholds, definitions, compliance requirements.", "critical"),
            SectionDef("対象・適用範囲", "Who is affected: industries, entities, jurisdictions, exemptions."),
            SectionDef("施行・執行体制", "Enforcement agency, penalties, audit mechanisms, reporting requirements.", "critical"),
            SectionDef("ステークホルダーの反応", "Industry response, advocacy groups, public opinion, lobbying positions."),
            SectionDef("国際比較・他法域との関係", "Comparable regulations in other jurisdictions, cross-border implications.", "critical"),
            SectionDef("影響分析", "Economic impact, compliance costs, market effects, innovation implications."),
            SectionDef("最近の動向とその意味合い", "Recent amendments, court rulings, enforcement actions, public consultations — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Regulatory uncertainty, pending changes, jurisdictional conflicts. What is unverified."),
            SectionDef("総合評価と示唆", "Overall policy assessment and implications for affected stakeholders."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo"],
        _prompt_source="generated",
    ),
    "product.general": ResearchFramework(
        framework_id="product.general",
        topic="product",
        subtype="general",
        display_name="プロダクト・サービス調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences: what this product is, who it serves, market position, and significance.", "critical"),
            SectionDef("プロダクト概要", "Product name, category, provider, launch date, current version, platform.", "critical"),
            SectionDef("機能・特徴", "Core features, key capabilities, technical specifications, UX highlights.", "critical"),
            SectionDef("ターゲットユーザー・利用シーン", "Target audience, personas, primary use cases, adoption context."),
            SectionDef("価格・ビジネスモデル", "Pricing tiers, licensing model, freemium/premium, TCO considerations.", "critical"),
            SectionDef("競合製品・比較", "Direct competitors, feature comparison, positioning, switching costs.", "critical"),
            SectionDef("技術基盤・アーキテクチャ", "Underlying technology, integrations, APIs, extensibility."),
            SectionDef("ユーザー評価・レビュー", "User reviews, ratings, NPS signals, common praise and complaints."),
            SectionDef("導入事例・実績", "Notable customers, case studies, deployment scale, ROI evidence."),
            SectionDef("ロードマップ・将来展望", "Announced features, development direction, strategic roadmap signals."),
            SectionDef("最近の動向とその意味合い", "Recent releases, updates, partnerships, market moves — with analysis."),
            SectionDef("リスク・不確実性・未確認事項", "Vendor lock-in, sustainability, feature gaps. What is unverified."),
            SectionDef("総合評価と示唆", "Overall product assessment and implications for adoption or procurement."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo"],
        _prompt_source="generated",
    ),
    "general.general": ResearchFramework(
        framework_id="general.general",
        topic="general",
        subtype="general",
        display_name="汎用調査",
        section_outline=[
            SectionDef("エグゼクティブサマリー", "3-5 sentences covering the most critical findings."),
            SectionDef("背景・概要", "Background context, definitions, scope of the topic."),
            SectionDef("主要な調査結果", "Key findings organized by sub-topic or theme."),
            SectionDef("分析・考察", "Analysis, implications, and interpretation of findings."),
            SectionDef("最近の動向とその意味合い", "Recent developments and their significance."),
            SectionDef("リスク・不確実性・未確認事項", "What is uncertain or unverified."),
            SectionDef("総合評価と示唆", "Overall assessment and actionable implications."),
            SectionDef("主な情報源", "List of source domains used."),
        ],
        deliverables=["evidence_memo"],
        answer_system_prompt=_ANSWER_SYSTEM_GENERAL,
        _prompt_source="manual",
    ),
}

# Auto-generate answer_system_prompt for generated frameworks
for _fw in FRAMEWORKS.values():
    if _fw._prompt_source == "generated" and not _fw.answer_system_prompt:
        _fw.answer_system_prompt = build_answer_prompt(_fw)


# -- intent → framework mapping (backward compatibility) ----------------------

_INTENT_TO_FRAMEWORK: Dict[str, str] = {
    "company_research": "company.general",
    "person_research":  "person.general",
    "interview_prep":   "person.general",
    "tech_review":      "technology.general",
    "policy_analysis":  "policy.general",
    "issue_analysis":   "general.general",
}


# -- lookup functions ---------------------------------------------------------


def get_framework(topic: str, subtype: str) -> ResearchFramework:
    """Look up framework by topic + subtype with fallback chain.

    Fallback: {topic}.{subtype} → {topic}.general → general.general
    """
    key = f"{topic}.{subtype}"
    if key in FRAMEWORKS:
        return FRAMEWORKS[key]
    topic_key = f"{topic}.general"
    if topic_key in FRAMEWORKS:
        return FRAMEWORKS[topic_key]
    return FRAMEWORKS["general.general"]


def get_framework_by_intent(intent: str) -> ResearchFramework:
    """Look up framework by legacy intent string.

    Used for backward compatibility during Phase 1 where planner still
    returns intent only (no topic/subtype).
    """
    fw_id = _INTENT_TO_FRAMEWORK.get(intent, "general.general")
    return FRAMEWORKS.get(fw_id, FRAMEWORKS["general.general"])


def get_framework_by_id(framework_id: str) -> ResearchFramework:
    """Look up framework by dotted framework_id with staged fallback.

    Fallback chain for "topic.subtype.variant":
      topic.subtype.variant → topic.subtype → topic.general → general.general

    For "topic.subtype" (not in registry):
      topic.subtype → topic.general → general.general
    """
    if framework_id in FRAMEWORKS:
        return FRAMEWORKS[framework_id]
    # Drop rightmost segment and retry
    parts = framework_id.rsplit(".", 1)
    if len(parts) > 1:
        parent = parts[0]
        if parent in FRAMEWORKS:
            return FRAMEWORKS[parent]
        # parent not found either — try topic.general
        topic = framework_id.split(".")[0]
        topic_general = f"{topic}.general"
        if topic_general in FRAMEWORKS:
            return FRAMEWORKS[topic_general]
    return FRAMEWORKS["general.general"]
