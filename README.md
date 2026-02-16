▪️ Research OS 100 — Overview:

Research OS 100 is a self-directed 100-day project starting on January 1, 2026, designed to build a personal research operating system for an AI-native investor. Inspired by rapid daily building, the project focuses on creating a flexible and evolving research environment for work on startups, venture capital, and innovation policy.

This repository contains only public-facing artifacts. Exploratory work, raw data, and intermediate experiments are maintained separately in a private repository.

▪️ Purpose:

Generative AI has dramatically lowered the cost of producing research outputs. As a result, the primary bottleneck in research has shifted from execution to iteration.

The purpose of Research OS 100 is to develop the habit of continuous experimentation by prioritizing speed over polish and improvement over completion. Rather than optimizing for finished products, the project emphasizes documenting the thinking process itself, so that methods can be revisited, refined, or replaced as tools and models evolve.

▪️ Project Scope:

The project covers practical research utilities, including the following areas.

・Literature discovery and review

・Data collection and lightweight processing

・Visualization and exploratory analysis

・Simple modeling and evaluation workflows

Large language models may be used for tasks such as summarization, extraction, scoring, or workflow support. Each day contributes one small but functional component toward a reusable and extensible research system.

▪️ Non-goals:

Research OS 100 does not aim to produce polished academic papers, investment recommendations, or production-grade systems.

Outputs are intentionally lightweight, experimental, and subject to frequent revision. The focus is on learning velocity and system design rather than completeness or optimality.

▪️ Repository Structure:

notebooks/
・Public Jupyter notebooks demonstrating daily tools, experiments, and prototypes.

src/
・Reusable scripts, helpers, and shared utilities.

docs/
・Figures, references, and supporting materials.

Private drafts, intermediate experiments, and raw datasets are managed separately in a private repository.

▪️ Progress Summary (as of Day 051):

Day 001 focused on designing and implementing the initial arXiv ingestion workflow, enabling daily monitoring of research related to startups, venture capital, and innovation policy.

Day 002 focused on building a keyword-based paper retrieval pipeline using the OpenAlex API, including parameterized search, cursor-based pagination, normalization into tabular form, and exportable artifacts (CSV / metadata) for downstream analysis.

Day 003 focused on constructing a citation network crawler using the OpenAlex API, starting from a focal paper and expanding via cited-by relationships with BFS. The workflow produces graph-ready node and edge data, persisted as reusable artifacts for visualization and subsequent network analysis.

Day 004 focused on converting a local PDF paper into a set of presentation-ready slide images by generating structured summaries and full slide visuals via LLMs and Gemini, enabling rapid grasp of a paper’s structure before detailed reading.

Day 005 focused on building a startup portfolio intelligence pipeline that scrapes public portfolio pages, enriches company-level data with metadata and textual signals, normalizes features, and produces reusable portfolio- and company-level artifacts for comparison, diagnostics, and downstream analysis.

Day 006 focused on building a RAG-based research question generation workflow that integrates Google Drive PDFs and prior Notion RQ memos. The notebook enables an end-to-end loop from idea input and evidence retrieval to structured RQ generation, refinement, and persistence into a Notion research database.

Day 007 focused on building a parametric VC portfolio MOIC estimation workflow. The notebook combines round-level investment assumptions with CSE-based investment and exit evidence, applies fast snippet-based exit tagging and confidence heuristics, and aggregates deal-level outcomes into scenario-comparable portfolio MOIC estimates.

Day 008 focused on reconstructing structured startup profiles from public web sources using Google CSE and LLMs. The workflow separates search by attribute, enforces robots-aware scraping, and integrates field-level summaries with evidence URLs and confidence scores into reusable, inspection-ready startup profile artifacts.

Day 009 focused on building an LP candidate pre-research workflow using Google CSE and LLMs. The notebook collects public signals, performs facet-wise extraction with evidence and confidence, flags data gaps, and generates hypothesis-based LP profiles and report-ready outputs for downstream fundraising and analysis.

Day 010 focused on automating pre-meeting research for initial startup meetings using Google CSE and LLMs. The workflow integrates entity identification, company and people analysis, business, market, competition, funding, and recent changes into hypothesis-driven insights, meeting questions, watchouts, and a one-page, evidence-linked briefing artifact.

Day 011 focused on building a cross-market IPO revenue analysis pipeline using SEC and EDINET data. The workflow extracts, normalizes, and merges US and JP prospectus revenues, implements rigorous QA to detect and exclude outliers, and enables fair comparative analysis of revenue levels and growth dynamics.

Day 012 focused on designing and prototyping a lightweight, meeting-oriented person deep-dive notebook. Using Google CSE and LLMs, it synthesizes public signals into hypothesis-driven, evidence-linked individual briefs with explicit uncertainty, optimized for pre-meeting preparation.

Day 013 focused on building a reusable “Startup Databook” notebook that systematically discovers startups by country and period using public sources. The workflow emphasizes robots-aware collection, LLM-based classification and structuring, English-first outputs, deduplication, and auditable, reproducible exports for downstream analysis. 

Day 014 focused on building a forward-looking benchmark notebook using historical TSE Growth IPO data. By normalizing pre-IPO revenues and aligning relative years, it enables rapid sanity checks of an investment candidate’s revenue scale and trajectory against market percentiles.

Day 015 focused on building an end-to-end PDF-to-knowledge workflow. The notebook parses academic PDFs into structured representations, persists research summaries in Notion, and generates slide-ready visual one-pagers via Gemini, enabling fast, reusable paper understanding and presentation.

Day 016 focused on building a weekly news ingestion pipeline using NewsAPI for investment-relevant technology trends.The notebook defines a fixed 15-keyword taxonomy, handles free-tier constraints safely, normalizes and deduplicates articles, and persists week-partitioned raw datasets to support reproducible, long-term trend monitoring.

Day 017 focused on building a weekly aggregation and visualization layer on top of the NewsAPI pipeline. The notebook computes normalized keyword and category-level metrics, generates robust time-series plots and heatmaps, and produces a weekly insight table designed for stable, automated trend monitoring.

Day 018 focused on designing and implementing a weekly Japan corporate registry ingestion and aggregation pipeline using the National Tax Agency’s Web-API. The workflow computes municipality-level net changes from new registrations, closures, and relocations, producing reusable, geospatial-ready weekly datasets.

Day 019 focused on building a robust PDF ingestion and knowledge persistence pipeline. The notebook automatically fetches open-access papers from the seed corpus, handles retries and deduplication, uploads PDFs to Google Drive, and upserts structured literature records into a Notion database for downstream research workflows.

Day 020 focused on implementing a backward-fill workflow for high-quality literature expansion. Starting from highly cited, RQ-aligned core papers, the notebook traverses citation links backward to identify foundational works, prioritizes candidates by citation signals and RQ fit, and persists results to Drive and Notion for systematic human review.

Day 021 focused on transforming the accumulated literature corpus into a directed citation graph. The workflow unified paper identifiers, constructed node–edge tables, computed centrality and communities, and generated visualization-ready exports and reading-order maps to turn citation structure into actionable research navigation.

Day 022 focused on operationalizing research gap mining as a persistent, cluster-wise workflow. Citation-network-derived gaps were normalized, prioritized, and continuously tracked in Notion with diff-based updates, enabling gaps to be treated as evolving research assets rather than one-off analytical outputs.

Day 023 focused on building a fully incremental, state-aware daily paper scanner. The pipeline retrieves only newly published papers since the last run, classifies relevance against existing Research Questions with rationales, and robustly upserts only relevant results into Notion using a schema-flexible design.

Day 024 focused on systematically updating Research Questions from new evidence and human intent. The notebook synthesizes recent literature updates and meeting-driven change requests, proposes versioned RQ edits or new RQs with evidence links, and writes draft proposals back to Notion for structured human review.

Day 025 focused on demonstrating end-to-end research automation orchestration. The pipeline sequentially executes prior notebooks from corpus updates to RQ revisions, compiles daily and weekly summary reports, records run manifests, and provides a minimal CLI and CI-ready skeleton to validate researchOS as an integrated, operable system.

Day 026 focused on building a structure-driven, skill-aware notebook generator. Using a two-phase approach, the system treats structure as the single source of truth and injects accumulated “human correction” Skills into Claude prompts, enabling reproducible, incrementally improvable Jupyter notebook generation.

Day 027 focused on ingesting high-value human corrections from LLM interaction logs into a persistent Skill store. The pipeline extracts reasoning and structural fixes, normalizes them into reusable Skill records, deduplicates via JSONL, and synchronizes them to a schema-adaptive Notion database.

Day 028 focused on establishing a stable execution bootstrap and state layer for researchOS. The notebook centralizes environment loading, configuration normalization, run context initialization, persistent JSON-based state, and logging, enabling downstream pipelines to run deterministically, idempotently, and safely.

Day 029 focused on building a robust, SDK-independent Notion I/O layer for researchOS. The notebook implements REST-based clients, schema- and data-source–aware introspection, error classification, retries, and idempotent CRUD wrappers, enabling reliable daily operation across all research and monitoring databases.

Day 030 focused on implementing a safe, idempotent daily paper scanner. The pipeline incrementally fetches recent papers from arXiv and Semantic Scholar, applies multi-stage deduplication, generates short summaries, and ingests only new, relevant items into the Notion Papers inbox with full run-state tracking.

Day 031 focused on processing a PDF inbox into a fully tracked Papers pipeline. It extracts and repairs metadata, deduplicates against Notion, creates INBOX records, generates one-slide visual summaries, uploads PDFs and artifacts to Google Drive, writes links back to Notion, and safely moves files with partial-failure tolerance.

Day 032 focused on operationalizing daily VC monitoring as a robust, idempotent event pipeline. It loads VC targets from Notion, fetches RSS/HTML (optionally NewsAPI), normalizes date-safe event candidates, applies freshness/window filters, deduplicates in-memory and against Notion, writes only new Events via 029 wrappers, and outputs clean Markdown/Slack run summaries.

Day 033 focused on launching a startup-domain daily monitoring pipeline. It queries startup targets from Notion, fetches recent updates, normalizes items into tagged startup events, extracts high-value correction signals, and writes results back to Notion with robust partial-failure handling, producing an execution summary for daily operations.

Day 034 focused on extending the daily monitoring framework to the POLICY domain. By cloning the VC pipeline with a strict domain swap, the notebook fetches RSS/HTML/NewsAPI updates, normalizes date-safe policy events, applies consistent deduplication, writes new Events to Notion via shared wrappers, updates state, and produces standardized Markdown and Slack run summaries.

Day 035 focused on extending the daily monitoring stack to the PEOPLE domain via a strict domain swap. Cloned from the policy monitor, it fetches RSS/HTML/NewsAPI updates for tracked individuals, enriches and deduplicates items with rerun-safe keys, writes only new events to Notion via 029 wrappers, updates people-scoped state, and outputs Markdown/Slack summaries.

Day 036 focused on orchestrating the full daily researchOS pipeline into a single, resilient run. The orchestrator executes config, scanning, PDF processing, and multiple monitoring modules sequentially, captures per-module metrics and errors, tolerates partial failures, and produces consolidated daily summaries in Notion-ready Markdown, standard Markdown, and Slack formats.

Day 037 focused on automating minimal-diff cloning of daily monitoring notebooks. Using a structure-first, two-phase workflow, the system generates a deterministic skeleton from Cell 00, then plans keep/edit decisions and applies per-cell patches one at a time, optionally guided by a retrieved Skill Pack to preserve invariants and reduce maintenance drift.

Day 038 focused on generating meeting-ready, fully editable Japanese PPTX decks via an LLM-driven, human-in-the-loop workflow. It collects meeting context, uses OpenAI to produce an outline and slide JSON specs, generates cover/right-panel images with Gemini, and assembles a strict two-column PPTX with editable text boxes and replaceable visuals.

Day 039 focused on extending the structure-driven notebook generator into an iterative build-test-repair loop. After generating a skeleton from Cell 00, the system fills cells one-by-one, executes a deterministic prefix to capture outputs, and automatically repairs failures by generating OpenAI repair prompts and applying Claude JSON patches—advancing only when the prefix succeeds.

Day 040 focused on building a weekly papers review workflow to reduce noise and produce a curated reading list. It queries the last 7 days of Notion-ingested papers via the data_sources API, ranks them by importance and RQ relevance, applies READ/KEEP/SKIP decisions with optional human overrides, and exports weekly artifacts and summaries for worldview updates.

Day 041 focused on producing a weekly events digest from Notion Events. It queries the past week in JST via the data_sources API, normalizes records, deduplicates and filters noise, clusters events into interpretable themes, ranks themes by impact, and renders a Markdown digest with exported JSON/MD artifacts, optionally upserting an append-only weekly digest page in Notion.

Day 042 focused on updating weekly RQ understanding as review-ready, evidence-linked proposals. The notebook screens this week’s papers and events for RQ relevance, generates overwrite-ready updates per change category (Rationale/Approach/Gap/etc.), writes one Notion page per RQ×Category with explicit evidence relations and confidence, and links all updates back to the latest Weekly Digest.

Day 043 focused on establishing a weekly review and tuning loop for all monitoring targets. The notebook aggregates recent Events to evaluate signal vs noise per target, proposes concrete configuration changes (priority, cadence, status, keywords, sources), writes reviewable proposal pages to Notion, and exports charts, CSVs, and a text report for the weekly review packet.

Day 044 focused on generating PowerPoint-ready decks via an SVG-first workflow. The pipeline converts meeting agendas into slide plans, uses Gemini to create strict PowerPoint-compatible SVGs with real text, sanitizes and validates outputs, and delivers per-slide SVG assets plus a PPTX assembled either via SVG→PNG (python-pptx) or direct SVG insertion (AppleScript) for editability.

Day 045 and day 046 focused on building a task-driven notebook repair control plane. It pulls real tasks from Notion, resolves data_source IDs once, creates sandbox notebook copies, enqueues deterministic scaffold/patch/verify steps into a StateStore, and runs a one-step loop that records execution evidence and artifacts for safe, auditable iteration.

Day 047 focused on extracting the weekly papers review pipeline from notebooks into a reusable, CLI-driven module architecture. The logic for fetching, normalizing, and exporting recent papers was migrated into src/, with a standalone script that produces structured JSON, Markdown summaries, and provenance metadata. This refactor established a clean separation between orchestration (notebooks) and reusable intelligence logic (src), enabling deterministic weekly runs and smoother agent-assisted development via Claude Code.

Day 048 focused on building a weekly event digestion engine that consolidates VC, Startup, Policy, and People events into a structured signal layer. The script normalizes raw event data, removes duplicates and noise, groups by theme, and surfaces high-signal movements for the week. Outputs include machine-readable JSON, human-readable summaries, and metadata, forming the second core input to the weekly intelligence cycle.

Day 049 focused on updating Research Question (RQ) status in a structured, evidence-linked format. The pipeline links weekly Papers and Events back to each RQ, summarizes newly surfaced insights, highlights unresolved gaps, and records confidence-aware updates. Each RQ produces a structured status record suitable for Notion persistence, enabling longitudinal tracking of understanding over time rather than treating research as static documentation.

Day 050 focused on formalizing a weekly monitoring-target review loop. The system evaluates each tracked target using transparent signal vs noise heuristics (event volume, confidence, recency, duplication), proposes concrete actions (keep, drop candidate, review), and suggests configuration adjustments such as priority, cadence, and keyword tuning. All decisions are rule-based, reproducible, and exportable to both JSON artifacts and Notion update records, turning subjective monitoring into a measurable weekly discipline.

Day 051 focused on expanding discovery beyond the current monitoring universe. A new discovery engine extracts emerging entities from weekly Papers and Events using heuristic entity detection (multi-token capitalized phrases, Japanese patterns, alias preservation), scores them by frequency, diversity, and growth, and generates structured candidate records with rationale and evidence snippets. Optional LLM post-processing refines classification while preserving reproducibility. The result is a weekly candidate list for expanding the monitoring graph, closing the loop between review and exploration.

▪️ Technical Environment:

The project is implemented primarily in Python using Jupyter Lab and standard scientific computing libraries. External APIs and large language models are integrated as needed, with an intentionally flexible technical stack to accommodate rapid changes in tools, models, and best practices.

▪️ Note on AI Usage:

Large language models are used as research aids, such as summarization, extraction, ideation, and workflow support, not as authoritative sources.

All interpretations, judgments, and conclusions remain human-driven, and any errors or omissions are the author’s responsibility.

▪️ Long-term Goal:

The long-term goal of Research OS 100 is to establish a reproducible, scalable, and continuously evolving research foundation at the intersection of startups, venture capital, and innovation policy.

Beyond the 100-day challenge, the system is intended to remain under active development, gradually enabling partial automation of the research pipeline while preserving transparency and human judgment.

▪️ Disclaimer:

This project is a personal, self-directed research initiative.
All views, analyses, and tools presented here are solely my own and do not represent the views of any current or past employer, organization, or affiliated institution.