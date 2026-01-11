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

▪️ Progress Summary (as of Day 011):

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