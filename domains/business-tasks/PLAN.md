# Domain Plan: Business Tasks

A future domain instance using the TinyLLM framework, focused on structured business tasks in a typical US corporate environment.

**Status:** Planning only — not started. Come back to this after the React/TS domain is proven.

---

## Domain Assessment

| Dimension | Rating | Notes |
|---|---|---|
| Data availability | Moderate | Some public sources, heavily synthetic-dependent |
| Validator strength | Weak-moderate | SQL/formula checkers exist, writing quality is subjective |
| Correctness checkable | Partially | Structured outputs yes, prose quality no |
| Recommended tier | $1K (1B model) | Lean on RAG, not model knowledge |
| Hybrid value | Moderate | Fewer validators than code domain |

---

## Task Categories

### Tier 1: Structured and checkable (best fit for tiny LLM)

- **SQL queries** — "Show total revenue by quarter for 2025"
- **Excel/Sheets formulas** — "Cumulative YTD growth formula"
- **JIRA/ticket creation** — structured fields from a description
- **Status report generation** — bullet points → formatted report
- **Meeting agenda generation** — topic list → structured agenda
- **Data summary from numbers** — table → key takeaways paragraph
- **Email templates** — fill-in-the-blank for common patterns (follow-up, intro, scheduling)
- **Cron/scheduling expressions** — "every weekday at 9am" → cron syntax

### Tier 2: Semi-structured (moderate fit)

- Document summarization (needs source comparison validator)
- Project brief drafting (template-guided)
- Job description writing (pattern-heavy, good for LoRA)
- Slide outline generation (structured output)
- Rewrite for audience (formal ↔ casual)

### Tier 3: Open-ended writing (weak fit for tiny LLM)

- Persuasive emails
- Strategy memos
- Executive summaries requiring judgment
- General business advice

Focus the model on Tier 1 and 2. Use RAG + templates for Tier 3 or defer to a larger model.

---

## Data Sources

### Modern public sources

| Source | Size estimate | Quality | How to get |
|---|---|---|---|
| **GitLab public handbook** | Large | Very high — modern company ops, written like real internal docs | Git clone (public repo) |
| **Basecamp/37signals public writing** | Medium | Very high — Shape Up, internal comms style | Public web |
| **Public company handbooks** (Valve, Netflix culture deck, etc.) | Medium | High | Various public links |
| **GitHub Issues and PRs** (popular repos) | Massive | High — real project management writing | GitHub API |
| **Public Notion templates** | Medium | High — modern structured business docs | Notion template gallery |
| **Y Combinator public applications** | Small | High — modern pitch/business writing | Public examples |
| **Product Hunt launches** | Large | Medium-high — structured product descriptions | API/scrape |
| **Public OKR/goal examples** | Small | High | Various company blogs |
| **Stack Overflow** (SQL, Excel, Google Sheets, Airtable tags) | Large | Medium-high | Data dump |
| **Reddit** (r/consulting, r/projectmanagement, r/startups, r/ExcelTips) | Medium | Mixed | Pushshift dump |
| **Public RFPs and government contracts** | Large | Medium — formal but real | Government portals |
| **Open source project governance** (Apache, Linux Foundation meeting notes, decision records) | Medium | High | Public repos |
| **Modern business blogs** (First Round Review, Lenny's Newsletter public posts, etc.) | Medium | High | Public web |
| **Public retrospective/postmortem templates** | Small | High | Various repos and blogs |
| **Job descriptions** (modern, from company career pages) | Large | Medium | Common Crawl filter |

### Deliberately excluded

- **Enron emails** — 25 years old, different tools and communication style, not representative of modern workflows
- **SEC filings** — useful for formal writing but too specialized; limit to small percentage if used
- Internal corporate communications (private by nature)

### Key insight

The best sources are **public company handbooks and operational docs** (GitLab, Basecamp, etc.) — they read like real internal business writing because they ARE real internal business writing that happens to be public. This is the equivalent of picking high-quality repos for the React/TS domain.

### Synthetic data (critical for this domain)

Generate via Claude API (~$200-400):

| Task type | Volume | Example |
|---|---|---|
| SQL from natural language | 5,000 | "revenue by quarter" → SQL query |
| Excel formulas from description | 3,000 | "running average of column B" → formula |
| Bullet points → status report | 3,000 | bullets → formatted report |
| Email templates | 3,000 | scenario → professional email |
| JIRA ticket from description | 2,000 | bug report → structured ticket |
| Meeting agenda from topics | 2,000 | topic list → agenda |
| Document summary pairs | 2,000 | long text → summary |
| Rewrite for audience | 2,000 | casual → formal, formal → casual |

Total: ~22,000 synthetic examples

### Estimated training mix

```
30%  General business text (SEC filings, press releases, Wikipedia)
25%  Synthetic task pairs (instruction → structured output)
20%  Email/communication (Enron filtered + synthetic)
15%  Technical business (SQL, Excel, Stack Overflow)
10%  Templates and formatting examples
```

---

## Validators

### Strong (automated)

- **SQL validator** — parse and validate syntax, run EXPLAIN against schema
- **Excel formula parser** — check syntax validity
- **JSON/schema validator** — structured outputs match expected format
- **Template compliance** — does the output follow the required template structure
- **Grammar/spelling** — basic language quality (LanguageTool or similar)

### Weak (heuristic)

- **Professionalism scorer** — check for informal language, slang, emoji in business context
- **Length checker** — is the output appropriate length for the task type
- **Format checker** — does it have proper greeting/closing for emails, headers for reports
- **Completeness checker** — does it address all points in the prompt

### Not automatable

- Persuasiveness
- Strategic quality
- Political sensitivity
- Cultural appropriateness

---

## Memory Schema

```sql
-- User profile
users (user_id, role, department, writing_style_preference, common_tasks)

-- Task history
tasks (task_id, user_id, task_type, input, output, accepted, timestamp)

-- Templates
templates (template_id, task_type, template_text, usage_count)

-- Frequently referenced entities
entities (entity_id, name, type, context)
-- e.g., "Q3 2026", "Project Atlas", "Sarah Chen (VP Engineering)"
```

---

## Tools

- **SQL executor** — run queries against a sample/real database
- **Excel formula evaluator** — test formulas with sample data
- **Calendar tool** — check dates, calculate business days
- **Template engine** — fill structured templates
- **Word counter** — enforce length constraints
- **Grammar checker** — LanguageTool API

---

## Eval Sets

| Eval set | Size | Type |
|---|---|---|
| SQL generation | 100 | Automated (syntax + execution) |
| Excel formulas | 50 | Automated (formula parser) |
| Status report quality | 50 | Human-rated |
| Email appropriateness | 50 | Human-rated |
| JIRA ticket completeness | 50 | Semi-automated (field coverage) |
| Template compliance | 50 | Automated (schema match) |
| Summarization accuracy | 50 | Human-rated (against source) |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Training data is mostly formal/public, real business writing is different | Heavy synthetic data generation mimicking real tasks |
| Weak validators for prose quality | Focus on structured tasks, use templates as constraints |
| Model generates inappropriate content | Professionalism filter, DPO on appropriate/inappropriate pairs |
| Enron data has sensitive content | Filter aggressively, remove names/numbers |
| Model memorizes SEC filing boilerplate | Deduplicate, limit SEC data to 15% of mix |

---

## When to Build This

Prerequisites:
- [ ] React/TS domain is working end-to-end
- [ ] Domain framework (Part XVII) is implemented
- [ ] Config-driven domain setup is tested
- [ ] At least one cloud training run is complete

This domain would be a great test of the framework's generalizability — it's different enough from React/TS (natural language vs code, weak vs strong validators) to prove the framework works across domains.

---

## Estimated Effort

| Phase | Time |
|---|---|
| Data collection and filtering | 3-5 days |
| Synthetic data generation | 1-2 days |
| Validator setup | 1-2 days |
| Training (1B model, $1K cloud) | 2-3 days |
| Eval and iteration | 2-3 days |
| **Total** | **~2-3 weeks** |
