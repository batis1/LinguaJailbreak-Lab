# AI Agent Security Research Ideas - 2026-06-04

Scope: mobile and web AI-agent attacks that can be evaluated in Colab Pro with an A100 GPU and published as a GitHub project plus paper.

Status note: Chrome automation cannot open `chrome://bookmarks`, so the TrustAI bookmark folder itself was not inspected. If the bookmarked pages are opened as normal Chrome tabs, they can be read in a later pass. The ideas below are grounded in logged-in X search signals from followed accounts plus current public papers and reports.

## Executive Recommendation

Build **Mobile-Web AgentTrapBench**: a benchmark and guardrail testbed for indirect prompt injection against visual web and mobile agents.

The project should generate realistic but sanitized attack carriers:

- Web pages with hidden or visually suppressed instruction-like content.
- Mobile screenshots with user-generated content regions such as comments, ads, messages, and notifications.
- Agent tasks with mock sensitive actions such as `send_message`, `approve_ad`, `purchase_item`, `delete_record`, and `exfiltrate_mock_secret`.
- Defense variants that compare visible screenshot content, DOM/accessibility text, task intent, tool arguments, and post-action policy checks.

Why this is the strongest direction:

- It directly combines mobile and web.
- It is current: 2026 work shows real-world web indirect prompt injection, new mobile GUI-agent attacks, and dynamic agent-defense gaps.
- It is feasible on one Colab A100 using open models such as Qwen2.5-VL-7B-Instruct.
- It can produce measurable results: attack success rate, benign task success, over-block rate, pass-all-k stability, and guard latency.
- It is publishable as a benchmark plus defense, even if the defense is simple, because the threat model is fresh and practical.

## Evidence Signals

### Logged-in X search signals

Searches over people the user follows surfaced these themes:

- Web-based indirect prompt injection observed in the wild.
- Zero-click prompt injection and runtime defenses.
- Agent-native risks missed by classic scanners: hidden instructions, tool poisoning, trigger abuse, excessive agency, and skill behavior mismatch.
- AI-agent skills as a supply-chain attack vector.
- MCP/tool poisoning and malicious tool descriptors.
- Autoresearch loops for jailbreak and prompt-injection discovery.
- "Agent detonation chamber" thinking: test what text causes an agent to do, not only whether text looks malicious.
- Agent-evaluation stability: pass-all-k rather than one lucky safe run.

Useful X links captured:

- https://x.com/rryssf/status/2061351794177950078
- https://x.com/rryssf/status/2061001006201344065
- https://x.com/tom_doerr/status/2059378639750218161
- https://x.com/tom_doerr/status/2058758398854795494
- https://x.com/AlexFinn/status/2057988300926075346
- https://x.com/akshay_pachaar/status/2046151867177308181
- https://x.com/mbrg0/status/2057837820815811064

### Current literature and reports

- Unit 42, "Fooling AI Agents: Web-Based Indirect Prompt Injection Observed in the Wild" - March 3, 2026. Key point: web IDPI is no longer only theoretical; telemetry found real-world attacker intents and 22 payload engineering techniques. https://unit42.paloaltonetworks.com/ai-agent-prompt-injection/?_wpnonce=f273666aa8&lg=en&pdf=print
- Google DeepMind, "AI Agent Traps" - March/April 2026. Key point: agent traps target perception, reasoning, memory, action, multi-agent dynamics, and human oversight. https://www.rivista.ai/wp-content/uploads/2026/04/ssrn-6372438.pdf
- MIRAGE, "Context-Aware Prompt Injection against Mobile GUI Agents via User-Generated Content" - May 27, 2026. Key point: benign mobile screenshots can be converted into realistic prompt-injection samples by placing attacker-controlled text in user-generated regions. https://arxiv.org/abs/2605.28116
- "Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels" - revised November 2025. Key point: mobile agents are exploitable through ads, embedded webviews, notifications, malware-install workflows, and cross-app data exfiltration paths. https://arxiv.org/abs/2510.27140
- MobileSafetyBench, AAAI 2026. Key point: Android-emulator mobile agents need benchmarked safety tests including indirect prompt injection. https://ojs.aaai.org/index.php/AAAI/article/view/41090
- WebAgentGuard - April 14, 2026. Key point: web VLM agents are vulnerable to prompt injection in HTML and screenshots; a separate guard model can detect attacks. https://arxiv.org/abs/2604.12284
- AgentVisor - April 27, 2026. Key point: semantic privilege separation and tool-call interception can sharply reduce ASR while preserving utility. https://arxiv.org/abs/2604.24118
- AgentDyn - revised May 7, 2026. Key point: existing agent-security benchmarks are too static; real-world defenses fail or over-defend under dynamic open-ended tasks. https://arxiv.org/abs/2602.03117
- SafeSearch - revised June 3, 2026, accepted ICML 2026. Key point: search agents are vulnerable to unreliable or malicious web results; highest ASR reached 90.5 percent in one setting. https://arxiv.org/abs/2509.23694
- IterInject - May 23, 2026. Key point: feedback-guided iterative optimization substantially improves indirect prompt-injection attacks on AgentDojo, InjectAgent, and Claude Code targets. https://arxiv.org/abs/2605.24659
- MCPTox - August 2025. Key point: tool poisoning on 45 real MCP servers and 353 tools shows widespread vulnerability, with one tested model reaching 72.8 percent ASR. https://arxiv.org/abs/2508.14925
- OWASP MCP Top 10. Key point: MCP introduces risks around token exposure, privilege scope creep, contextual prompt injection, and context over-sharing. https://owasp.org/www-project-mcp-top-10/

## Ranked Research Ideas

### 1. Mobile-Web AgentTrapBench

Research question: Do visual agents fail differently when the same attacker intent is delivered through DOM text, accessibility labels, rendered web pixels, mobile screenshots, ads, notifications, or embedded webviews?

Experiment:

- Create 200-500 benign web/mobile tasks.
- Mutate each task with sanitized trap carriers across web and mobile channels.
- Evaluate Qwen2.5-VL-7B-Instruct first; optionally add Gemma 3, Llama 4 vision, or API models later.
- Use mock tools and safe labels only.
- Report attack success rate, benign success rate, over-block rate, pass-all-k, and latency.

Paper claim:

- A unified benchmark reveals cross-channel failure modes not captured by web-only or mobile-only benchmarks.
- A lightweight action-boundary guard improves safety without killing utility.

Expected GitHub deliverables:

- Dataset generator.
- Colab notebook.
- Evaluation harness.
- Guard baseline.
- Result tables and plots.

Risk: medium. The dataset and harness are work, but the first version is feasible in Colab.

### 2. Dynamic Cloaking for AI Web Agents

Research question: Can agent-specific page rendering or DOM mutation bypass defenses that only inspect static HTML or visible screenshot text?

Experiment:

- Generate pages where visible content is benign.
- Inject sanitized trap text only after agent-like events, delayed rendering, accessibility traversal, or DOM-state changes.
- Compare static scanner, screenshot OCR scanner, DOM scanner, and action-boundary guard.

Paper claim:

- Defenses need multi-view provenance checks; a single view of a page is not enough.

Risk: medium-high because browser automation and agent detection can get messy. Strong novelty if implemented well.

### 3. Agent Detonation Chamber

Research question: Is action-outcome testing better than text classification for detecting dangerous external content?

Experiment:

- Put untrusted web/mobile content into a sandboxed agent environment.
- Observe proposed tool calls and arguments.
- Score whether the action violates the user task, leaks mock secrets, or changes mock state.
- Compare against LLM-as-judge and keyword/pattern scanners.

Paper claim:

- Runtime action causality is a stronger security signal than asking whether the text "looks malicious."

Risk: low-medium. Very practical and useful as a GitHub project.

### 4. Mobile User-Generated Content Injection

Research question: Are mobile GUI agents more vulnerable when prompt-like instructions appear in realistic user-controlled regions rather than obvious overlays?

Experiment:

- Start with benign screenshots from open Android UI datasets or synthetic app screens.
- Localize controllable regions: comments, chat bubbles, product reviews, ads, notification banners.
- Insert sanitized instruction-like text matching the native UI style.
- Test VLM task completion and unsafe action rate.

Paper claim:

- Visual realism is not enough to defend; the agent must know which regions are trusted UI versus untrusted content.

Risk: medium. Closest to MIRAGE, so novelty needs cross-channel comparison or a better defense.

### 5. Feedback-Guided Indirect Injection Optimization

Research question: Can an IterInject-style optimizer discover stronger web/mobile trap variants under a fixed A100 budget?

Experiment:

- Start with 20 sanitized seed attacks.
- Use a diagnoser to label failures: ignored, detected, wrong action, partial action, successful action.
- Let a helper LLM rewrite payload styles and placement strategies without generating harmful real-world instructions.
- Evaluate on held-out tasks and pages.

Paper claim:

- Security benchmarks should include adaptive attack generation, not only static payload lists.

Risk: high. Potentially strong results, but more moving parts.

### 6. MCP and Skill Descriptor Poisoning

Research question: Do agents follow malicious natural-language descriptors in tools/skills more than they follow direct untrusted page text?

Experiment:

- Create mock MCP tools with clean descriptors, poisoned descriptors, rug-pull descriptor changes, and shadowing descriptors.
- Evaluate tool selection and tool-argument safety.
- Add integrity checks and descriptor vetting.

Paper claim:

- Tool metadata is executable security context, not passive documentation.

Risk: low-medium. Strong GitHub tooling angle, less aligned with "mobile/web" unless combined with web/mobile agent tools.

### 7. Pass-all-k Agent Security Metric

Research question: How many agent defenses look safe under pass@k but fail under pass-all-k or repeated-run stability?

Experiment:

- Run each task 5-20 times with sampling.
- Compare at-least-one-safe, majority-safe, and all-safe metrics.
- Evaluate defenses under random UI ordering, visual noise, and paraphrased trap carriers.

Paper claim:

- Safety needs reliability under repeated runs, not one successful trace.

Risk: low. Good secondary contribution for Idea 1.

## Recommended First Colab MVP

Title: **Mobile-Web AgentTrapBench: A Cross-Channel Benchmark for Indirect Prompt Injection in Visual AI Agents**

Model:

- Start with `Qwen/Qwen2.5-VL-7B-Instruct` because it is open, runs on A100, and supports visual agent / phone-use style evaluations.

Minimum dataset:

- 40 web pages: 20 benign, 20 with sanitized hidden trap carriers.
- 40 mobile screenshots: 20 benign, 20 with sanitized user-generated trap carriers.
- 5 mock tool actions: `approve_ad`, `send_message`, `open_url`, `purchase_item`, `delete_record`.
- 4 attack intents: task diversion, mock secret leakage, unsafe approval, destructive mock action.

Metrics:

- `benign_task_success`
- `attack_success_rate`
- `unsafe_tool_call_rate`
- `guard_block_rate`
- `benign_overblock_rate`
- `pass_all_5`
- `latency_seconds`

Baselines:

- No defense.
- Reminder prompt.
- Text/DOM scanner.
- Screenshot OCR scanner.
- Action-boundary policy guard.
- Combined guard.

Expected first result:

- No-defense agents should show measurable attack success on at least one channel.
- Reminder-only defenses should be weak.
- Action-boundary guards should reduce unsafe tool calls but may over-block benign tasks.
- Cross-channel results should show that web DOM attacks and mobile screenshot attacks fail differently.

## Paper Angle

Proposed abstract:

> We introduce Mobile-Web AgentTrapBench, a benchmark for evaluating indirect prompt injection against visual AI agents across web and mobile channels. Unlike prior web-only or mobile-only evaluations, our benchmark compares semantically matched attacker intents delivered through DOM text, rendered pixels, accessibility-like labels, user-generated mobile content, notifications, and embedded webviews. Across open VLM agents, we measure unsafe action rates, benign utility, and repeated-run stability under sanitized mock tools. We further evaluate lightweight guardrails and show that action-boundary enforcement is more reliable than prompt reminders or static text scanning, while still leaving cross-channel over-defense challenges.

Target venues:

- AISec / CCS workshop.
- USENIX Security artifact track if results become strong.
- NDSS workshop.
- SaTML workshop.
- ICML/NeurIPS safety workshop if framed as benchmark/evaluation.

## Next Implementation Steps

Implemented MVP scaffold:

- `notebooks/mobile_web_agenttrapbench_mvp_colab.ipynb`
- `scripts/create_agenttrapbench_mvp_notebook.py`

The notebook starts with a `dry_run` backend for harness validation, then supports `qwen_vl` with `Qwen/Qwen2.5-VL-7B-Instruct` on Colab A100. It generates synthetic web/mobile trap assets, evaluates mock tool-call behavior, compares simple defenses, and saves CSV summaries plus a metrics plot. When `SAVE_TO_GOOGLE_DRIVE = True`, results are saved to `MyDrive/AgentTrapBench/runs/<run_id>/` and `MyDrive/AgentTrapBench/latest/`.

1. Create a Colab notebook that loads Qwen2.5-VL-7B-Instruct with 4-bit or bf16 on A100.
2. Generate a small synthetic dataset of HTML pages and mobile screenshots.
3. Add a mock tool-call protocol and JSON output schema.
4. Run no-defense and reminder-prompt baselines.
5. Add a simple action-boundary guard.
6. Save results as CSV and plots.
7. Push notebook and benchmark generator to GitHub.

## Open Blocker

The saved Chrome bookmark folder named TrustAI has not been directly inspected because Chrome automation blocks access to `chrome://bookmarks`. This can be resolved only by opening the TrustAI bookmarked pages as normal web tabs in Chrome, then asking Codex to inspect the open tabs.
