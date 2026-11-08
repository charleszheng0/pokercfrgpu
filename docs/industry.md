# Industry fit: what employers ask for, and what to publish

Research date: 2026-11-08. Sources are mostly job-board copies of company postings. Many are closed or old. "Unverified" means I did not find a usable posting. Check each company's careers page before you rely on a claim.

## 1. What the roles ask for

**Google (new grad / intern SWE).**
- Current enrollment in a CS-type degree.
- Code in general-purpose languages (C, C++, Java, JavaScript, Python). One US BS posting asks for two or more.
- Data structures and algorithms.
- Sources: [Google SWE Intern, Summer 2027 (board copy)](https://jobs.anitab.org/companies/google-24698/jobs/82819261-software-engineering-intern-summer-2027), [Google Careers listing](https://google.com/about/careers/applications/jobs/results/85430979253936838-software-engineering-intern-summer-2027). The official page did not load text for me. Preferred qualifications: unverified.
- Meaning: Google screens on general skill. A strong C++ project is a bonus, not a rule.

**Meta (University Grad / intern).**
- A degree in progress and coding experience in any "industry-standard" language. C++ is one of several examples.
- Preferred: internships, contests, or publications as proof of skill.
- Systems-flavored postings ask for C/C++ with modern concurrency.
- Sources: [Meta University Grad (board copy)](https://jointaro.com/jobs/meta/software-engineer-university-grad), [Meta Careers](https://metacareers.com/jobs/8187200048015588), [Meta systems role (board copy)](https://beringstraits.dejobs.org/redmond-wa/software-engineer-sw-specialist-systems/DEE409374B20436997577E4A633EDDC6/job/).

**Apple, Microsoft, Amazon.** Unverified. My searches found no usable posting. Do not assume anything.

**NVIDIA (new grad; closest match to the project).**
- Strong C/C++ and CUDA.
- Performance profiling and optimization.
- Parallel programming, HPC, linear algebra.
- Git, CMake or Bazel, CI, code coverage, static analysis.
- Knowledge of CPU and GPU architecture is a "stand out" item.
- Sources: [DL Libraries, New Grad 2026](https://jobs.anitab.org/companies/nvidia/jobs/89969070-software-engineer-deep-learning-libraries-new-college-graduate-2026) (read in full; posting now closed), [DevTech AI, New Grad 2026](https://jobs.anitab.org/companies/nvidia/jobs/62870446-developer-technology-engineer-ai-new-college-grad-2026), [Compiler Front-End, New Grad 2026](https://jobs.anitab.org/companies/nvidia/jobs/68139378-compiler-engineer-compute-front-end-new-college-grad-2026), [TensorRT, New Grad 2025](https://jobs.anitab.org/companies/nvidia/jobs/57405377-software-engineer-tensorrt-specialized-platforms-new-college-grad-2025). The last three are from search summaries only.

**Jump Trading (intern).**
- C++ on Linux is preferred.
- Required: data structures and threading models.
- Interest in hardware, networking, and OS to build very fast software.
- Nice to have: coursework in algorithms, OS, networks, compilers.
- No finance background needed.
- Source: [Campus C++ Intern (removed Nov 2025, still a good template)](https://builtin.com/job/campus-c-software-engineer-intern/6812219).

**Hudson River Trading.**
- Intern: Python, C, or C++. Clean, efficient code. Strong problem solving and communication. No trading knowledge needed. Source: [HRT SWE Intern Summer 2026 (removed)](https://www.builtinboston.com/job/software-engineering-internship-summer-2026/6747111).
- Experienced roles: deep C++, OS internals, CPU architecture, networking. Source: [HRT Low Level C++](https://builtin.com/job/experienced-software-engineer-low-level-c/7686799). This is the long-term bar, not the intern bar.

**Citadel / Citadel Securities.**
- C++ is the top listed skill. Entry-level postings ask for strong C++ and passion for technology.
- Sources: [C++ Software Engineer, Citadel Securities](https://www.citadelsecurities.com/careers/details/c-software-engineer-2/) (page blocked me; summary from [board copy](https://builtin.com/job/c-software-engineer/6900111)), [University Grad (board copy)](https://builtin.com/job/software-engineer-2025-university-graduate-us/3384902). Pay figures on boards are unverified.

**Jane Street.**
- Looks for curiosity, teamwork, and how you think and learn.
- Does not require finance or OCaml experience.
- Source: [Jane Street internships](https://www.janestreet.com/internships), [SWE posting](https://job-boards.greenhouse.io/janestreet/jobs/8062722002) (not read in full).
- Meaning: a clear design write-up matters more than a language match.

**Two Sigma.** Only an AI research posting found. It wants Python and a deep learning framework. C++ is a plus. Source: [Two Sigma AI Research Scientist](https://jobs.anitab.org/companies/two-sigma-2/jobs/65255414-ai-research-scientist-campus-full-time). SWE intern requirements: unverified.

**DeepMind, Anthropic, OpenAI (reference).**
- DeepMind student/intern roles: general programming, Python, JAX/PyTorch. C++ is one option. Source: [DeepMind postings](https://job-boards.greenhouse.io/deepmind/jobs/7606920).
- Anthropic GPU performance roles ask for CUDA, Triton, custom kernels, and multi-node work. They look senior, not new grad. Source: [Anthropic Performance Engineer, GPU](https://job-boards.greenhouse.io/anthropic/jobs/4926227008) (page showed only an index; details from [board copy](https://builtin.com/job/performance-engineer-gpu/7198083)).
- OpenAI: only an inference GPU role found (HIP/CUDA/Triton kernels). Seniority unverified.

**Pattern.** Big-tech SWE roles test general skill. NVIDIA and trading firms test performance, C++, concurrency, and hardware sense. Your project fits the second group best. Show numbers, not claims.

## 2. Ranked metrics and optimizations

Effort: S = 1-2 days, M = 3-5 days, L = 1+ week. "X" is a number you measure.

| # | Item | Values it | Effort | Resume bullet template |
|---|---|---|---|---|
| 1 | Reproducible benchmark harness: fixed seeds, pinned threads, warm-up, median and 95% CI over N runs, JSON output, one script | All groups (NVIDIA, HRT, Jump, Citadel, Google) | S | "Built a benchmark harness (median, 95% CI over N runs) that tracks solver speed per commit." |
| 2 | Throughput: iterations/s and hand-pair evaluations/s, per thread and total | NVIDIA, HRT, Jump, Citadel | S | "Raised DCFR throughput from X to Y M hand-pairs/s (Z x) on an Apple M2." |
| 3 | Exploitability-vs-time curves, with a reference solve | Quant/ML groups, DeepMind, Jane Street | S | "Reached X% of pot exploitability in Y s; plotted convergence for N spots." |
| 4 | Profiler-driven optimizations: perf/Instruments counters, cache misses, one before/after story per fix | NVIDIA, HRT, Jump, Citadel | M | "Cut cache misses by X% and time by Y% after profiling the showdown sweep." |
| 5 | SIMD/NEON for the showdown sweep and regret matching | NVIDIA, HRT, Jump, Citadel | M | "Vectorized regret matching with NEON: X x speedup, bit-exact within 1e-6." |
| 6 | Strong and weak scaling across threads, with efficiency and Amdahl fit | NVIDIA, Jump, Citadel, Meta infra | M | "Reached X% parallel efficiency on 8 threads; found the serial limit at Y%." |
| 7 | Memory per info-set (bytes/node), with the 16-bit compression and isomorphism gains | NVIDIA, HRT, Jump, ML infra | S | "Cut memory from X to Y bytes per info-set (Z x); flop solve fits in W GB." |
| 8 | Correctness: toy games vs theory, vs a reference solver, property tests, ASan/TSan/UBSan in CI, range-parser fuzzing | All; NVIDIA lists coverage and static analysis | M | "Added CI with ASan/TSan/UBSan and fuzzing; found X bugs; N tests, X% coverage." |
| 9 | CUDA kernels with roofline and occupancy analysis (Nsight) | NVIDIA, Anthropic/OpenAI (reference), ML infra | L | "Wrote CUDA kernels reaching X% of memory bandwidth peak; Y x faster than CPU." |
| 10 | Server latency p50/p99 under load, plus DB size and query latency | Google, Meta, Amazon-type backend roles (their requirements unverified) | S | "Served solved spots at p50 X ms / p99 Y ms; SQLite with N spots queries in Z ms." |
| 11 | Design docs and ADRs: one-page decisions with trade-offs and measured results | Jane Street, Google, all code review | S | "Documented N design decisions (ADRs) with measured trade-offs." |
| 12 | Regression guard: CI fails if throughput drops more than X% | Trading firms, NVIDIA | S | "Added a perf-regression gate in CI that blocks a slowdown above X%." |

Ranking logic: items 1-3 are cheap and make every other number believable. Items 4-7 match the NVIDIA and trading-firm text most closely. Item 9 is the biggest effort and the biggest NVIDIA signal. Do not start it before items 1-7 are solid.

## 3. Recommended 3-week order

**Week 1: measure.**
- Day 1-2: harness (1) and throughput counters (2).
- Day 3: exploitability-vs-time (3) and memory per info-set (7).
- Day 4-5: baseline profile of the hot loop (4). Save raw numbers and a graph.
- Output: a `bench/` script, a results table in the README, a first plot.

**Week 2: optimize and prove.**
- Day 1-3: SIMD/NEON in the showdown sweep and regret matching (5). Keep a before/after table.
- Day 4: strong and weak scaling (6).
- Day 5: sanitizers and parser fuzzing in CI (8), plus the perf-regression gate (12).
- Output: speedups with CIs, a CI badge, a bug list.

**Week 3: reach and polish.**
- Day 1-3: start CUDA (9) on one kernel only (the showdown sweep). Report roofline and occupancy even if the gain is small. Say what you learned.
- Day 4: server latency and DB size/query latency (10).
- Day 5: ADRs (11), final README with 3 headline numbers, and 5 resume bullets.
- If CUDA slips, ship the profile and a clear plan. Honest partial work beats a claim with no data.

## Notes
- No guarantee any metric gets an interview. Use only numbers you measured.
- Verified in full: the NVIDIA DL Libraries, Jump Campus C++, and HRT intern postings. Everything else is from search summaries or board copies. Apple, Microsoft, Amazon, Two Sigma SWE, and OpenAI new-grad requirements are unverified.
