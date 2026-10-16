# ELIA AI

**ELIA is a CPU-first cognitive architecture that converts experience into tested predictors and executable procedures.**

This first vertical slice focuses on a smaller, falsifiable question: can explicit, source-linked hypotheses make prospective predictions that outperform a simple historical baseline? It is a local research prototype, not a general intelligence claim.

## Hardware-aware acceleration and local speed

ELIA is deliberately designed to run entirely on the local CPU and to expose the hardware it is using rather than pretending it can exceed the machine’s actual limits. The architecture now records a lightweight hardware profile, keeps cache bounds proportional to visible CPU resources, and exposes a local throughput benchmark so the system can be tuned against the actual host rather than a generic cloud configuration.

This is still bounded, auditable, and honest: it is a path toward faster local inference, not an unqualified claim of GPT-class capability or broad open-world intelligence.

### Measure the target VPS

Run `python -m elia.cognition.readiness --output readiness-report.json` on the Lightsail VPS from the project root (with `PYTHONPATH=src` if ELIA is not installed as a package). This reports visible CPU and RAM limits, checks for local speech/runtime tools without loading models, measures the existing cognition update/predict throughput, and times a bounded transitive-reasoning workload. The report is synthetic and does not download or invoke speech models; run it on the VPS to avoid confusing development-host measurements with deployment measurements.

### Optional local voice prototype

The dashboard has an opt-in microphone workflow backed by local Vosk speech recognition and a local Piper text-to-speech HTTP server. Audio is sent to this ELIA instance for transcription, transcripts remain in the page unless you explicitly store them elsewhere, and no cloud speech fallback or model download is performed. The recognized transcript is shown for review before ELIA receives it as a question. ELIA's existing bounded question interpreter only supports a small set of structured English patterns; this does not add general conversation or improve reasoning by itself.

1. Install ELIA with the optional Vosk runtime from the project root: `python -m pip install -e ".[voice]"`.
2. Download and unpack a Vosk speech model yourself, then set `ELIA_VOSK_MODEL` to its local directory. The Vosk runtime is optional; the separate model must also be obtained and reviewed for its license and suitability.
3. Install Piper and its HTTP extra in an isolated Python environment, obtain a compatible local voice model, then start the HTTP server with its bind/host explicitly set to `127.0.0.1` where supported. For the upstream Piper HTTP server, the documented form is `python -m piper.http_server -m <voice-model>`; check the installed version’s options and verify it listens only on loopback before use. Configure `ELIA_PIPER_URL` only if needed; it must be a loopback `http://.../synthesize` URL. Keep this service inaccessible from the public network.
4. Start ELIA and open its dashboard in a browser. Microphone access requires `localhost` or a secure HTTPS origin, and the browser will ask permission. Without the local models or services, the dashboard reports that voice is unavailable; typed evidence questions continue to work.

Check the license and redistribution terms for each runtime and each downloaded speech model/voice before deployment. Recognition is constrained to mono 16 kHz 16-bit PCM WAV up to 30 seconds and 5 MB; synthesis is bounded to 2,000 text characters. The status endpoint reports local Vosk configuration, but does not probe or load Piper. Measure recognition and synthesis latency and quality on the actual VPS before selecting model sizes or treating voice as production-ready.

## What works in this prototype

- Local HTTP API under `/api/v1` and a dashboard at `/`.
- SQLite storage for consented source episodes, observations, hypotheses, registered predictions, outcomes, and reproducible evaluation runs.
- A separate, opt-in CPU cognition prototype with a shared sparse core, bounded associative substrate, domain specialists, an uncertainty-aware router, and bounded replay.
- Cognitive training examples are source-linked and reconstruct the learned state from SQLite on startup. Removing a source episode also removes its cognitive training examples and rebuilds the learned state from what remains.
- A synthetic HTTP regression test exercises that deletion path behaviorally: after deleting one specialist's only source episode, that specialist becomes unknown/abstains while a second specialist backed by a retained episode still predicts. This checks the lifecycle wiring on a tiny controlled example; it is not a broad machine-unlearning guarantee.
- An experimental source-withholding counterfactual rebuilds a separate learner from all retained training examples except those linked to one selected source episode, then compares predictions. This makes source sensitivity inspectable without mutating live learned state. It combines a leave-one-group-out perturbation with ELIA's provenance-linked training lifecycle; it is not a claim of scientific novelty, causal attribution, or complete machine unlearning.
- Symbolic reasoning also supports source-withholding queries: temporarily omit all facts and rules sourced from one episode and recompute a query. This exposes whether a conclusion has an alternate proof from the remaining evidence, and shows the proof provenance that remains. It is a deterministic evidence-dependence check, not causal inference or verification of the source statements.
- A bounded source-resilience audit repeats that counterfactual once per evidence-bearing source episode (maximum 32 per request), identifying which single-source removals lose currently supported answer bindings and which do not. In the controlled test, two duplicate facts from separate episodes are individually redundant, but removing the only rule makes the conclusion unknown. This establishes behavior for that fixture only; it does not test multi-source combinations or prove truth, causality, or a novel algorithm.
- A bounded multi-source failure-set audit now tests all single-source removals and source pairs (up to 32 evidence-bearing episodes / 496 pairs, constrained by a caller-set evaluation budget and a 1,000,000 cumulative reasoning-work budget). A minimal pair failure set is reported only when either source alone can be withheld without losing that supported binding, but withholding both loses it. Outputs include exact pair states, lost bindings, alternative surviving bindings with proof-source observation IDs, evaluation counts, and an explicit incomplete flag/reason. The integration fixture finds the duplicate-fact pair as minimal while recognizing rule-dependent pairs are not minimal because the rule alone is pivotal. This bounded experiment does not check triples or larger combinations and does not establish scientific novelty, source truth, or causal responsibility.
- Evidence links distinguish support from challenge. An observation records what was reported; it does not certify that the report is objectively true.
- Binary predictions record their probability and a historical base-rate baseline at registration time. The expected outcome is inherited from the hypothesis so scores are comparable. An exact context cannot be predicted twice for the same hypothesis. Scoring fields cannot be edited afterward.
- Resolved predictions are scored with Brier scores. Hypotheses are promoted only after at least 10 distinct resolved predictions and a lower mean Brier score than the stored baseline; otherwise they remain candidates or are marked challenged.
- The on-demand CPU self-evaluation cycle re-scores recorded outcomes, detects hypotheses with both supporting and challenging observations, records TRACE status changes and aggregate predictor-vs-baseline Brier scores, and persists every cycle for inspection. It does not autonomously train on raw conversation data.
- The opt-in deliberation cycle selects one next research step from stored evidence using a small, explicit priority policy (review conflicting evidence, resolve pending outcomes, test a candidate, challenge a validated claim, or request consented evidence). It records the rationale and evidence counts. It only proposes; it does not create claims, register predictions, change learned weights, or execute actions.
- The symbolic reasoner (`src/elia/cognition/reasoning.py`) chains explicitly entered facts through source-linked Horn rules, answers variable queries, returns a bounded proof tree, and distinguishes supported, explicitly refuted, conflicting, mixed, and unknown. Predicate/argument indexes and semi-naive incremental closure let each inference round join only facts newly derived in the prior round; cached closure then serves repeated queries. Fixed work, 256 inference-round, 128 proof-depth, 256 proof-node, rule, fact, and answer limits keep it CPU-bounded. It does not parse ordinary sentences or treat missing facts as false.
- An explicit bounded composition benchmark now measures chained support reasoning under fixed proof and closure limits, checking that a direct support graph plus a two-hop rule yields the correct composed target set without allowing the engine to drift into unbounded search. This keeps symbolic composition measurable and auditable rather than hand-wavy.
- The bounded English interpreter (`src/elia/cognition/language.py`) can propose source-spanned symbolic atoms for a small grammar: “Alice is parent of Bob,” “Bob does not help Cara,” coordinated “Alice and Bob like Cara and Dana,” shared-subject “Alice likes Bob and helps Cara,” and passive “Bob was helped by Alice.” The workspace includes a review form that displays these proposals and unsupported segments. `POST /api/v1/language/interpret` is transient: it neither stores nor asserts proposals, requires review, and explicitly returns unsupported segments rather than guessing. It is a template-based prototype, not general language understanding.
- A bounded English question interface maps a small set of wh-, yes/no, and “why” forms (for example, “Who helps Bob?”, “Who does Alice help?”, “Does Alice help Bob?”, and “Why does Alice help Bob?”) to symbolic queries. `POST /api/v1/language/query` returns the interpreted query and proof provenance. Yes/no answers distinguish `yes`, `no`, `unknown`, and `conflict`; for who/what questions, `yes` means at least one supported witness, so explicitly refuted candidates do not negate a different supported answer. A supported “why” question returns a stored evidence trace (or says when the claim is refuted/unknown); the trace is a rule/fact derivation, not a real-world causal explanation. Unsupported phrasing is rejected rather than guessed. This is a small pattern interface, not general conversational understanding.
- `POST /api/v1/memory/compress` creates a source-linked memory compression report from evidence-bearing observations and hypotheses. The report groups duplicates by Unicode-normalized, case-folded text, retains a representative summary per cluster, records reduction statistics, and persists the report so the dashboard can show how much raw evidence was compacted without losing the underlying source-linked record set. This is a bounded text-normalization audit, not a claim of global semantic understanding or irreversible forgetting. Compression-only reports are marked `legacy_or_unbenchmarked_memory_compression` and do not claim a measured retention gate.
- `POST /api/v1/memory/benchmark` measures in-sample coverage of normalized memory-key groups, counting any evidence-bearing records that produce no usable key as lost. It rejects the result when coverage falls below the configured threshold (default 0.8); `memory_key_coverage` reports evaluated keys, unkeyed records, total evaluation units, and the method. The historical `held_out_coverage` field remains as a compatibility alias; this calculation does **not** use held-out tasks or establish generalization. A true held-out evaluation requires separately specified tasks/examples that are not used to construct the compressed memory.
- Deleting an episode deletes its observations and any hypotheses linked to those observations, along with their predictions and evaluation history. It also clears aggregate sleep-cycle, deliberation, and memory-compression snapshots because they lack per-source provenance and may have included or been measured against the deleted information; the HTTP deletion path also resets the live cognitive and reasoning caches. This intentionally favors complete removal of derived knowledge over retaining a claim or metric that may encode deleted information or describe a stale evidence set.
- All core operations use Python's standard library and SQLite. No model, GPU, vector database, cloud account, or external service is required.

Outcome labels and their source references are manually supplied; ELIA does not independently verify them. The promotion threshold is an initial engineering rule, not a statistically universal standard. Ten cases are still a small sample; reports include sample size and metrics so results are not presented as certainty.

## Cognitive architecture (experimental)

This subsystem tests whether small, auditable learning components can cooperate and retain labeled patterns on CPU. Its text encoder now represents bounded negation scope and short skip-gram relations as well as words, adjacent pairs, and character trigrams. This lets a supervised circuit learn some compositional contrasts from examples, but it does not provide broad language understanding, world knowledge, or autonomous agency.

- **Shared core** (`src/elia/cognition/core.py`): stable hashed word, adjacent-word-pair, short skip-gram, bounded negation-scope, and character-trigram features feed a sparse online logistic readout. Negation scope is a lightweight heuristic with explicit clause resets, not a grammatical parser; feature selection favors lexical and compositional cues under the active-feature bound.
- **Sparse substrate** (`substrate.py`): up to 10,000 addressable units with at most 16 outgoing learned associations per unit. It is a bounded associative network, not a biological brain simulation.
- **Specialist circuits** (`circuit.py`): separate binary readouts for up to 64 named domains, allowing the same text to receive different labels in different domains. Once a specialist has learned, its head makes that domain's prediction; the shared readout is a fallback before that.
- **Router** (`router.py`): routes by similarity to learned domain profiles, but also compares a request with a bounded bank of 128 retained examples per specialist. Automatic routing abstains on weak/ambiguous profiles or if the nearest retained example is below a conservative 0.45 cosine-similarity threshold. This is a prototype novelty guard, not a statistically calibrated guarantee; explicit specialist selection intentionally bypasses automatic routing.
- **Replay** (`replay.py`): retains up to 128 labeled examples per specialist and replays a balanced, surprise-prioritized batch (16 per specialist by default), with a hard global limit of 128 updates per training example.
- **Coordinator** (`system.py`): serializes prediction and training operations. The database stores the source-linked examples; weights and associations are reconstructed deterministically at startup and after source deletion.
- **Experimental plasticity** (`plasticity.py`): a separate 96-unit CPU prototype uses top-5 competitive activity, local receptive-field updates, and independently bounded co-activation links. It is not wired into prediction, trained from conversations, or included in dashboard learning counts. It must remain isolated until it is evaluated on consented, representative held-out tasks.

The compositional encoder has a focused synthetic unit check: after four labeled positive/negative example phrasings, it classified held-out variants “not really helpful” and “not at all helpful” below 0.36 probability while “helpful” was above 0.69. This tests a tiny learned negation contrast, not broad English understanding. On this machine, 1,000 warmed predictions of a 776-character input took about 2.6 seconds total (about 2.6 ms each); this is a local microbenchmark, not a latency guarantee.

Training is deliberately not automatic conversation ingestion. To train, first store an episode with explicit consent, create a linked observation, then submit that observation with a specialist ID, a binary label, explicit consent, and a verification-source reference. The reference documents the basis for the label; ELIA does not independently verify it. A prediction request can specify a specialist or let the router infer one. Automatic routing now abstains when the input is unlike both the specialist profile and its bounded retained exemplars. This guard is based on similarity, not a learned/certified drift detector; explicit specialist selection is an intentional override. Prediction requests are transient and are not saved as training data.

### Cognitive API

- `GET /api/v1/cognition/status` — subsystem configuration and counts, including stored source-linked examples.
- `GET /api/v1/cognition/experiences` — the most recent 100 training records and their pre-update probability, surprise, replay count, and verification reference.
- `POST /api/v1/cognition/experiences` — train from an existing consented episode and its linked observation. Required JSON fields: `consented: true`, `episode_id`, `observation_id`, `specialist_id`, `label` (`true`/`false` or `1`/`0`), and `verification_source`.
- `POST /api/v1/cognition/predict` — make a transient prediction. Required JSON field: `text`; optional `specialist_id`. The response includes the decision, route, probability (when it can predict), and active-unit count.
- `POST /api/v1/cognition/counterfactual` — compare a transient prediction with a temporary learner rebuilt without one selected source episode, e.g. `{"episode_id":"episode-id","text":"transient input","specialist_id":"support"}`. Returns the complete-memory and withheld-source predictions, numbers of examples retained/withheld, and probability delta when comparable. It never mutates live or persistent learned state; differences measure sensitivity, not proof of cause.
- `GET /api/v1/functional/status` — inspect the additive workspace, operational self-model, reflection report, state-controller counters, bounded rollout health, and a live `global_workspace` snapshot with `attention_stack` and `priority_snapshot`.
- `POST /api/v1/functional/update` — inject a transient task and metadata into the workspace/self-model/reflection loop. Accepts `task`, `evidence`, `hypotheses`, `questions`, `priorities`, `confidence`, `uncertainty`, `decision`, `trace`, `conflict`, and `pending` in a bounded JSON payload.
- `GET /api/v1/functional/focus` — inspect the workspace's highest-priority active item and current evidence counts.
- `POST /api/v1/functional/consolidation/enqueue` — queue the current workspace snapshot for bounded consolidation. Send `{}` as the JSON body. The queue is capped at 8 batches of up to 64 items each.
- `POST /api/v1/functional/consolidation/run` — process one queued batch and return a deduplicated, review-required report. Send `{}` as the JSON body. Reports are not persisted to primary memory and never replace source records.
- `POST /api/v1/sleep` — run and persist an evidence-only self-evaluation cycle. `GET /api/v1/sleep` returns the latest report and up to 20 past cycles.
- `POST /api/v1/think` — inspect the current evidence and persist one explainable, propose-only next step. `GET /api/v1/think` returns the latest thought and up to 20 prior recommendations.
- `POST /api/v1/reasoning/facts` — save a ground logical fact linked to an existing source observation, e.g. `{"fact":"parent(alice, bob)","observation_id":"..."}`.
- `POST /api/v1/reasoning/facts/reviewed-language` — persist only explicitly reviewed interpretation proposals. Requires `reviewed: true`, the selected source `observation_id`, and selected proposals containing `fact`, `text`, `start`, and `end`. ELIA checks that the source span exactly matches the observation and that the fact is still proposed by the interpreter for that observation.
- `POST /api/v1/reasoning/rules` — save a source-linked rule such as `parent(?x, ?y) & parent(?y, ?z) -> grandparent(?x, ?z)`. Rules must be range-restricted; negated premises and inferred existence of unbound values are rejected.
- `POST /api/v1/reasoning/query` — query with the same notation, e.g. `{"query":"grandparent(alice, ?who)"}`. The response includes bindings and proof provenance. `GET /api/v1/reasoning` lists stored rules/facts with the source observations that justify adding them.
- `POST /api/v1/reasoning/counterfactual` — compare a symbolic query with and without all facts/rules linked to one source episode, e.g. `{"query":"grandparent(alice, ?who)","episode_id":"episode-id"}`. The temporary run leaves stored knowledge untouched and can reveal alternate proofs that survive withholding; a changed state indicates dependence on the removed records in this knowledge base, not real-world causality.
- `POST /api/v1/reasoning/resilience` — audit a query by withholding every evidence-bearing source episode one at a time, e.g. `{"query":"grandparent(alice, ?who)"}`. Returns pivotal and individually redundant source IDs and per-source lost bindings; capped at 32 source episodes. It is a single-source sensitivity audit, not a test of all combinations.
- `POST /api/v1/reasoning/failure-sets` — test individual source removals and source pairs for a query, e.g. `{"query":"grandparent(alice, ?who)","evaluation_budget":529}`. Reports minimal two-source failure sets, surviving answers/proof-source IDs, exact evaluations and pair coverage, cumulative reasoner work, and elapsed milliseconds. Up to 32 source episodes (496 pairs); the optional budget is capped at 529 total reasoning evaluations, and cumulative reasoning work at 1,000,000. `complete: false` means the audit stopped at one of those limits. It does not test triples or larger groups.
- `POST /api/v1/language/interpret` — propose review-required atoms from a bounded subset of English, e.g. `{"text":"Alice is not a parent of Bob."}`. Responses include source character offsets and unsupported segments. Proposals are never persisted by this endpoint. In the dashboard, select the matching source observation, check only the facts you approve, and explicitly confirm to store them; the server verifies each fact and span against the source.
- `POST /api/v1/language/query` — ask a supported simple-English question, e.g. `{"question":"Who does Alice help?"}` or `{"question":"Why does Alice help Bob?"}`. ELIA returns a user-facing answer label and interpreted symbolic query. “Why” responses include a stored proof/counterproof trace, explicitly not a causal explanation. Who/what questions answer yes only when at least one witness is supported, otherwise unknown or conflict. Unsupported question forms return HTTP 400.

The reasoning UI lives under “Build a learning record.” Example: add `parent(alice, bob)` and `parent(bob, cara)` as facts, add `parent(?x, ?y) & parent(?y, ?z) -> grandparent(?x, ?z)` as a rule, then query `grandparent(alice, ?who)`. ELIA should return `?who = cara` and a proof chain. Facts and rules each require a source-linked observation; deleting their source removes them. Supported rule syntax is intentionally small: predicates with simple constant/variable arguments, positive premises joined by `&`, and one `->` conclusion. No quoted/free-text terms, functions, disjunction, or negation-as-failure.

On the local development machine, a microbenchmark with 1,501 facts and one rule built closure in about 4 ms; 100 repeated indexed queries took about 2.5 ms total. This synthetic timing only demonstrates that the bounded/indexed implementation is lightweight on that machine; it is not an end-to-end service SLA.

The dashboard polls the API every five seconds; it is not a push-based live stream. Its counts describe actual stored and computed state, not a generated “intelligence” score. The cognitive-space view is an interactive 3D-projected canvas: rotate by dragging, zoom with the wheel/buttons or a touch pinch, filter by cloud, and select a node to inspect its stored record and provenance. When motion is allowed by the device settings, clouds drift gently, nodes breathe, and connection pulses travel along actual links; newly received records briefly glow into the graph with their new links. Animation pauses in background tabs and honors reduced-motion preferences. It draws the implemented architecture and records returned by the local APIs (up to the latest 100 per feed), with links derived from stored source/evidence relationships; it does not invent placeholder learning activity. Keyboard focus on the graph supports arrow-key rotation and +/- zoom.

### Research package findings applied

The supplied `C3_Cognitive_Architecture_Benchmark_Upgrade` and `C3_Neuron_Self_Organization_Tests` packages contain synthetic controlled experiments, not conversational-agent or general-intelligence tests. Their most relevant design constraints are:

- Keep specialists and shared/fallback paths distinct. In the synthetic clean task suite, the 512-unit sparse specialists scored 95.85% versus 94.09% for dense top-2 experts; under severe context shift, the online sparse path scored 78.58% versus 83.58% for dense experts. Router confidence also stayed high while routing accuracy degraded. These results motivate testing drift-aware abstention and a fallback, not trusting confidence alone.
- Treat neural self-organization as a separate experimental branch. Across 20 synthetic seeds, rate-based local plasticity raised held-out concept-pair F1 from 53.39% (frozen weights) to 78.73% and learned partner-cue recall from 27.50% to 91.25%. However, 64-unit Hopfield exact recall fell from 94.67% with six patterns to 8% with sixteen patterns at 20% corruption. The simplified STDP condition had strong receptive-field alignment but only 31.25% partner recall; recurrent association must be measured independently from feature learning.
- The isolated ELIA plasticity benchmark compares frozen and plastic features on eight overlapping synthetic concepts, using 320 unlabeled learning examples and 240 separately generated held-out examples per seed. An offline nearest-centroid readout scores the learned representation. Across five initialization/data seeds, mean held-out accuracy was 91.6% frozen and 99.1% plastic (+7.5 percentage points). Separate tests confirmed that recurrent association recall appears only after explicit pair-training and that every unit's fan-out remains bounded. This is evidence only for the generated toy task; the repeated seeds are not real-world validation, and the benchmark does not show language understanding or user-driven learning. Run it with `python -m elia.cognition.plasticity_benchmark --seed 2026 --seeds 5` from the project root after setting `PYTHONPATH` to `C:\elia_ai\src`.
- Preserve bounded replay, source provenance, held-out evaluation, and explicit promotion/rollback gates. The new self-evaluation cycle audits existing predictions and evidence only; it does not yet retrain a candidate model, and no claims of general language reasoning follow from these experiments.
- Before changing the current learning substrate based on these results, reproduce the useful comparisons against ELIA's actual tasks and CPU environment. Synthetic benchmark accuracy is not a user-facing capability metric.

### Synthetic continual-learning benchmark

Run a multi-order comparison from `C:\elia_ai`:

```powershell
$env:PYTHONPATH = "C:\elia_ai\src"
python -m elia.cognition.benchmark --seed 2026 --seeds 5 --examples-per-class 12
```

The benchmark compares shared-only/no-replay, shared-plus-specialists/no-replay, and shared-plus-specialists/balanced-replay. It reports accuracy, per-domain accuracy, routing accuracy, abstentions, forgetting, runtime, and variability across training-example orderings. A seed changes only training order: generated examples and held-out cases stay fixed, so this is **not** an independent-sample confidence interval or evidence of generalization. The generated examples have known lexical domain cues and balanced binary labels; this is a controlled retention/routing test, **not** a natural-language, open-world, or general-intelligence evaluation. Use separate, independently generated real-world task benchmarks before claiming useful generalization. The benchmark’s own numbers should not be conflated with results in the supplied PyTorch experiment package, which used a different model and task setup.

Latest measured run: 24 training examples per class/task and five training orders (seeds 2026–2030) produced mean held-out accuracy of **50.1%** for shared-only/no-replay, **53.4%** for specialists/no-replay, and **100%** for specialists with balanced replay. Replay had zero measured forgetting and all cases routed without abstention in this toy suite. This perfect score is evidence only that the current components learn this deliberately regular synthetic curriculum; it is not evidence of broad learning ability. The seed sweep is order sensitivity, not independent task or test-set replication.

## Run on Windows

Requires Python 3.11 or newer.

```powershell
cd C:\elia_ai
python -m unittest discover -s tests -v
.\run.ps1
```

Open <http://127.0.0.1:8765>. The server binds only to loopback. Set `ELIA_DATA_DIR` to choose another local data directory and `ELIA_PORT` to change the port.
The dashboard refreshes from the API every five seconds; it does not use a push event stream yet.

## First experiment

1. Add a consented episode and a source-backed observation.
2. Create a narrow hypothesis and link supporting or challenging observations.
3. Register predictions before outcomes are known.
4. Record explicitly verified outcomes. Unknown outcomes are retained but not counted as successes.
5. Run evaluation and inspect the sample size, predictor Brier score, and registration-time baseline Brier score.
6. Create a benchmark before claiming improvement; compare these results with transcript retrieval and summaries on the same chronological, held-out cases.

Predictions are not independent merely because they have different IDs or contexts; the exact same context cannot be registered twice for one hypothesis, but similar cases may still be correlated. Evaluation reports expose raw sample counts and should not be interpreted as proof of broad capability. User silence is never an outcome.

## API overview

- `GET /api/v1/dashboard` — measured counts and recent activity.
- `GET /api/v1/episodes`, `POST /api/v1/episodes` — source episodes.
- `GET /api/v1/episodes/{id}` — inspect the locally stored source.
- `DELETE /api/v1/episodes/{id}` — delete an episode and derived data.
- `POST /api/v1/observations` — source-linked observation.
- `GET /api/v1/hypotheses`, `POST /api/v1/hypotheses` — hypotheses and evidence links.
- `POST /api/v1/predictions` — immutable prospective prediction.
- `POST /api/v1/outcomes` — explicitly verified outcome.
- `GET /api/v1/evaluations`, `POST /api/v1/evaluations` — inspect or calculate evaluation reports.

POST requests use JSON. See `tests/test_app.py` for request examples. The API intentionally does not ingest conversations automatically or run executable procedures.

### Held-out generalization benchmark

ELIA includes a narrow synthetic transfer benchmark for a simple binary relation. It tests the shared sparse core and specialist router on entity names and pairs held out from training, using relation templates already present in training. Training and held-out name sets are disjoint (reported as `entity_names_disjoint: true`). The held-out set contains 230 balanced examples: 23 new entity pairs evaluated with each of five positive and five negative templates. It does **not** test unseen paraphrases. The benchmark remains synthetic and intentionally bounded:

- Run: `python src/elia/cognition/benchmark.py --benchmark generalization --seed 2026`
- Or from the project root: `PYTHONPATH=src python -m elia.cognition.benchmark --benchmark generalization --seed 2026`
- The report identifies its scope as `unseen_entity_names_and_pairs_with_training_seen_templates`; its held-out examples use names disjoint from the training set, in new combinations under familiar help/no-help templates.
- Passing criteria: accuracy >= 0.8 with positive and negative classes both predicted above chance; the report exposes this as `class_balance_pass`.
- Meaning: this is a controlled transfer test for bounded relational pattern learning; it is not a claim of unseen-paraphrase transfer, open-ended understanding, general reasoning, or human-level intelligence.

This benchmark complements the existing continual-learning curriculum by checking whether the system can adapt beyond the exact lexical patterns it observed.

### Unseen-template transfer benchmark

The stricter `unseen-templates` benchmark trains on the same bounded help/no-help curriculum, then evaluates 230 balanced examples over entirely held-out names using ten relation templates absent from training. Some words are shared with the training vocabulary, so this is template-level recombination, not open-ended language understanding.

- Run: `python src/elia/cognition/benchmark.py --benchmark unseen-templates --seed 2026`
- Check training-order sensitivity with: `python src/elia/cognition/benchmark.py --benchmark unseen-templates --seed 2026 --seeds 30`; the sweep reports per-seed outcomes, pass rate and a Wilson 95% interval for sampled training-order pass rate, mean accuracy, and order variability. The interval applies only to this fixed synthetic curriculum.
- The report separately lists class accuracy and accuracy for each held-out template; it passes only at >=0.8 overall accuracy with both classes above chance.
- At seed 2026 it scores 0.874 overall (0.957 positive, 0.791 negative). Across 30 training-order seeds (2026–2055), 13 pass (43.3%; Wilson 95% interval 27.4%–60.8%); mean accuracy is 0.744 (order SD 0.122), and mean negative-class accuracy is 0.518. This seed/order sensitivity is a known measured limitation, not hidden by the default run.
- Meaning: a pass for one training order is not evidence of robust paraphrase generalization. The benchmark is deliberately retained to expose this weakness; the model has not been tuned against these held-out templates.

### Recent implementation pillars

The current prototype includes three concrete extensions that keep ELIA bounded but more useful:

- Stronger representation learning: the shared sparse core now preserves relation and quality cues such as `help(subject, object)` and `is_helpful(subject)` as explicit features, improving alignment between syntax and task-specific learning without unbounded expansion.
- Richer language interpretation: the bounded English interpreter now recognizes coordinated quality statements like “Alice and Bob are helpful and available,” while preserving exact source spans and requiring review before storing facts.
- Better planning: the deliberation path now routes through a task planner that prioritizes conflicting evidence, pending verification, candidate tests, and evidence gathering in a deterministic order.

### Additive functional layers

The existing planner, SQLite memory lifecycle, and prediction response remain the baseline. The functional layers run alongside that baseline and do not change planner decisions or write to primary memory:

- The workspace holds bounded, transient foreground evidence, hypotheses, priorities, confidence values, decisions, and trace entries. Its status includes capacities, update count, and evictions.
- The self-model reports configured constraints and measured task/update counters. Any configured capability scores are explicitly identified as estimates rather than benchmark results; no subjective experience or consciousness is claimed.
- Reflection is a review-only quality assessment. It records the latest report and bounded evaluation/risk counters without changing predictions, plans, or memory.
- The state controller records a bounded mode history and counts for its explicit modes. It is observational coordination metadata; the existing task planner remains authoritative.
- Background consolidation uses a bounded queue and processes one snapshot at a time when explicitly scheduled. It preserves a human-review boundary and reports `persisted: false`; it never edits SQLite memory or learned weights.

These layers are staged and inspectable through the functional API. Workspace and controller state are process-local and transient; only the existing memory and evaluation APIs persist records.

## Research boundaries

The current baseline is the smoothed historical frequency of the predicted outcome for that hypothesis, captured when the prediction is registered. The score is binary Brier loss; lower is better. Unknown outcomes are excluded from both scores. Candidate hypotheses can be useful before they have enough evidence to validate; the API does not represent them as validated.

Meta-TRACE, confidence debt, automatic translation of conversation into logical facts, voice, generated procedures, cross-app collection, and a live push event stream are deliberately deferred. The symbolic rule engine demonstrates multi-step deduction on structured inputs, but it is not general natural-language reasoning, consciousness, or autonomous action. The deliberation endpoint is deterministic evidence triage. The cognitive classifier is currently a narrow, supervised pattern learner; none of these components replaces an LLM-level reasoner. The dashboard polls the local API and displays actual stored measurements, not simulated learning activity.
