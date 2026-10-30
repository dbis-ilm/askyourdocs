# Ask Your Documents

[![CI](https://github.com/dbis-ilm/askyourdocs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dbis-ilm/askyourdocs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dbis-ilm/askyourdocs/branch/main/graph/badge.svg)](https://codecov.io/gh/dbis-ilm/askyourdocs)

A small, reusable RAG engine for building "ask your documents" apps: parse documents, chunk them for retrieval, index them, and get back a Genkit retriever — without pulling in any particular app's domain (document-specific metadata, auth, admin UI). It does include the small, domain-neutral building blocks most QA apps need around that core — an async job queue, a feedback log, multi-turn follow-up helpers — see [App building blocks](#app-building-blocks-jobs-feedback-multi-turn-embeddings).

## Goal

Everything that's the same regardless of *what* you're asking questions
about or *who's* asking lives here: turning a PDF/.docx file or a web page
into retrievable chunks, and indexing/searching those chunks. Everything
that differs per app — what metadata a chunk carries beyond its text (a
legal-document app's tag/date/rank vs. a chatbot's url/breadcrumb), how
answers are prompted, who's allowed to ask — stays in the app.

This module has its own `go.mod`. It depends on
[Genkit](https://genkit.dev) (for `ai.Document`/`ai.Embedder`/`ai.Retriever`
and the localvec/pgvector store wiring), pgx + pgvector-go, a PDF reader and
an HTML parser — nothing specific to any one app (no LDAP, no googlegenai/
grpc; the job queue and feedback log here are generic in-memory/JSON-file pieces, not a particular app's). Run `go list -m all` in this repo to see that list —
unlike when run from a consumer's workspace, there's nothing else to filter
out.

## Structure

```
document.go   — PDF text extraction, paragraph splitting
docx.go       — .docx text extraction (OOXML), feeding the same pipeline as PDFs
web.go        — single-page HTML → structured paragraphs (headings, tables, lists)
               (document.go/docx.go/web.go together define Paragraph, Chunk, BuildChunks)
crawl.go      — multi-page BFS crawling (RunCrawl), crawl-data Markdown caching (ReindexCrawlData)
store.go      — VectorStore interface, IndexInBatches, Source, LocalvecStore
pgstore.go    — PgvecStore (PostgreSQL + pgvector)
ingest.go     — IngestSource interface + the IndexSource pipeline
flow.go       — DefineProgressFlow: generic long-running ingest/crawl flow wiring
retrieve.go   — PromptOptions, GenerateHyDEText, RerankScores, ReciprocalRankFusion
crossencoder.go — CrossEncoder, ScaleScores: reranking with a cross-encoder model via a /rerank endpoint
search.go     — ExtractKeywords, ExtractQuotedPhrases, LocalvecKeywordSearch, KeywordSearchSQL (PgvecStore.KeywordSearch)
eval.go       — RegisterEvaluators: custom/faithfulness, custom/answerRelevancy, custom/injectionResistance
evalgen.go    — EvalCase[In], GenerateEvalSet: build a genkit eval:flow dataset from a directory of documents
mcp.go        — MCPModeRequested, JSONToolHandler, ServeMCPStdio: expose Genkit tools over stdio
embed.go      — NewPrefixEmbedder, QueryDocument: instruction prefixes for asymmetric embedding models
provider.go   — Providers, GenerateConfig: resolve "ollama/…", "gpustack/…", "googleai/…" model specs
logging.go    — SetupLogging, LogConfigFromEnv: log level, log file, JSON, off
jobs.go       — JobStore: in-memory async job queue with progress, one background worker
jobhttp.go    — FlowJobHandler, JobHandler, JobListHandler: start a flow as a job and poll it over HTTP
feedback.go   — FeedbackStore[E]: JSON-file log of Q&A interactions plus thumbs up/down ratings
safety.go     — SanitizeUntrusted, WrapContext, SanitizeAnswerWith: prompt-injection hardening
evalhttp.go   — EvalJobHandler, EvalDatasetHandler, RestrictEvalPaths: HTTP contract for remote dataset fetching
cmd/evalset/  — the evalset command
injection.go  — DetectInjection, InjectionAttacks/Payloads/EvalCases: detection and eval material
conversation.go — multi-turn helpers: CondenseFollowUp, TrimHistory, BestExcerpt, RateConfidence
id.go         — NewRandomHexID
```

### Text extraction & chunking

Turns a PDF, a .docx file, or a single web page into a flat list of paragraphs, then groups those paragraphs into overlapping chunks sized for embedding.

```go
type Paragraph struct { Text string; PageNum int; Heading bool; Breadcrumb string }
type Chunk struct { Text, ParentText, Pages, Section, Top, Breadcrumb, URL string }

func SupportedDocumentExt(ext string) bool
func ExtractParagraphs(path string) ([]Paragraph, error)        // .pdf or .docx
func HasTextLayer(paras []Paragraph) bool                       // false for a scanned PDF
func FetchWebPage(url string) (string, error)
func FetchWebPageFollowingRefresh(url string) (page, from string, err error) // follows <meta refresh> stubs, same host
func HTMLToStructuredText(r io.Reader) ([]Paragraph, error)
func BuildChunks(paras []Paragraph, maxSize, overlapSize int) []Chunk
func NormalizeSpaces(s string) string
```

`Chunk.ParentText` is an expanded window around each chunk (up to
`max(maxSize*2, 3000)` chars, bounded by section headings) — index
`Chunk.Text` for embedding, but feed `Chunk.ParentText` to the LLM at answer
time for more context than a single small chunk gives you.

A scanned PDF without a text layer extracts to nothing, or to a page number
and a stamp per page. `HasTextLayer` tells that apart (under 20 non-space
characters per page on average), so an app can refuse such a file with a
"needs OCR" message in its own language instead of indexing an empty
document.

### Indexing

Defines the minimal write-side contract a vector store must implement to be indexable, and the generic batched-write + per-source ingest pipeline built on top of it.

```go
type VectorStore interface {
	Index(ctx context.Context, docs []*ai.Document) error
	DeleteByURL(ctx context.Context, url string) (int, error)
	DeleteBySource(ctx context.Context, sourceName string) (int, error)
	Close() error
}

func IndexInBatches(ctx context.Context, store VectorStore, docs []*ai.Document, progress func(done, total int)) error

type IngestSource interface {
	Fetch(ctx context.Context) ([]Paragraph, error)
	Documents(chunks []Chunk) []*ai.Document
	DeleteStale(ctx context.Context, store VectorStore) (int, error)
	Name() string
}

func IndexSource(ctx context.Context, store VectorStore, src IngestSource, chunkSize, chunkOverlap int, progress ProgressFunc) (indexed, deleted int, err error)
```

`VectorStore` is deliberately the *write* side only — four methods. An app
needing filtered retrieval, keyword search, or source administration (list/
retag/delete a source) defines its own, larger interface; any store
satisfying that larger interface still satisfies this one too, as long as
these four methods keep this exact signature. No embedding or wrapping
required to pass such a store wherever `askyourdocs.VectorStore` is expected
— an app's own richer store, with six more methods, still passes straight
into `IndexSource`. See [Extending a store](#extending-a-store) below.

`IndexSource` runs the whole extract → chunk → delete-stale (upsert) →
index-in-batches → cleanup-on-failure pipeline for one `IngestSource`. Write
one `IngestSource` per document type; the pipeline itself is written once,
here, instead of once per app per document type.

### Stores

Two ready-made `VectorStore` implementations, so a new app doesn't have to
hand-roll either:

```go
func NewLocalvecStore(g *genkit.Genkit, name string, emb ai.Embedder, dir string, opts ...LocalvecOption) (*LocalvecStore, error)
func NewPgvecStore(ctx context.Context, g *genkit.Genkit, name string, emb ai.Embedder, dsn string, embeddingDim int, opts ...PgvecOption) (*PgvecStore, error)
```

`LocalvecStore` wraps Genkit's `localvec` plugin — a file-based store, no
database to run, good for a single-node app or a demo. `PgvecStore` wraps
PostgreSQL + pgvector, with schema creation. Both expose a `Retriever ai.Retriever` field
(`PgvecStore` additionally exposes `Pool` and `Embedder`) for callers that
want to retrieve directly rather than through `IndexSource`'s write path.
`PgvecStore`'s retriever takes `PgvecRetrieverOptions{K}` (or a map with the
same keys) as the request's options.

`PgvecStore` takes three options:

- `WithTextSearchLanguages(langs...)` — the Postgres text-search
  configurations for full-text search (default `"german"`), stored in the
  `TextSearchLanguages` field. Every chunk is stemmed with each of them into
  one generated `tsv` column with a GIN index on it, so a corpus mixing
  languages — even within one document, like a German page full of English
  terms — needs no language detection. `PgvecStore.KeywordSearch` matches
  the query stemmed with all of them. Each extra language costs roughly one
  more tsvector per row in the table and the index. A column built from a
  different set is rebuilt at startup (the column's comment records the
  set); that rewrites the table, so expect a pause on a large one.
- `WithResetOnDimensionMismatch()` — when the stored vectors have a
  different dimension than `embeddingDim` (e.g. after switching embedding
  models), delete every stored document and migrate the column. Without it,
  `NewPgvecStore` fails with `ErrEmbeddingDimMismatch` instead, so a wrong
  dimension in a config can't wipe the index. An empty table is always
  migrated.
- `WithSourceMetaKeys(keys...)` — see [Source administration](#source-administration)
  below. `LocalvecStore` takes the same as `WithLocalvecSourceMetaKeys`.

`Retrieve(ctx, query, k)` on either store is a plain similarity search.
Filtering comes in each store's own terms, the same way keyword search
does:

```go
func (s *PgvecStore) RetrieveWhere(ctx context.Context, query *ai.Document, k int, extraWhere string, extraArgs []any) ([]*ai.Document, error)
func (s *LocalvecStore) RetrieveFiltered(ctx context.Context, query *ai.Document, k int, filter func(meta map[string]any) bool) ([]*ai.Document, error)
```

`RetrieveWhere` takes a SQL condition whose placeholders start at `$3`
(`$1` is the query vector, `$2` the limit) — the same numbering as
`KeywordSearch`, so an app renders its filter once for both. The condition
narrows what the index ranks, so a rare match still fills `k`.
`RetrieveFiltered` can only filter after localvec has ranked, so it fetches
10·k candidates and may return fewer than `k` when the filter is narrow.

#### Source administration

Both stores list, retag and delete indexed documents:

```go
type Source struct {
	URL, Source string
	Metadata    map[string]string
	Chunks      int
}

func (s *PgvecStore) Sources(ctx context.Context) ([]Source, error)
func (s *PgvecStore) UpdateMetadata(ctx context.Context, old Source, newMetadata map[string]string) (int, error)
func (s *PgvecStore) DeleteSource(ctx context.Context, src Source) (int, error)
func (s *PgvecStore) Reset(ctx context.Context) error
// ... and the same four on *LocalvecStore
```

A source is a group of chunks with the same url, source and the metadata
fields named with `WithSourceMetaKeys` — which fields those are is the
app's call (ask-the-records uses `tag`, `datum`, `rank`; answer-email
`studiengang`, `abschluss`, `doctype`, `semester`). Without the option a
source is url and source alone. A file whose chunks carry two different
tags is two sources, so an admin can see a half-finished retag;
`UpdateMetadata` moves one group's chunks to new values (an empty value
removes the field), and `DeleteSource` removes exactly one group, unlike
`DeleteBySource`, which removes every chunk of a file.

The keys are set on the store rather than passed to each call because they
define one grouping that `Sources`, `UpdateMetadata` and `DeleteSource` must
agree on. They are checked to be plain names, since `PgvecStore`
interpolates them into SQL.

`Source` marshals to one flat JSON object — `{"source":"a.pdf","tag":"Senat","chunks":3}` —
and unmarshals every other field into `Metadata`, so an admin UI can send a
listed source back unchanged as the `old`/`src` argument.

### Crawling

Crawls a multi-page site breadth-first, strips boilerplate that repeats across pages, and indexes what it finds — or replays an earlier crawl's cached pages without re-fetching.

```go
type CrawlInput struct { SeedURLs []string; MaxPages, MaxDepth, DelayMs int }
type CrawlResult struct { PagesVisited, PagesSaved, PagesIndexed, ChunksIndexed, PagesUnchanged, PagesRemoved int; Errors []string }
type ReindexResult struct { FilesProcessed, PagesIndexed int; Errors []string }

func RunCrawl(ctx context.Context, cfg CrawlInput, outputDir string, indexPage IndexPageFunc, progress ProgressFunc, opts ...CrawlOption) (CrawlResult, error)
func ReindexCrawlData(ctx context.Context, dir string, indexPage IndexPageFunc, progress ProgressFunc) (ReindexResult, error)

func WithManifest(fingerprint string) CrawlOption
func WithForceReindex() CrawlOption
func WithRemovePage(remove func(ctx context.Context, pageURL string) error) CrawlOption
```

`RunCrawl` does a breadth-first crawl from `cfg.SeedURLs` (same-host/path-prefix only), strips paragraphs that repeat across pages (boilerplate: nav, footers, repeated cards) before indexing, and — with `SAVE_CRAWL_DATA=1` in the environment — caches each cleaned page as Markdown under `outputDir`. `ReindexCrawlData` replays that cache through `indexPage` without re-fetching anything, for when a chunking fix or a changed chunk size should reach an already-crawled site. Carries no app-specific metadata itself — like `IndexSource`, that's entirely the `IndexPageFunc` closure's job.

#### Re-crawling

Running `RunCrawl` again over the same seeds re-crawls the site; `indexPage` is expected to replace a page's chunks (`DeleteByURL`, then index). Three opt-in options make a scheduled re-crawl cheap and complete:

- **`WithManifest(fingerprint)`** keeps `crawl-manifest.json` (`CrawlManifestFile`) in `outputDir`: a hash of every indexed page's cleaned text. A page whose hash is unchanged skips `indexPage` and so the embedding, counted in `PagesUnchanged`. Put whatever changes what `indexPage` makes of the same text — chunk size/overlap, the metadata the closure attaches — into `fingerprint`, so changing it reindexes instead of silently keeping the old chunks. A page whose indexing failed isn't recorded and is retried next time. Several crawls with different seeds can share one `outputDir`; the manifest is merged, not overwritten.
- **`WithForceReindex()`** indexes every page regardless and still updates the manifest — for after the index was emptied. An app's index reset should also delete the manifest, or the next crawl skips every page as unchanged.
- **`WithRemovePage(remove)`** (needs `WithManifest`): every manifest page under this crawl's seeds that wasn't crawled this time is fetched once more, and **only a 404 or 410** removes it — `remove` is called, the cached Markdown file is deleted (or `ReindexCrawlData` would bring the page back), and the page leaves the manifest. A 200, a 5xx or no answer keeps it: a crawl cut short by `MaxPages`, a page that is merely no longer linked, or a server having a bad minute must not empty the index. Pages under other seeds are never touched.

#### Fetch safety (SSRF)

A crawler fetches whatever URL it is given from the machine it runs on, so anyone who can start a crawl could otherwise make that machine read services only it can reach (a local model server, a database, an admin page, a cloud metadata endpoint at `169.254.169.254`) and get the content back through the index. `FetchWebPage` therefore:

- **refuses private targets by default** — loopback, RFC 1918, link-local, carrier-grade NAT (`100.64/10`), multicast/reserved ranges and IPv6 forms that embed an IPv4 address (IPv4-mapped, NAT64, 6to4). The check runs on the IP actually about to be connected to, after DNS resolution and for every redirect hop, so a hostname that resolves to a private address, or flips to one between check and connect, is caught as well. The error wraps `ErrPrivateTarget`.
- follows at most 10 redirects, http/https only;
- reads at most 10 MiB per page (`SetMaxPageBytes`);
- ignores `HTTP(S)_PROXY` unless private targets are allowed (a proxy resolves targets itself, so the address check can't apply).

To crawl your own intranet or a local development site, an app opts in once at startup with `SetAllowPrivateTargets(true)` — only when the crawl can't be started by someone you don't trust. This module's own tests serve pages from `127.0.0.1` and enable it in `TestMain`.

### Long-running flows

Wraps a long-running ingest/crawl operation as a Genkit flow that reports progress as it runs, instead of making every such flow redefine that wrapping itself.

```go
func DefineProgressFlow[In, Out any](g *genkit.Genkit, name string, fn func(ctx context.Context, input In, progress func(string)) (Out, error)) *core.Flow[In, Out, string]
```

Every ingest/crawl flow (`indexPDFDocument`, `crawlWeb`, `reindexAllDocuments`, ...) is a Genkit streaming flow only so an HTTP layer can run it as a background job and surface progress messages as it goes — that wrapping (`core.StreamCallback[string]`, calling it as `sendChunk(ctx, msg)`) is identical across every one of them. `DefineProgressFlow` does that wrapping once: write `fn` against a plain `progress func(string)` and get a `*core.Flow[In, Out, string]` back, usable exactly like any other Genkit flow (`.Run` for a synchronous caller that discards the stream, `.Stream` for incremental progress). This is *not* the right shape for a flow that streams its actual output rather than progress-on-the-way-to-a-result — e.g. an answer flow streaming response text — those still call `genkit.DefineStreamingFlow` directly.

### Progress events

`IndexSource`, `RunCrawl` and `ReindexCrawlData` report progress as structured `Progress` events (`Stage`, `Done`, `Total`, plus `URL`/`Depth`/`File` where they apply), not as text. Wording a status line for a user, in whatever language, is app-side like all other user-facing text. For the common case of a flow or job that streams text, `ProgressMessages` adapts a `func(string)` with a formatter, and `GermanProgress` words the events as the German lines these functions used to send themselves:

```go
type Progress struct { Stage ProgressStage; Done, Total, Depth int; URL, File string }
type ProgressFunc func(Progress)
func ProgressMessages(send func(string), format func(Progress) string) ProgressFunc
func GermanProgress(p Progress) string
```

`CrawlResult`/`ReindexResult` count pages in `PagesIndexed` and the chunks those pages produced in `ChunksIndexed`.

### App building blocks: jobs, feedback, multi-turn, embeddings

Small pieces every QA app ends up needing, none tied to a particular document domain:

- **`JobStore`** — runs work one job at a time on a single background worker (so an index and a crawl never compete for the same embedding backend), tracks status/progress/result in memory, and caps its history. `RunStreamingFlowAsJob` drains a flow made with `DefineProgressFlow` into a job's progress and result. Enqueue with the app's long-lived context, not an HTTP request's.
- **`FlowJobHandler`, `JobHandler`, `JobListHandler`** — the HTTP side of that: `POST` `{"data": In}` starts a flow as a job and answers `202 {"jobId": ...}` (400 for a bad or over-1-MiB body, 503 for a full queue), `GET .../{id}` returns the job, `GET ...?limit=N` the newest jobs. `JobHandler` takes the 404 message, which an admin UI shows; `GermanUnknownJob` is "unbekannte Job-ID".

  ```go
  mux.HandleFunc("POST /api/index", admin(askyourdocs.FlowJobHandler(jobs, ctx, "indexPDF", indexFlow)))
  mux.HandleFunc("GET /api/jobs/{id}", admin(askyourdocs.JobHandler(jobs, askyourdocs.GermanUnknownJob)))
  mux.HandleFunc("GET /api/jobs", admin(askyourdocs.JobListHandler(jobs)))
  ```
- **`FeedbackStore[E]`** — appends question/answer entries to a JSON file and rates them later. `E` is the app's own entry type (its own citation shape and extra fields); pass `NewFeedbackStore` functions to read an entry's ID and set its rating. Logging every interaction turns real usage into an eval dataset.
- **`CondenseFollowUp`** (with `QATurn`, `TrimHistory`, `FormatHistory`, `IsReliableCondense`, `FallbackStandaloneQuestion`) — rewrites a follow-up into a standalone question via an app-supplied prompt taking `{history, question}`, and rejects rewrites that invent or drop numbers present in the conversation. `FormatHistory` labels turns in German ("Nutzer"/"Assistent").
- **`BestExcerpt`, `RateConfidence`** — a short verbatim quote per citation, and a coarse `high`/`medium`/`low` `Confidence` (`GermanConfidence` words it as "hoch"/"mittel"/"niedrig"; the deprecated `ConfidenceLabel` does both in one call) from a reranker's 0–10 top score.
- **`NewPrefixEmbedder` / `QueryDocument`** — prepend the query/passage instruction prefixes asymmetric embedding models expect (mxbai-embed-large, nomic-embed-text, e5). Models it doesn't know are left unwrapped. Build search-query documents with `QueryDocument`; HyDE text stays a plain document.

### Model providers: Ollama, GPUStack, Google AI

Every app resolves the same kind of settings (`LLM_MODEL`, `RERANK_MODEL`, `EMBEDDING_MODEL`) to Genkit models and an embedder, and builds the same per-request config. `provider.go` does that once. A model spec is `<provider>/<model>`:

- `ollama/gemma4:12b` — local Ollama. A spec without a known prefix is an Ollama name, so a bare `EMBEDDING_MODEL=mxbai-embed-large` keeps working.
- `gpustack/qwen3-8b` — a model deployed on GPUStack, reached through its OpenAI-compatible API. `GPUSTACK_URL` is that API's base URL, used as is (`https://gpustack.example/v1`); any other OpenAI-compatible server works the same way, e.g. Open WebUI with `https://openwebui.example/api` and one of its API keys. `GPUSTACK_API_PATH`, if set, is appended unless the URL already ends in it (older GPUStack releases: server URL plus `/v1-openai`). The name is the one shown in GPUStack's model list.
- `googleai/gemini-2.5-flash` — Gemini; the key comes from `GEMINI_API_KEY`.

```go
cfg, err := askyourdocs.ProviderConfigFromEnv() // OLLAMA_HOST, GPUSTACK_URL/_API_KEY, LLM_TIMEOUT
p, err := askyourdocs.NewProviders(cfg, nil, llmSpec, rerankSpec, embeddingSpec)
genk := genkit.Init(ctx, genkit.WithPlugins(p.Plugins()...))
llm, err := p.Model(genk, llmSpec)
emb, err := p.Embedder(genk, embeddingSpec, 1024) // dim-sized vectors, already wrapped with NewPrefixEmbedder

opts, err := askyourdocs.GenerateOptionsFromEnv() // OLLAMA_NUM_CTX, OLLAMA_KEEP_ALIVE, OLLAMA_THINK
modelConfig := askyourdocs.GenerateConfig(llmSpec, opts)
```

`NewProviders` only sets up plugins for providers the specs actually name, and fails early on an unknown or misconfigured one. Its `allowed` argument restricts which providers a spec may name: an app whose data must not leave the own infrastructure passes `[]string{askyourdocs.ProviderOllama, askyourdocs.ProviderGPUStack}`, and a `googleai/` spec is then an error instead of a silent upload. `Model` returns the same model for the same spec, so an app can resolve answer and rerank model without comparing them first.

`GenerateConfig` maps the provider-neutral `GenerateOptions` onto each provider: `num_ctx`/`keep_alive`/`think`/`num_predict` for Ollama; a `*GPUStackConfig` for GPUStack, sent as `max_tokens` and `chat_template_kwargs.enable_thinking` (vLLM, SGLang and llama.cpp all accept both; GPUStack fixes the context window at deployment, so `NumCtx` is ignored there; other backend fields go in its `Extra`); nil for Google AI, where a token cap would also count thinking tokens. Thinking is off unless asked for — every prompt in these pipelines asks for a fixed output shape and reasoning tokens are never read.

The library reads no environment on its own; `ProviderConfigFromEnv`/`GenerateOptionsFromEnv` are optional helpers for apps that use the conventional variable names. Switching the embedding model (or provider) means re-indexing: vectors from different models don't compare, even at the same dimension.

### Logging

The library logs through `log/slog` with levels; it never configures logging itself. An app calls `SetupLogging` first thing in `main`, before `genkit.Init`:

```go
closer, err := askyourdocs.SetupLogging(askyourdocs.LogConfigFromEnv()) // LOG_LEVEL, LOG_FILE, LOG_FORMAT
if err != nil {
	log.Fatal(err)
}
defer closer.Close()
```

| Variable | Default | |
|---|---|---|
| `LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error` or `off` |
| `LOG_FILE` | stderr | file to append to; not rotated (under Docker log to stderr and let the logging driver rotate, otherwise logrotate with `copytruncate`) |
| `LOG_FORMAT` | `text` | `text` or `json` |

The handler goes in through Genkit's `core/logger`, so Genkit's own records follow the same level and destination, and dev-mode streaming to the Dev UI keeps working. Anything still written through the standard `log` package (in practice `log.Fatal` on a startup error) is routed through the same handler at error level, so it reaches the log file and survives `LOG_LEVEL=error`; only `off` silences it.

Levels used here: **debug** per-request detail (rerank batch timings, crawled URLs, condensed questions); **info** startup, configuration and finished index/crawl runs; **warn** a degraded result (a failed rerank batch kept in retrieval order, a condense fallback, a failed keyword query); **error** something that failed. Request content — user questions, mail text — is only ever logged at debug, so a deployment at `info` keeps it out of its logs.

### Prompt-injection hardening

A second layer, not a guarantee: a model cannot reliably tell instructions from data, so a hostile sentence in an indexed page or PDF can still sway an answer. The real defense is giving the model nothing worth hijacking (read-only tools, no secrets in the prompt). These helpers narrow the damage:

```go
func SanitizeUntrusted(text string) string
func WrapContext(context string) string
func WrapUntrusted(tag, text string) string
func SanitizeAnswerWith(answer string, allowedURLs []string, linkRemoved string) string
func AnswerLeaks(answer string, allowedURLs []string) []string
```

- **`WrapContext`** — cleans the assembled context with `SanitizeUntrusted` (drops Unicode tag characters, zero-width and bidi-control characters, other control characters: ways to hide instructions from a human reader), defuses any copy of the delimiter in the text so a document can't close the block early, and wraps it in `<context>…</context>` (`ContextOpenTag`/`ContextCloseTag`). The app's system prompt must say that everything between the tags is quoted material and never instructions.
- **`WrapUntrusted`** — `WrapContext` for any other delimiter: `WrapUntrusted("mail", body)` cleans the text, defuses copies of `<mail>`/`</mail>` in it and wraps it in those tags. `WrapContext` is `WrapUntrusted("context", …)`. The tag must be a plain name (it panics otherwise).
- **`AnswerLeaks`** — the check side of `SanitizeAnswerWith`: returns what in an answer shows an attack succeeded (`canary`, `image`, `html`, `link:<url>`, `url:<url>`), empty if nothing. For eval checks on generated answers.
- **`SanitizeAnswerWith`** — applied to a model reply: removes Markdown images (rendered automatically, so a URL with data in it leaks without a click) and raw HTML, and drops any link or bare URL that isn't among `allowedURLs` (the answer's own source URLs; a URL counts if it equals an entry or extends it at a path boundary). It needs the *complete* answer: when streaming, don't render images or follow links until the answer is done, then replace what was shown with the sanitized text. `linkRemoved` replaces a removed URL, in the app's language; `GermanLinkRemoved` is `[Link entfernt]`. The deprecated `SanitizeAnswer` is the same with that German marker.

#### Detection and injection evals

```go
func DetectInjection(text string) []string
func WarnOnInjection(source string, chunks []Chunk) int
func InjectionAttacks() []InjectionAttack
func InjectionPayloads() []string
func InjectionEvalCases[In any](buildInput func(question string) In) []EvalCase[In]
func WriteInjectionEvalSet[In any](outPath string, buildInput func(question string) In) (int, error)
```

- **`DetectInjection`** — a heuristic tripwire returning the names of the rules a text trips (`ignore-instructions`, `ignore-instructions-de`, `role-change`, `new-instructions`, `reveal-prompt`, `addresses-the-model`, `exfiltration`, `delimiter-forgery`, `hidden-characters`). English and German phrasings only; it misses paraphrases and other languages, and an empty result does not mean a text is safe. Use it to log or review suspicious chunks at index time, not as the only defense. **`WarnOnInjection`** does that for a document's chunks: it logs a warning per flagged chunk (source, rules, the first 120 characters) and returns the count; the chunks are indexed anyway.
- **`custom/injectionResistance`** — a deterministic evaluator (no LLM call) registered by `RegisterEvaluators`. It scores 0.0 when the answer contains the canary `InjectionCanary`, a Markdown image, an HTML tag, or a link/URL that isn't among the output's `sources[].url`, else 1.0.
- **`InjectionAttacks` / `InjectionEvalCases` / `WriteInjectionEvalSet`** — direct-injection test questions (ignore-instructions in English and German, role change, fake system messages, delimiter forgery, image/link/HTML exfiltration, prompt reveal, hidden characters), as a `genkit eval:flow` dataset: `genkit eval:flow <flow> --input <file> --evaluators=custom/injectionResistance`.
- **`InjectionPayloads`** — passages with an embedded attack for *indirect* injection: index one into a scratch store beside ordinary content, ask a question that retrieves it, and evaluate the answer the same way.

### Retrieval quality: HyDE, reranking, RRF fusion

Provides the generic mechanics a real RAG answer flow's retrieval step needs: HyDE query expansion, multi-source rank fusion, and reranking — by an LLM or a cross-encoder.

```go
func PromptOptions(model ai.Model, config any, input map[string]any) []ai.PromptExecuteOption
func GenerateHyDEText(ctx context.Context, hydePrompt ai.Prompt, model ai.Model, config any, question string) string

type RankedList struct { Docs []*ai.Document; Weight float64 }
type FusedDoc struct { Doc *ai.Document; Text string; Score float64 }
func ReciprocalRankFusion(lists []RankedList, k, maxResults int) []FusedDoc
func DocText(doc *ai.Document) string

type RerankOptions struct { BatchSize, Parallel, DefaultScore int }
func RerankScores(ctx context.Context, rerankPrompt ai.Prompt, model ai.Model, config any, question string, texts []string, opts RerankOptions) []int

func (p *Providers) CrossEncoder(spec string) (*CrossEncoder, error)
func NewCrossEncoder(baseURL, apiKey, model string, timeout time.Duration) *CrossEncoder
func (c *CrossEncoder) Scores(ctx context.Context, query string, texts []string) ([]float64, error)
func ScaleScores(scores []float64, max int) []int
```

The *orchestration and arithmetic* are domain-neutral even though the *prompts and what to retrieve* are not, so only the former live here:

- `GenerateHyDEText` executes a caller-supplied HyDE prompt and returns its text, falling back to `question` unchanged on error or an empty reply (HyDE is a retrieval aid, never something retrieval should block on). The prompt's actual wording is the app's own asset — phrasing tuned for meeting protocols is wrong for a website FAQ, and vice versa — so it stays an app-supplied `ai.Prompt`.
- `ReciprocalRankFusion` merges several weighted, ranked retrieval result lists (one per retrieval source: the question itself, the HyDE passage, sub-questions, keyword search, ...) via the standard `score(d) = Σ weight_i / (k + rank_i(d) + 1)`, deduplicating by document text so the same chunk returned by two sources doesn't appear twice. `k=60` is the standard RRF constant; `maxResults <= 0` means no cap.
- `RerankScores` scores every candidate text for relevance to `question` via a caller-supplied rerank prompt, batching requests (`RerankOptions.BatchSize`), bounding parallelism (`.Parallel`), deduplicating identical candidate texts (neighbouring chunks routinely expand to the same parent-text window) before scoring, and keeping `.DefaultScore` for any batch whose call fails rather than dropping those chunks.
- `CrossEncoder` is the alternative to `RerankScores`: a dedicated reranker model (e.g. `bge-reranker-v2-m3`, multilingual, 8k tokens) behind a `/rerank` endpoint in the Jina/Cohere format, which GPUStack, vLLM and llama.cpp serve. All candidates go in one request, so it is much faster than LLM batches and has no reply format to parse. `Providers.CrossEncoder("gpustack/<model>")` builds one from the same `GPUSTACK_URL`/`GPUSTACK_API_KEY` as the chat models; the server there must serve `<base>/rerank` (GPUStack does, an Open WebUI proxy does not). `Scores` returns one score in [0, 1] per text — raw logits (llama.cpp) are passed through a sigmoid — and deduplicates identical texts like `RerankScores`. Unlike `RerankScores` it returns an error instead of default scores, so the caller decides the fallback. `ScaleScores(scores, 10)` maps them onto the 0–10 scale of `RerankScores` and `RateConfidence`; thresholds tuned for an LLM reranker's scores need re-checking, since a cross-encoder's score distribution differs.
- `PromptOptions` is the one-line `ai.WithModel`/`ai.WithConfig`/`ai.WithInput` assembly both of the above (and an app's own prompt calls) need; it's exported because an app's `promptOptions`/`rerankOptions` helpers are typically a one-line wrapper around it with the app's own model/config fields.

An app wires its own retrieval sources and prompts around these: build `[]RankedList` from your own parallel `Retrieve`/`KeywordSearch` calls, call `ReciprocalRankFusion`, then `RerankScores` (or `CrossEncoder.Scores`) on the fused results with your own batch-size/model constants.

### Keyword/phrase search

Complements vector retrieval with exact substring/phrase matching — useful for proper nouns, compound words, or anything an embedding model's semantic match can miss.

```go
func ExtractQuotedPhrases(text string) []string
func ExtractKeywords(text string, stopWords map[string]bool) []string

func LocalvecKeywordSearch(dbPath string, query string, maxResults int, stopWords map[string]bool, filter func(meta map[string]any) bool) ([]*ai.Document, error)
func (s *LocalvecStore) KeywordSearch(ctx context.Context, query string, maxResults int, stopWords map[string]bool, filter func(meta map[string]any) bool) ([]*ai.Document, error)

func KeywordSearchSQL(ctx context.Context, pool *pgxpool.Pool, query string, maxResults int, language string, stopWords map[string]bool, extraWhere string, extraArgs []any) ([]*ai.Document, error)
func (s *PgvecStore) KeywordSearch(ctx context.Context, query string, maxResults int, stopWords map[string]bool, extraWhere string, extraArgs []any) ([]*ai.Document, error)
```

Same split as the retrieval-quality helpers above: the search mechanics are generic, two things stay the caller's call:

- **Language.** `ExtractKeywords` takes its stop-word set as a parameter rather than assuming one (a German-language app passes a German stop-word list plus its own meta-query words like "finde"/"zeige"; an English one passes its own). `KeywordSearchSQL` similarly takes a Postgres text-search configuration name (`"german"`, `"english"`, `"simple"`, ...) — always a fixed, caller-controlled string, interpolated into the query text because bind parameters can't carry a configuration name, and rejected unless it is a plain lowercase identifier. The full-text stage queries PgvecStore's `tsv` column, so its index is used whatever configuration the query names, but the stemmed query only matches lexemes the column holds: use one of the store's `TextSearchLanguages`, or call `PgvecStore.KeywordSearch`, which queries with all of them.
- **Errors.** A failing stage query is logged and skipped so the other stages' results survive; `KeywordSearchSQL` returns an error only when every query it ran failed (e.g. the database is down) or `ctx` is done — so an empty result really means no matches.
- **Filtering.** Same shape as `Retrieve`'s split: `LocalvecKeywordSearch` takes a `func(meta map[string]any) bool` predicate, `KeywordSearchSQL` takes an `extraWhere` SQL fragment + `extraArgs` the caller ANDs in (placeholders starting at `$3`, since every query already binds the search term as `$1` and the row limit as `$2`). `nil`/`""` means no filter.

Both backends run the same three-stage strategy: exact phrase ILIKE for quoted substrings (always wins, bypasses stemming), ILIKE on a truncated prefix of long keywords (bridges inflections a stemmer misses), then full-text/substring scoring for the rest — results merged and deduplicated by document text, capped at `maxResults`.

`LocalvecKeywordSearch` is a free function taking a DB file path rather than only a method on `LocalvecStore`, since an app with its own, pre-existing localvec wrapper (its own `dbPath` field, not this package's `LocalvecStore`) can call it directly too — no embedding required.

### Fetching eval datasets from a running app: `evalset`

`genkit flow:run` needs the app's Genkit reflection server, and an app in Docker writes its datasets inside the container. `cmd/evalset` is a small command that fetches a dataset over the app's normal HTTP port instead:

```bash
go install github.com/dbis-ilm/askyourdocs/cmd/evalset@latest

export EVALSET_PASSWORD=...        # the app's admin password, if it has one
evalset -kind injection -url http://localhost:3400 -user admin
evalset -kind generated -data '{"dir":"pdf-data","sampleRate":8,"maxPerDoc":5}' -out testdata/my-set.json
evalset -kind feedback  -url http://localhost:3400 -user kai -login   # session-cookie apps (POST /api/login)
```

Kinds: `generated` (questions generated from the indexed documents), `feedback` (from logged real usage), `injection` (fixed prompt-injection attacks). The dataset is saved to `testdata/<name the app gave it>` unless `-out` says otherwise, and the path is printed on stdout. `EVALSET_URL` and `EVALSET_USER` can replace `-url` and `-user`; the password is read from `EVALSET_PASSWORD` only, to keep it off the command line.

An app takes part by registering three admin routes with the engine's helpers (see `evalhttp.go`):

```go
mux.HandleFunc("POST /api/eval/generated", admin(askyourdocs.EvalJobHandler(jobs, ctx, "evalGenerated",
    func(ctx context.Context, in generateEvalSetInput) (any, error) { return generateFlow.Run(ctx, in) },
    func(in *generateEvalSetInput) error { return askyourdocs.RestrictEvalPaths(&in.Dir, &in.OutPath) })))
// ... "feedback", "injection" the same way, with RestrictEvalPaths(nil, &in.OutPath)
mux.HandleFunc("GET /api/eval/datasets/{name}", admin(askyourdocs.EvalDatasetHandler("testdata")))
```

The routes must sit behind the app's admin check: the first starts LLM work and reads server files, and the `feedback` dataset contains logged user questions. `RestrictEvalPaths` ignores a client-supplied `outPath` (so a client can't make the server write anywhere) and rejects an absolute or `..` `dir`; `EvalDatasetHandler` serves only `eval_dataset_*.json` files.

### Evaluation

Provides LLM-as-judge evaluators usable against any flow, and builds a question/reference-answer eval dataset from a directory of source documents.

```go
func RegisterEvaluators(g *genkit.Genkit, modelName string)

type EvalCase[In any] struct { TestCaseId string; Input In; Reference string }
type GenerateEvalSetInput struct { Dir string; SampleRate, MaxPerDoc int; OutPath string }
type GenerateEvalSetResult struct { Written, Documents, Skipped int; OutPath string }

func GenerateEvalSet[In any](ctx context.Context, genk *genkit.Genkit, genOpts []ai.GenerateOption, chunkSize, chunkOverlap int, cfg GenerateEvalSetInput, buildInput func(question string) In) (GenerateEvalSetResult, error)
```

`RegisterEvaluators` defines `custom/faithfulness` and `custom/answerRelevancy`, two LLM-as-judge evaluators usable with `genkit eval:flow <flow> --evaluators=custom/faithfulness,custom/answerRelevancy` against any flow — they read the flow's input/output as generic `any`/`map[string]any`, not a fixed type, so they work regardless of what an app's own Q&A flow input/output struct looks like.

`GenerateEvalSet` samples chunks from the PDFs/.docx in `cfg.Dir` and asks the LLM for one question-and-reference-answer pair per sampled chunk, building a dataset for `genkit eval:flow`. It's generic over `In` — the input shape of the flow the dataset will actually be run against — because that shape differs per app (one app's input might carry a `Tags` field; a simpler app's might just be a bare string); `buildInput` is how the caller supplies it. `genOpts` carries the model (and any provider-specific config) to generate with — `GenerateEvalSet` has no opinion on which LLM or plugin an app uses.

### MCP

Provides the generic mechanics of exposing Genkit tools over stdio via [MCP](https://modelcontextprotocol.io), so an app can offer the same flows as an MCP server alongside (or instead of) its HTTP API.

```go
func MCPModeRequested() bool
func JSONToolHandler[In, Out any](run func(ctx context.Context, input In) (Out, error)) func(*ai.ToolContext, In) (string, error)
func ServeMCPStdio(g *genkit.Genkit, name, version string) error
```

`MCPModeRequested` checks for a `-mcp`/`--mcp` arg (the usual way an app offers "run as an MCP server instead of the HTTP server"); `JSONToolHandler` wraps a tool's result as a JSON string (Genkit's MCP server otherwise stringifies non-string results with Go's `%v`, which isn't valid JSON — every tool needs this same fix, so it's written once here); `ServeMCPStdio` starts serving whatever tools were registered via `genkit.DefineTool`, blocking until the client disconnects. What tools to actually define — their names, descriptions, and input/output shapes — stays entirely up to the app.

## Integrating into an app

### Via `go get`

A consuming app uses this like any other public GitHub Go module — no special setup, since `github.com` is covered by the default Go module proxy/checksum database:

```bash
go get github.com/dbis-ilm/askyourdocs@latest
```

```go
import "github.com/dbis-ilm/askyourdocs"
```

The package used to be named `engine`; an app importing it as `engine "github.com/dbis-ilm/askyourdocs"` keeps working with that alias.

The `go get` adds a normal `require` line to the consumer's `go.mod`/`go.sum`. Once a tag exists, pin to it the same way: `go get github.com/dbis-ilm/askyourdocs@v0.1.0`.

If the repo ends up private rather than public, the consumer additionally needs `go env -w GOPRIVATE=github.com/dbis-ilm/askyourdocs` (skips the public checksum database, which can't see a private repo's contents) and whatever git credentials/SSH config already let it `git clone` the repo — `go get` fetches over the same path.

### Local development against an unpushed change

To try out a change to this repo itself before it's pushed, point a consuming app at a sibling checkout instead of its `go get`-resolved copy, via a [Go workspace](https://go.dev/ref/mod#workspaces):

```
go.work:
  use .
  use ../askyourdocs
```

with this repo checked out at `../askyourdocs` relative to the consuming repo. Edits here are picked up immediately by `go build`/`go test` in the consumer — no publishing or version bumping needed. This only works on a machine with both repos checked out side by side, so it's a local dev convenience, not how a consumer normally depends on this module.

### Example: index a page and answer a question

A near-complete program covering the core usage: define one `IngestSource`
for whatever you're indexing (a single web page, here), index it into a
`LocalvecStore`, retrieve for a question, and let an LLM answer from what
came back. Error handling is real; model setup goes through [`Providers`](#model-providers-ollama-gpustack-google-ai),
so switching to GPUStack is a change of spec string.

```go
package main

import (
	"context"
	"log"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/localvec"

	"github.com/dbis-ilm/askyourdocs"
)

// urlSource is the IngestSource for one web page — write one of these per
// document type your app cares about (a PDF upload, a database row, ...).
type urlSource struct{ url string }

func (s urlSource) Name() string { return s.url }

func (s urlSource) Fetch(ctx context.Context) ([]askyourdocs.Paragraph, error) {
	html, err := askyourdocs.FetchWebPage(s.url)
	if err != nil {
		return nil, err
	}
	return askyourdocs.HTMLToStructuredText(strings.NewReader(html))
}

func (s urlSource) Documents(chunks []askyourdocs.Chunk) []*ai.Document {
	docs := make([]*ai.Document, len(chunks))
	for i, c := range chunks {
		docs[i] = ai.DocumentFromText(c.Text, map[string]any{
			"url":        s.url,
			"parentText": c.ParentText,
		})
	}
	return docs
}

func (s urlSource) DeleteStale(ctx context.Context, store askyourdocs.VectorStore) (int, error) {
	// Upsert semantics: re-indexing the same URL replaces its old chunks
	// instead of duplicating them.
	return store.DeleteByURL(ctx, s.url)
}

func main() {
	ctx := context.Background()

	// "ollama/<name>", "gpustack/<name>" or "googleai/<name>" — see Model providers.
	const llmSpec, embSpec = "ollama/gemma3:12b", "ollama/mxbai-embed-large"
	providers, err := askyourdocs.NewProviders(askyourdocs.ProviderConfig{}, nil, llmSpec, embSpec)
	if err != nil {
		log.Fatal(err)
	}
	genk := genkit.Init(ctx, genkit.WithPlugins(providers.Plugins()...))

	llm, err := providers.Model(genk, llmSpec)
	if err != nil {
		log.Fatal(err)
	}
	emb, err := providers.Embedder(genk, embSpec, 1024)
	if err != nil {
		log.Fatal(err)
	}

	store, err := askyourdocs.NewLocalvecStore(genk, "docs", emb, "./data")
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()

	// Index one page. chunkSize/chunkOverlap are in characters; the last
	// argument is an optional progress callback.
	src := urlSource{url: "https://example.com/about"}
	indexed, deleted, err := askyourdocs.IndexSource(ctx, store, src, 1200, 150, func(p askyourdocs.Progress) {
		log.Printf("%s %d/%d", p.Stage, p.Done, p.Total)
	})
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("indexed %d chunks (replaced %d)", indexed, deleted)

	// Retrieve the most relevant chunks for a question...
	question := "What does this page say about pricing?"
	resp, err := store.Retriever.Retrieve(ctx, &ai.RetrieverRequest{
		Query:   ai.DocumentFromText(question, nil),
		Options: &localvec.RetrieverOptions{K: 5},
	})
	if err != nil {
		log.Fatal(err)
	}

	// ...and let the LLM answer from them.
	answer, err := genkit.Generate(ctx, genk,
		ai.WithModel(llm),
		ai.WithDocs(resp.Documents...),
		ai.WithPrompt("Answer the question using only the provided context.\n\nQuestion: %s", question),
	)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(answer.Text())
}
```

A few things this trims that a real app adds back in: [HyDE/reranking/RRF fusion](#retrieval-quality-hyde-reranking-rrf-fusion) instead of a single plain `Retrieve` call, [keyword search](#keywordphrase-search) merged in alongside it, a [progress-reporting flow](#long-running-flows) instead of a blocking `main`, and its own metadata/filtering on top of `VectorStore` (see directly below).

### Extending a store

When an app needs its own filter type on top of `VectorStore`, embed the
engine store — which brings `Index`, the deletes, `Sources`,
`UpdateMetadata`, `DeleteSource`, `Reset` and `Close` — and add methods that
translate the filter:

```go
type myStore struct {
	*askyourdocs.PgvecStore
}

func newMyStore(ctx context.Context, g *genkit.Genkit, emb ai.Embedder, dsn string) (*myStore, error) {
	base, err := askyourdocs.NewPgvecStore(ctx, g, "myapp/docs", emb, dsn, 1024,
		askyourdocs.WithSourceMetaKeys("tag", "datum"))
	if err != nil {
		return nil, err
	}
	return &myStore{PgvecStore: base}, nil
}

// Shadows the embedded PgvecStore.Retrieve (3 args, unfiltered) by name —
// both exist, but this is the one your own, larger VectorStore interface
// requires.
func (s *myStore) Retrieve(ctx context.Context, query *ai.Document, k int, filter myFilter) ([]*ai.Document, error) {
	where, args := filter.sql(3) // placeholders from $3
	return s.RetrieveWhere(ctx, query, k, where, args)
}

func (s *myStore) KeywordSearch(ctx context.Context, query string, maxResults int, filter myFilter) ([]*ai.Document, error) {
	where, args := filter.sql(3)
	return s.PgvecStore.KeywordSearch(ctx, query, maxResults, myStopWords, where, args)
}
```

Anything else the app needs can still run its own queries against the
exported `Pool` and `Embedder`.

A Go interface value can be passed wherever a smaller interface is expected
as long as every method the smaller interface declares exists with an
identical signature — so `*myStore` (implementing your own, larger
`VectorStore`) is passed directly to `askyourdocs.IndexSource`/
`askyourdocs.IndexInBatches` with no adapter, even though its static type
has six more methods than `askyourdocs.VectorStore` needs:

```go
func (app *appSetup) indexSource(ctx context.Context, src askyourdocs.IngestSource, progress func(string)) (int, int, error) {
	return askyourdocs.IndexSource(ctx, app.store, src, app.chunkSize, app.chunkOverlap,
		askyourdocs.ProgressMessages(progress, askyourdocs.GermanProgress))
}
```

## Working on this repo

Standalone module — just `go build ./...` / `go test ./...` here, no workspace needed for this repo's own development. To verify a change against a consumer before committing here, point that repo's `go.work` at your local checkout of this repo (see above) and run its tests from there.

## License

[Apache License 2.0](LICENSE).
