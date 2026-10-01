# Ask Your Documents

A small, reusable RAG engine for building "ask your documents" apps: parse documents, chunk them for retrieval, index them, and get back a Genkit retriever — without pulling in any particular app's domain (document-specific metadata, auth, admin UI, job queues).

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
grpc, no job queue). Run `go list -m all` in this repo to see that list —
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
search.go     — ExtractKeywords, ExtractQuotedPhrases, LocalvecKeywordSearch, KeywordSearchSQL
eval.go       — RegisterEvaluators: custom/faithfulness, custom/answerRelevancy
evalgen.go    — EvalCase[In], GenerateEvalSet: build a genkit eval:flow dataset from a directory of documents
mcp.go        — MCPModeRequested, JSONToolHandler, ServeMCPStdio: expose Genkit tools over stdio
```

### Text extraction & chunking

Turns a PDF, a .docx file, or a single web page into a flat list of paragraphs, then groups those paragraphs into overlapping chunks sized for embedding.

```go
type Paragraph struct { Text string; PageNum int; Heading bool; Breadcrumb string }
type Chunk struct { Text, ParentText, Pages, Section, Top, Breadcrumb, URL string }

func SupportedDocumentExt(ext string) bool
func ExtractParagraphs(path string) ([]Paragraph, error)        // .pdf or .docx
func FetchWebPage(url string) (string, error)
func HTMLToStructuredText(r io.Reader) ([]Paragraph, error)
func BuildChunks(paras []Paragraph, maxSize, overlapSize int) []Chunk
func NormalizeSpaces(s string) string
```

`Chunk.ParentText` is an expanded window around each chunk (up to
`max(maxSize*2, 3000)` chars, bounded by section headings) — index
`Chunk.Text` for embedding, but feed `Chunk.ParentText` to the LLM at answer
time for more context than a single small chunk gives you.

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

func IndexSource(ctx context.Context, store VectorStore, src IngestSource, chunkSize, chunkOverlap int, sendChunk func(string)) (indexed, deleted int, err error)
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
func NewLocalvecStore(g *genkit.Genkit, name string, emb ai.Embedder, dir string) (*LocalvecStore, error)
func NewPgvecStore(ctx context.Context, g *genkit.Genkit, name string, emb ai.Embedder, dsn string, embeddingDim int) (*PgvecStore, error)
```

`LocalvecStore` wraps Genkit's `localvec` plugin — a file-based store, no
database to run, good for a single-node app or a demo. `PgvecStore` wraps
PostgreSQL + pgvector, with schema creation and automatic migration if the
embedding dimension changes. Both expose a `Retriever ai.Retriever` field
(`PgvecStore` additionally exposes `Pool` and `Embedder`) for callers that
want to retrieve directly rather than through `IndexSource`'s write path.

`PgvecStore.Retrieve` does a plain, unfiltered similarity search. An app
wanting SQL-side filtering (date ranges, tags, ...) runs its own query
against `Pool`/`Embedder` instead — see [Extending a store](#extending-a-store)
below.

Both stores also implement:

```go
type Source struct { URL, Source string; Chunks int }

func (s *LocalvecStore) Sources(ctx context.Context) ([]Source, error)
func (s *PgvecStore) Sources(ctx context.Context) ([]Source, error)
```

— a bare "what's indexed" listing, grouped only by (url, source) with no
further metadata, good enough for a simple admin page. An app layering its
own metadata on top (tag/date/rank, say) defines its own, richer `Sources`
instead — same shadowing relationship as `Retrieve` above.

### Crawling

Crawls a multi-page site breadth-first, strips boilerplate that repeats across pages, and indexes what it finds — or replays an earlier crawl's cached pages without re-fetching.

```go
type CrawlInput struct { SeedURLs []string; MaxPages, MaxDepth, DelayMs int }
type CrawlResult struct { PagesVisited, PagesSaved, PagesIndexed int; Errors []string }
type ReindexResult struct { FilesProcessed, PagesIndexed int; Errors []string }

func RunCrawl(ctx context.Context, cfg CrawlInput, outputDir string, indexPage IndexPageFunc, progress func(string)) (CrawlResult, error)
func ReindexCrawlData(ctx context.Context, dir string, indexPage IndexPageFunc, progress func(string)) (ReindexResult, error)
```

`RunCrawl` does a breadth-first crawl from `cfg.SeedURLs` (same-host/path-prefix only), strips paragraphs that repeat across pages (boilerplate: nav, footers, repeated cards) before indexing, and — with `SAVE_CRAWL_DATA=1` in the environment — caches each cleaned page as Markdown under `outputDir`. `ReindexCrawlData` replays that cache through `indexPage` without re-fetching anything, for when a chunking fix or a changed chunk size should reach an already-crawled site. Carries no app-specific metadata itself — like `IndexSource`, that's entirely the `IndexPageFunc` closure's job.

### Long-running flows

Wraps a long-running ingest/crawl operation as a Genkit flow that reports progress as it runs, instead of making every such flow redefine that wrapping itself.

```go
func DefineProgressFlow[In, Out any](g *genkit.Genkit, name string, fn func(ctx context.Context, input In, progress func(string)) (Out, error)) *core.Flow[In, Out, string]
```

Every ingest/crawl flow (`indexPDFDocument`, `crawlWeb`, `reindexAllDocuments`, ...) is a Genkit streaming flow only so an HTTP layer can run it as a background job and surface progress messages as it goes — that wrapping (`core.StreamCallback[string]`, calling it as `sendChunk(ctx, msg)`) is identical across every one of them. `DefineProgressFlow` does that wrapping once: write `fn` against a plain `progress func(string)` and get a `*core.Flow[In, Out, string]` back, usable exactly like any other Genkit flow (`.Run` for a synchronous caller that discards the stream, `.Stream` for incremental progress). This is *not* the right shape for a flow that streams its actual output rather than progress-on-the-way-to-a-result — e.g. an answer flow streaming response text — those still call `genkit.DefineStreamingFlow` directly.

### Retrieval quality: HyDE, reranking, RRF fusion

Provides the generic mechanics a real RAG answer flow's retrieval step needs: HyDE query expansion, multi-source rank fusion, and LLM-based reranking.

```go
func PromptOptions(model ai.Model, config any, input map[string]any) []ai.PromptExecuteOption
func GenerateHyDEText(ctx context.Context, hydePrompt ai.Prompt, model ai.Model, config any, question string) string

type RankedList struct { Docs []*ai.Document; Weight float64 }
type FusedDoc struct { Doc *ai.Document; Text string; Score float64 }
func ReciprocalRankFusion(lists []RankedList, k, maxResults int) []FusedDoc
func DocText(doc *ai.Document) string

type RerankOptions struct { BatchSize, Parallel, DefaultScore int }
func RerankScores(ctx context.Context, rerankPrompt ai.Prompt, model ai.Model, config any, question string, texts []string, opts RerankOptions) []int
```

The *orchestration and arithmetic* are domain-neutral even though the *prompts and what to retrieve* are not, so only the former live here:

- `GenerateHyDEText` executes a caller-supplied HyDE prompt and returns its text, falling back to `question` unchanged on error or an empty reply (HyDE is a retrieval aid, never something retrieval should block on). The prompt's actual wording is the app's own asset — phrasing tuned for meeting protocols is wrong for a website FAQ, and vice versa — so it stays an app-supplied `ai.Prompt`.
- `ReciprocalRankFusion` merges several weighted, ranked retrieval result lists (one per retrieval source: the question itself, the HyDE passage, sub-questions, keyword search, ...) via the standard `score(d) = Σ weight_i / (k + rank_i(d) + 1)`, deduplicating by document text so the same chunk returned by two sources doesn't appear twice. `k=60` is the standard RRF constant; `maxResults <= 0` means no cap.
- `RerankScores` scores every candidate text for relevance to `question` via a caller-supplied rerank prompt, batching requests (`RerankOptions.BatchSize`), bounding parallelism (`.Parallel`), deduplicating identical candidate texts (neighbouring chunks routinely expand to the same parent-text window) before scoring, and keeping `.DefaultScore` for any batch whose call fails rather than dropping those chunks.
- `PromptOptions` is the one-line `ai.WithModel`/`ai.WithConfig`/`ai.WithInput` assembly both of the above (and an app's own prompt calls) need; it's exported because an app's `promptOptions`/`rerankOptions` helpers are typically a one-line wrapper around it with the app's own model/config fields.

An app wires its own retrieval sources and prompts around these: build `[]RankedList` from your own parallel `Retrieve`/`KeywordSearch` calls, call `ReciprocalRankFusion`, then `RerankScores` on the fused results with your own batch-size/model constants.

### Keyword/phrase search

Complements vector retrieval with exact substring/phrase matching — useful for proper nouns, compound words, or anything an embedding model's semantic match can miss.

```go
func ExtractQuotedPhrases(text string) []string
func ExtractKeywords(text string, stopWords map[string]bool) []string

func LocalvecKeywordSearch(dbPath string, query string, maxResults int, stopWords map[string]bool, filter func(meta map[string]any) bool) ([]*ai.Document, error)
func (s *LocalvecStore) KeywordSearch(ctx context.Context, query string, maxResults int, stopWords map[string]bool, filter func(meta map[string]any) bool) ([]*ai.Document, error)

func KeywordSearchSQL(ctx context.Context, pool *pgxpool.Pool, query string, maxResults int, language string, stopWords map[string]bool, extraWhere string, extraArgs []any) ([]*ai.Document, error)
```

Same split as the retrieval-quality helpers above: the search mechanics are generic, two things stay the caller's call:

- **Language.** `ExtractKeywords` takes its stop-word set as a parameter rather than assuming one (a German-language app passes a German stop-word list plus its own meta-query words like "finde"/"zeige"; an English one passes its own). `KeywordSearchSQL` similarly takes a Postgres text-search configuration name (`"german"`, `"english"`, `"simple"`, ...) — always a fixed, caller-controlled string, interpolated into the query text because bind parameters can't carry a configuration name.
- **Filtering.** Same shape as `Retrieve`'s split: `LocalvecKeywordSearch` takes a `func(meta map[string]any) bool` predicate, `KeywordSearchSQL` takes an `extraWhere` SQL fragment + `extraArgs` the caller ANDs in (placeholders starting at `$3`, since every query already binds the search term as `$1` and the row limit as `$2`). `nil`/`""` means no filter.

Both backends run the same three-stage strategy: exact phrase ILIKE for quoted substrings (always wins, bypasses stemming), ILIKE on a truncated prefix of long keywords (bridges inflections a stemmer misses), then full-text/substring scoring for the rest — results merged and deduplicated by document text, capped at `maxResults`.

`LocalvecKeywordSearch` is a free function taking a DB file path rather than only a method on `LocalvecStore`, since an app with its own, pre-existing localvec wrapper (its own `dbPath` field, not this package's `LocalvecStore`) can call it directly too — no embedding required.

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
import engine "github.com/dbis-ilm/askyourdocs" // alias: package is "engine", path's last segment isn't
```

which adds a normal `require` line to the consumer's `go.mod`/`go.sum`. Once a tag exists, pin to it the same way: `go get github.com/dbis-ilm/askyourdocs@v0.1.0`.

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
came back. Error handling is real; wiring details that don't change (the
Ollama plugin setup, matching any other Genkit app) are the only thing
trimmed.

```go
package main

import (
	"context"
	"log"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/localvec"
	"github.com/firebase/genkit/go/plugins/ollama"

	askyourdocs "github.com/dbis-ilm/askyourdocs" // alias: package is "engine", path's last segment isn't
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

	ollamaPlugin := &ollama.Ollama{ServerAddress: "http://localhost:11434"}
	genk := genkit.Init(ctx, genkit.WithPlugins(ollamaPlugin))

	if ollamaPlugin.DefineModel(genk, ollama.ModelDefinition{Name: "gemma3:12b", Type: "chat"}, nil) == nil {
		log.Fatal("failed to define model")
	}
	llm := ollama.Model(genk, "gemma3:12b")

	emb := ollamaPlugin.DefineEmbedder(genk, "mxbai-embed-large", 1024, nil)
	if emb == nil {
		log.Fatal("failed to define embedder")
	}

	store, err := askyourdocs.NewLocalvecStore(genk, "docs", emb, "./data")
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()

	// Index one page. chunkSize/chunkOverlap are in characters; the last
	// argument is an optional progress callback.
	src := urlSource{url: "https://example.com/about"}
	indexed, deleted, err := askyourdocs.IndexSource(ctx, store, src, 1200, 150, func(msg string) {
		log.Println(msg)
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

When an app needs more than `VectorStore` offers — filtered retrieval,
keyword search, retagging/deleting a named source — embed the engine store
and add those methods against its exported `Pool`/`Embedder`:

```go
type myStore struct {
	*askyourdocs.PgvecStore // Index, DeleteByURL, DeleteBySource, Close, Retriever
}

func newMyStore(ctx context.Context, g *genkit.Genkit, emb ai.Embedder, dsn string) (*myStore, error) {
	base, err := askyourdocs.NewPgvecStore(ctx, g, "myapp/docs", emb, dsn, 1024)
	if err != nil {
		return nil, err
	}
	return &myStore{PgvecStore: base}, nil
}

// Shadows the embedded PgvecStore.Retrieve (3 args, unfiltered) by name —
// both exist, but this is the one your own, larger VectorStore interface
// requires.
func (s *myStore) Retrieve(ctx context.Context, query *ai.Document, k int, filter myFilter) ([]*ai.Document, error) {
	// ... query s.Pool directly, using s.Embedder to embed the query ...
}

func (s *myStore) KeywordSearch(ctx context.Context, query string, maxResults int, filter myFilter) ([]*ai.Document, error) {
	// ... askyourdocs.KeywordSearchSQL(ctx, s.Pool, query, maxResults, "english", myStopWords, ...) ...
}
```

A Go interface value can be passed wherever a smaller interface is expected
as long as every method the smaller interface declares exists with an
identical signature — so `*myStore` (implementing your own, larger
`VectorStore`) is passed directly to `askyourdocs.IndexSource`/
`askyourdocs.IndexInBatches` with no adapter, even though its static type
has six more methods than `askyourdocs.VectorStore` needs:

```go
func (app *appSetup) indexSource(ctx context.Context, src askyourdocs.IngestSource, progress func(string)) (int, int, error) {
	return askyourdocs.IndexSource(ctx, app.store, src, app.chunkSize, app.chunkOverlap, progress)
}
```

## Working on this repo

Standalone module — just `go build ./...` / `go test ./...` here, no workspace needed for this repo's own development. To verify a change against a consumer before committing here, point that repo's `go.work` at your local checkout of this repo (see above) and run its tests from there.

## License

[Apache License 2.0](LICENSE).
