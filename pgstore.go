// Copyright 2026 Kai-Uwe Sattler
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	pgvector "github.com/pgvector/pgvector-go"
)

// PgvecStore is a ready-made VectorStore backed by PostgreSQL + pgvector —
// an alternative to LocalvecStore for an app that wants a real database
// (concurrent access, SQL-side filtering, keyword search) instead of a
// single file. Pool and Embedder are exported so an app needing more than
// the plain, unfiltered Retrieve below (e.g. filtered retrieval, keyword
// search, source administration) can run its own queries against the same
// connection and embedder — see the ask-pdf app's pgvecStore, which embeds this
// type and adds exactly that.
type PgvecStore struct {
	Pool      *pgxpool.Pool
	Embedder  ai.Embedder
	Retriever ai.Retriever

	// TextSearchLanguages are the Postgres text-search configurations
	// the tsv column is built from (see WithTextSearchLanguages).
	// KeywordSearch queries with all of them.
	TextSearchLanguages []string
}

// DefaultTextSearchLanguages are the text-search configurations
// NewPgvecStore builds its full-text column from unless
// WithTextSearchLanguages says otherwise.
var DefaultTextSearchLanguages = []string{"german"}

// ErrEmbeddingDimMismatch is returned (wrapped) by NewPgvecStore when the
// documents table already holds vectors of a different dimension and
// WithResetOnDimensionMismatch was not given.
var ErrEmbeddingDimMismatch = errors.New("embedding dimension mismatch")

// PgvecOption configures NewPgvecStore.
type PgvecOption func(*pgvecConfig)

type pgvecConfig struct {
	languages   []string
	resetOnDims bool
}

// WithTextSearchLanguages sets the Postgres text-search configurations
// ("german", "english", "simple", ...) for full-text search. Every chunk is
// stemmed with each of them into one tsvector column, so a corpus mixing
// languages — even within one document — is searchable without knowing
// any chunk's language. Each extra configuration grows the column and its
// index by roughly one more tsvector per row. If the stored column was
// built from a different set, it is rebuilt at startup, which rewrites the
// whole table.
func WithTextSearchLanguages(langs ...string) PgvecOption {
	return func(c *pgvecConfig) { c.languages = langs }
}

// WithResetOnDimensionMismatch lets NewPgvecStore delete every stored
// document when the table's embedding dimension differs from embeddingDim
// (e.g. after switching embedding models), so the column can be migrated.
// Without it, NewPgvecStore refuses with ErrEmbeddingDimMismatch instead —
// unless the table is empty, which is always migrated.
func WithResetOnDimensionMismatch() PgvecOption {
	return func(c *pgvecConfig) { c.resetOnDims = true }
}

// textSearchConfigName matches a plain, unqualified Postgres identifier —
// the only shape a text-search configuration name may have here, since it
// is interpolated into SQL text.
var textSearchConfigName = regexp.MustCompile(`^[a-z_][a-z0-9_]*$`)

func checkTextSearchLanguage(lang string) error {
	if !textSearchConfigName.MatchString(lang) {
		return fmt.Errorf("invalid text-search configuration name %q", lang)
	}
	return nil
}

// normalizeTextSearchLanguages validates langs and drops duplicates,
// keeping the first occurrence's position.
func normalizeTextSearchLanguages(langs []string) ([]string, error) {
	if len(langs) == 0 {
		return nil, errors.New("no text-search configuration given")
	}
	seen := make(map[string]bool, len(langs))
	var out []string
	for _, l := range langs {
		if err := checkTextSearchLanguage(l); err != nil {
			return nil, err
		}
		if !seen[l] {
			seen[l] = true
			out = append(out, l)
		}
	}
	return out, nil
}

// ftsText is the text the tsv column indexes.
const ftsText = `content || ' ' || coalesce(parent_text, '')`

// tsvectorExpr returns the tsv column's generation expression: one
// to_tsvector per configuration, concatenated.
func tsvectorExpr(langs []string) string {
	parts := make([]string, len(langs))
	for i, l := range langs {
		parts[i] = fmt.Sprintf("to_tsvector('%s', %s)", l, ftsText)
	}
	return strings.Join(parts, " || ")
}

// tsqueryExpr returns a tsquery matching param stemmed with any of langs.
func tsqueryExpr(langs []string, param string) string {
	parts := make([]string, len(langs))
	for i, l := range langs {
		parts[i] = fmt.Sprintf("websearch_to_tsquery('%s', %s)", l, param)
	}
	return "(" + strings.Join(parts, " || ") + ")"
}

// ftsMarker is stored as the tsv column's comment, recording which
// configurations it was built from.
func ftsMarker(langs []string) string {
	return "askyourdocs:fts=" + strings.Join(langs, ",")
}

// NewPgvecStore connects to PostgreSQL, ensures the schema exists for
// embeddingDim-sized vectors, and registers a Genkit retriever named name.
// If an existing documents table's embedding column has a different
// dimension (e.g. after switching embedding models), NewPgvecStore fails
// with ErrEmbeddingDimMismatch unless the table is empty or
// WithResetOnDimensionMismatch is given.
func NewPgvecStore(ctx context.Context, g *genkit.Genkit, name string, emb ai.Embedder, dsn string, embeddingDim int, opts ...PgvecOption) (*PgvecStore, error) {
	cfg := pgvecConfig{languages: DefaultTextSearchLanguages}
	for _, o := range opts {
		o(&cfg)
	}
	langs, err := normalizeTextSearchLanguages(cfg.languages)
	if err != nil {
		return nil, err
	}

	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("pgxpool.New: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pgvector ping: %w", err)
	}

	s := &PgvecStore{Pool: pool, Embedder: emb, TextSearchLanguages: langs}

	if err := s.ensureSchema(ctx, embeddingDim, cfg.resetOnDims); err != nil {
		pool.Close()
		return nil, err
	}

	s.Retriever = genkit.DefineRetriever(g, name, nil, s.retrieve)
	return s, nil
}

// ensureSchema creates the pgvector extension, documents table, and indexes.
// If the embedding column dimension does not match embeddingDim, it alters
// the column type — after truncating the table, which it only does when the
// table is already empty or resetOnDims is set.
func (s *PgvecStore) ensureSchema(ctx context.Context, embeddingDim int, resetOnDims bool) error {
	ddl := fmt.Sprintf(`
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
  id          SERIAL PRIMARY KEY,
  content     TEXT NOT NULL,
  embedding   vector(%d),
  metadata    JSONB DEFAULT '{}',
  parent_text TEXT DEFAULT '',
  created_at  TIMESTAMPTZ DEFAULT NOW()
);
`, embeddingDim)
	if _, err := s.Pool.Exec(ctx, ddl); err != nil {
		return fmt.Errorf("ensureSchema: %w", err)
	}

	// Detect dimension mismatch (e.g. after changing embedding model).
	var colType string
	err := s.Pool.QueryRow(ctx, `
		SELECT pg_catalog.format_type(atttypid, atttypmod)
		FROM pg_attribute
		WHERE attrelid = 'documents'::regclass
		  AND attname = 'embedding'
		  AND attnum > 0
		  AND NOT attisdropped
	`).Scan(&colType)
	if err != nil {
		return fmt.Errorf("check embedding column type: %w", err)
	}

	want := fmt.Sprintf("vector(%d)", embeddingDim)
	if colType != want {
		var hasRows bool
		if err := s.Pool.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM documents)`).Scan(&hasRows); err != nil {
			return fmt.Errorf("check documents table: %w", err)
		}
		if hasRows && !resetOnDims {
			return fmt.Errorf("%w: documents table has %s, embedder needs %s; reindex into a fresh database or pass WithResetOnDimensionMismatch to delete all stored documents",
				ErrEmbeddingDimMismatch, colType, want)
		}
		slog.WarnContext(ctx, "PgvecStore: embedding dimension mismatch, truncating and migrating", "have", colType, "want", want, "had_rows", hasRows)
		for _, stmt := range []string{
			"DROP INDEX IF EXISTS idx_documents_embedding",
			"TRUNCATE documents RESTART IDENTITY",
			fmt.Sprintf("ALTER TABLE documents ALTER COLUMN embedding TYPE %s", want),
		} {
			if _, err := s.Pool.Exec(ctx, stmt); err != nil {
				return fmt.Errorf("migrate embedding column (%s): %w", stmt, err)
			}
		}
		slog.WarnContext(ctx, "PgvecStore: embedding column migrated, all documents cleared")
	}

	// (Re-)create HNSW index — idempotent.
	const idxSQL = `CREATE INDEX IF NOT EXISTS idx_documents_embedding
	  ON documents USING hnsw (embedding vector_cosine_ops)`
	if _, err := s.Pool.Exec(ctx, idxSQL); err != nil {
		return fmt.Errorf("create HNSW index: %w", err)
	}

	if err := s.ensureFTSColumn(ctx); err != nil {
		return err
	}

	// Expression indexes for the per-document lookups: DeleteByURL,
	// DeleteBySource, and an app's own source administration, which would
	// otherwise scan the whole table.
	for _, stmt := range []string{
		`CREATE INDEX IF NOT EXISTS idx_documents_url ON documents ((metadata->>'url'))`,
		`CREATE INDEX IF NOT EXISTS idx_documents_source ON documents ((metadata->>'source'))`,
	} {
		if _, err := s.Pool.Exec(ctx, stmt); err != nil {
			return fmt.Errorf("create metadata index: %w", err)
		}
	}

	// Trigram index for LIKE '%…%' lookups (e.g. an app's own KeywordSearch).
	// Without it such lookups are sequential scans; see the ask-pdf app's
	// KeywordSearch for the measured cost of skipping this.
	const trgmSQL = `CREATE INDEX IF NOT EXISTS idx_documents_trgm
	  ON documents USING gin ((lower(content || ' ' || parent_text)) gin_trgm_ops)`
	if _, err := s.Pool.Exec(ctx, trgmSQL); err != nil {
		return fmt.Errorf("create trigram index: %w", err)
	}

	slog.DebugContext(ctx, "PgvecStore: schema ensured")
	return nil
}

// ensureFTSColumn makes sure the generated tsv column holds every chunk
// stemmed with exactly s.TextSearchLanguages, with a GIN index on it. The
// column's comment records the configurations it was built from; a
// missing or different one rebuilds column and index. It also drops the
// older per-configuration expression index idx_documents_fts.
func (s *PgvecStore) ensureFTSColumn(ctx context.Context) error {
	want := ftsMarker(s.TextSearchLanguages)

	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("full-text column: begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	// Several app instances starting at once must not migrate concurrently.
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtext('askyourdocs.documents.tsv'))`); err != nil {
		return fmt.Errorf("full-text column: lock: %w", err)
	}

	var have *string
	err = tx.QueryRow(ctx, `
		SELECT col_description(attrelid, attnum) FROM pg_attribute
		WHERE attrelid = 'documents'::regclass AND attname = 'tsv' AND NOT attisdropped`).Scan(&have)
	exists := err == nil
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("full-text column: check: %w", err)
	}

	var stmts []string
	if !exists || have == nil || *have != want {
		if exists {
			slog.WarnContext(ctx, "PgvecStore: full-text column built for other languages, rebuilding", "want", s.TextSearchLanguages)
			stmts = append(stmts, `DROP INDEX IF EXISTS idx_documents_tsv`, `ALTER TABLE documents DROP COLUMN tsv`)
		}
		stmts = append(stmts,
			fmt.Sprintf(`ALTER TABLE documents ADD COLUMN tsv tsvector GENERATED ALWAYS AS (%s) STORED`, tsvectorExpr(s.TextSearchLanguages)),
			// want consists of validated identifiers, commas and a fixed prefix.
			fmt.Sprintf(`COMMENT ON COLUMN documents.tsv IS '%s'`, want),
		)
	}
	stmts = append(stmts,
		`DROP INDEX IF EXISTS idx_documents_fts`,
		`CREATE INDEX IF NOT EXISTS idx_documents_tsv ON documents USING gin (tsv)`,
	)
	for _, stmt := range stmts {
		if _, err := tx.Exec(ctx, stmt); err != nil {
			return fmt.Errorf("full-text column (%s): %w", stmt, err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("full-text column: commit: %w", err)
	}
	return nil
}

// KeywordSearch runs KeywordSearchSQL's keyword/phrase search against this
// store, with its full-text stage matching the query stemmed with every
// configuration in TextSearchLanguages. It mirrors
// LocalvecStore.KeywordSearch, with extraWhere/extraArgs (placeholders from
// $3) in place of the filter function; an app wrapping the store with its
// own KeywordSearch shadows it, as with Retrieve.
func (s *PgvecStore) KeywordSearch(ctx context.Context, query string, maxResults int, stopWords map[string]bool, extraWhere string, extraArgs []any) ([]*ai.Document, error) {
	return keywordSearchSQL(ctx, s.Pool, query, maxResults, s.TextSearchLanguages, stopWords, extraWhere, extraArgs)
}

// Index embeds the documents and inserts them into PostgreSQL in a single
// transaction.
func (s *PgvecStore) Index(ctx context.Context, docs []*ai.Document) error {
	if len(docs) == 0 {
		return nil
	}

	// Embed all documents in one batch call.
	embedResp, err := s.Embedder.Embed(ctx, &ai.EmbedRequest{Input: docs})
	if err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	if len(embedResp.Embeddings) != len(docs) {
		return fmt.Errorf("embed: got %d embeddings for %d documents", len(embedResp.Embeddings), len(docs))
	}

	// Prepare every row first, so a document whose metadata can't be
	// serialized fails the call before anything is written.
	const q = `INSERT INTO documents (content, embedding, metadata, parent_text) VALUES ($1, $2, $3, $4)`
	batch := &pgx.Batch{}
	for i, doc := range docs {
		metaJSON := []byte("{}")
		parentText := ""
		if doc.Metadata != nil {
			b, err := json.Marshal(doc.Metadata)
			if err != nil {
				return fmt.Errorf("doc %d: marshal metadata: %w", i, err)
			}
			metaJSON = b
			if pt, ok := doc.Metadata["parentText"].(string); ok {
				parentText = pt
			}
		}
		vec := pgvector.NewVector(embedResp.Embeddings[i].Embedding)
		batch.Queue(q, DocText(doc), vec, metaJSON, parentText)
	}

	// One round trip for all rows instead of one per row, still in a single
	// transaction.
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	results := tx.SendBatch(ctx, batch)
	for i := range docs {
		if _, err := results.Exec(); err != nil {
			results.Close()
			return fmt.Errorf("insert doc %d: %w", i, err)
		}
	}
	if err := results.Close(); err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	slog.DebugContext(ctx, "PgvecStore: documents indexed", "count", len(docs))
	return nil
}

// retrieve is the Genkit retriever callback, used by the Developer UI and by
// any caller going through Retriever instead of Retrieve directly.
func (s *PgvecStore) retrieve(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
	k := 10
	if opts, ok := req.Options.(map[string]any); ok {
		if v, ok := opts["k"]; ok {
			switch kv := v.(type) {
			case float64:
				k = int(kv)
			case int:
				k = kv
			}
		}
	}
	docs, err := s.Retrieve(ctx, req.Query, k)
	if err != nil {
		return nil, err
	}
	return &ai.RetrieverResponse{Documents: docs}, nil
}

// Retrieve runs a plain vector similarity search with no filtering — an app
// needing SQL-side filtering (date ranges, tags, ...) builds that itself
// against Pool/Embedder instead (see the ask-pdf app's filtered Retrieve).
func (s *PgvecStore) Retrieve(ctx context.Context, query *ai.Document, k int) ([]*ai.Document, error) {
	if query == nil {
		return nil, fmt.Errorf("retrieve: query document is nil")
	}
	embedResp, err := s.Embedder.Embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{query},
	})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(embedResp.Embeddings) == 0 {
		return nil, fmt.Errorf("embed query: empty embeddings response")
	}
	qvec := pgvector.NewVector(embedResp.Embeddings[0].Embedding)

	const q = `
SELECT content, metadata, parent_text
FROM documents
ORDER BY embedding <=> $1
LIMIT $2`

	rows, err := s.Pool.Query(ctx, q, qvec, k)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	return ScanDocs(rows)
}

// ScanDocs reads content, metadata and parent text rows into documents. It
// expects exactly those three columns, in that order — the shape every
// query in this file and in the ask-pdf app's pgvecStore selects.
func ScanDocs(rows pgx.Rows) ([]*ai.Document, error) {
	var docs []*ai.Document
	for rows.Next() {
		var content, parentText string
		var metaJSON []byte
		if err := rows.Scan(&content, &metaJSON, &parentText); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}

		var meta map[string]any
		if err := json.Unmarshal(metaJSON, &meta); err != nil {
			meta = map[string]any{}
		}
		if parentText != "" {
			meta["parentText"] = parentText
		}

		docs = append(docs, ai.DocumentFromText(content, meta))
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("rows: %w", err)
	}
	return docs, nil
}

func (s *PgvecStore) DeleteByURL(ctx context.Context, url string) (int, error) {
	tag, err := s.Pool.Exec(ctx, "DELETE FROM documents WHERE metadata->>'url' = $1", url)
	if err != nil {
		return 0, fmt.Errorf("delete by url: %w", err)
	}
	return int(tag.RowsAffected()), nil
}

func (s *PgvecStore) DeleteBySource(ctx context.Context, sourceName string) (int, error) {
	tag, err := s.Pool.Exec(ctx, "DELETE FROM documents WHERE metadata->>'source' = $1", sourceName)
	if err != nil {
		return 0, fmt.Errorf("delete by source: %w", err)
	}
	return int(tag.RowsAffected()), nil
}

// Close releases the connection pool.
func (s *PgvecStore) Close() error {
	s.Pool.Close()
	return nil
}

// Sources returns aggregated source metadata across all stored documents,
// grouped by (url, source) alone.
func (s *PgvecStore) Sources(ctx context.Context) ([]Source, error) {
	const q = `
SELECT COALESCE(metadata->>'url', '') AS url, COALESCE(metadata->>'source', '') AS source, COUNT(*) AS chunks
FROM documents
WHERE metadata->>'url' IS NOT NULL OR metadata->>'source' IS NOT NULL
GROUP BY 1, 2
ORDER BY chunks DESC`

	rows, err := s.Pool.Query(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("sources query: %w", err)
	}
	defer rows.Close()

	sources := []Source{}
	for rows.Next() {
		var src Source
		if err := rows.Scan(&src.URL, &src.Source, &src.Chunks); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}
		sources = append(sources, src)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("rows: %w", err)
	}
	return sources, nil
}
