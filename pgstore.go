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
	"fmt"
	"log"
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
}

// NewPgvecStore connects to PostgreSQL, ensures the schema exists for
// embeddingDim-sized vectors, and registers a Genkit retriever named name.
// If an existing documents table's embedding column has a different
// dimension (e.g. after switching embedding models), it is truncated and
// migrated automatically.
func NewPgvecStore(ctx context.Context, g *genkit.Genkit, name string, emb ai.Embedder, dsn string, embeddingDim int) (*PgvecStore, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("pgxpool.New: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pgvector ping: %w", err)
	}

	s := &PgvecStore{Pool: pool, Embedder: emb}

	if err := s.ensureSchema(ctx, embeddingDim); err != nil {
		pool.Close()
		return nil, err
	}

	s.Retriever = genkit.DefineRetriever(g, name, nil, s.retrieve)
	return s, nil
}

// ensureSchema creates the pgvector extension, documents table, and indexes.
// If the embedding column dimension does not match embeddingDim, it truncates
// the table and alters the column type automatically.
func (s *PgvecStore) ensureSchema(ctx context.Context, embeddingDim int) error {
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

CREATE INDEX IF NOT EXISTS idx_documents_fts
  ON documents USING gin (to_tsvector('german', content || ' ' || parent_text));
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
		log.Printf("[PgvecStore] embedding dim mismatch: have %s, want %s — truncating + migrating", colType, want)
		for _, stmt := range []string{
			"DROP INDEX IF EXISTS idx_documents_embedding",
			"TRUNCATE documents RESTART IDENTITY",
			fmt.Sprintf("ALTER TABLE documents ALTER COLUMN embedding TYPE %s", want),
		} {
			if _, err := s.Pool.Exec(ctx, stmt); err != nil {
				return fmt.Errorf("migrate embedding column (%s): %w", stmt, err)
			}
		}
		log.Println("[PgvecStore] embedding column migrated, all documents cleared")
	}

	// (Re-)create HNSW index — idempotent.
	const idxSQL = `CREATE INDEX IF NOT EXISTS idx_documents_embedding
	  ON documents USING hnsw (embedding vector_cosine_ops)`
	if _, err := s.Pool.Exec(ctx, idxSQL); err != nil {
		return fmt.Errorf("create HNSW index: %w", err)
	}

	// Trigram index for LIKE '%…%' lookups (e.g. an app's own KeywordSearch).
	// Without it such lookups are sequential scans; see the ask-pdf app's
	// KeywordSearch for the measured cost of skipping this.
	const trgmSQL = `CREATE INDEX IF NOT EXISTS idx_documents_trgm
	  ON documents USING gin ((lower(content || ' ' || parent_text)) gin_trgm_ops)`
	if _, err := s.Pool.Exec(ctx, trgmSQL); err != nil {
		return fmt.Errorf("create trigram index: %w", err)
	}

	log.Println("[PgvecStore] schema ensured")
	return nil
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

	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	const q = `INSERT INTO documents (content, embedding, metadata, parent_text) VALUES ($1, $2, $3, $4)`

	for i, doc := range docs {
		// Extract text content.
		var content strings.Builder
		for _, part := range doc.Content {
			content.WriteString(part.Text)
		}

		// Serialize metadata to JSON.
		metaJSON := []byte("{}")
		if doc.Metadata != nil {
			metaJSON, _ = json.Marshal(doc.Metadata)
		}

		// Extract parentText from metadata.
		parentText := ""
		if doc.Metadata != nil {
			if pt, ok := doc.Metadata["parentText"].(string); ok {
				parentText = pt
			}
		}

		vec := pgvector.NewVector(embedResp.Embeddings[i].Embedding)

		if _, err := tx.Exec(ctx, q, content.String(), vec, metaJSON, parentText); err != nil {
			return fmt.Errorf("insert doc %d: %w", i, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	log.Printf("[PgvecStore] indexed %d documents", len(docs))
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
