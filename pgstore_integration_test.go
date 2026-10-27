//go:build integration

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

// Integration tests for PgvecStore and KeywordSearchSQL against a real
// PostgreSQL with pgvector. Run with
//
//	DATABASE_URL=postgres://... go test -tags integration -run Integration ./...
//
// They drop and recreate the documents table, so point DATABASE_URL at a
// throwaway database only.

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/jackc/pgx/v5/pgxpool"
)

// dimEmbedder returns a fixed-size vector per input, derived from its text,
// so documents get distinct but deterministic embeddings.
type dimEmbedder struct{ dim int }

func (e dimEmbedder) Name() string            { return "test/dim" }
func (e dimEmbedder) Register(r api.Registry) {}

func (e dimEmbedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	resp := &ai.EmbedResponse{}
	for _, doc := range req.Input {
		vec := make([]float32, e.dim)
		for i, b := range []byte(DocText(doc)) {
			vec[i%e.dim] += float32(b)
		}
		vec[0]++ // never the zero vector, which has no cosine distance
		resp.Embeddings = append(resp.Embeddings, &ai.Embedding{Embedding: vec})
	}
	return resp, nil
}

func integrationDSN(t *testing.T) string {
	t.Helper()
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		t.Skip("DATABASE_URL not set")
	}
	return dsn
}

// resetDB drops the documents table so each test starts from scratch, and
// returns a pool for direct checks.
func resetDB(t *testing.T, dsn string) *pgxpool.Pool {
	t.Helper()
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)
	// A freshly started container can refuse connections for a moment.
	deadline := time.Now().Add(15 * time.Second)
	for {
		if err = pool.Ping(ctx); err == nil || time.Now().After(deadline) {
			break
		}
		time.Sleep(300 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("ping: %v", err)
	}
	if _, err := pool.Exec(ctx, `DROP TABLE IF EXISTS documents`); err != nil {
		t.Fatalf("drop table: %v", err)
	}
	return pool
}

// openStore opens a PgvecStore on its own Genkit instance, so each call can
// register a retriever without name clashes.
func openStore(t *testing.T, dsn string, dim int, opts ...PgvecOption) (*PgvecStore, error) {
	t.Helper()
	s, err := NewPgvecStore(context.Background(), genkit.Init(context.Background()), "test/pgvec", dimEmbedder{dim}, dsn, dim, opts...)
	if err == nil {
		t.Cleanup(func() { s.Close() })
	}
	return s, err
}

func countDocs(t *testing.T, pool *pgxpool.Pool) int {
	t.Helper()
	var n int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM documents`).Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	return n
}

func columnType(t *testing.T, pool *pgxpool.Pool) string {
	t.Helper()
	var typ string
	err := pool.QueryRow(context.Background(), `
		SELECT format_type(atttypid, atttypmod) FROM pg_attribute
		WHERE attrelid = 'documents'::regclass AND attname = 'embedding'`).Scan(&typ)
	if err != nil {
		t.Fatalf("column type: %v", err)
	}
	return typ
}

func ftsIndexDef(t *testing.T, pool *pgxpool.Pool) string {
	t.Helper()
	var def string
	err := pool.QueryRow(context.Background(),
		`SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_documents_fts'`).Scan(&def)
	if err != nil {
		t.Fatalf("fts index: %v", err)
	}
	return def
}

func mustIndex(t *testing.T, s *PgvecStore, docs ...*ai.Document) {
	t.Helper()
	if err := s.Index(context.Background(), docs); err != nil {
		t.Fatalf("Index: %v", err)
	}
}

func TestIntegrationPgvecIndexRetrieveSearchDelete(t *testing.T) {
	dsn := integrationDSN(t)
	pool := resetDB(t, dsn)
	ctx := context.Background()

	s, err := openStore(t, dsn, 3)
	if err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	if s.TextSearchLanguage != DefaultTextSearchLanguage {
		t.Errorf("TextSearchLanguage = %q, want %q", s.TextSearchLanguage, DefaultTextSearchLanguage)
	}
	mustIndex(t, s,
		ai.DocumentFromText("Die Prüfungsordnung regelt die Bachelorarbeit.", map[string]any{"url": "https://a.example/po"}),
		ai.DocumentFromText("Die Mensa öffnet um elf Uhr.", map[string]any{"url": "https://a.example/mensa", "parentText": "Campus"}),
		ai.DocumentFromText("Anhang zur Prüfungsordnung.", map[string]any{"source": "anhang.pdf"}),
	)

	got, err := s.Retrieve(ctx, ai.DocumentFromText("Mensa", nil), 2)
	if err != nil || len(got) != 2 {
		t.Fatalf("Retrieve = %d docs, %v; want 2", len(got), err)
	}

	docs, err := KeywordSearchSQL(ctx, s.Pool, "Wann öffnet die Mensa?", 10, s.TextSearchLanguage, testStopWords, "", nil)
	if err != nil {
		t.Fatalf("KeywordSearchSQL: %v", err)
	}
	if len(docs) != 1 || !strings.Contains(DocText(docs[0]), "Mensa") || docs[0].Metadata["parentText"] != "Campus" {
		t.Fatalf("KeywordSearchSQL = %v, want the Mensa chunk with its parentText", docs)
	}

	// Long keyword: ILIKE stage plus FTS; quoted phrase: phrase stage.
	docs, err = KeywordSearchSQL(ctx, s.Pool, `"regelt die bachelorarbeit" Prüfungsordnung`, 10, s.TextSearchLanguage, testStopWords, "", nil)
	if err != nil || len(docs) != 2 {
		t.Fatalf("KeywordSearchSQL = %d docs, %v; want 2", len(docs), err)
	}

	// extraWhere placeholders start at $3.
	docs, err = KeywordSearchSQL(ctx, s.Pool, "Prüfungsordnung", 10, s.TextSearchLanguage, testStopWords,
		`metadata->>'source' = $3`, []any{"anhang.pdf"})
	if err != nil || len(docs) != 1 || docs[0].Metadata["source"] != "anhang.pdf" {
		t.Fatalf("filtered KeywordSearchSQL = %v, %v; want only anhang.pdf", docs, err)
	}

	srcs, err := s.Sources(ctx)
	if err != nil || len(srcs) != 3 {
		t.Fatalf("Sources = %v, %v; want 3", srcs, err)
	}
	if n, err := s.DeleteByURL(ctx, "https://a.example/mensa"); err != nil || n != 1 {
		t.Fatalf("DeleteByURL = %d, %v; want 1", n, err)
	}
	if n, err := s.DeleteBySource(ctx, "anhang.pdf"); err != nil || n != 1 {
		t.Fatalf("DeleteBySource = %d, %v; want 1", n, err)
	}
	if n := countDocs(t, pool); n != 1 {
		t.Fatalf("%d documents left, want 1", n)
	}
}

func TestIntegrationPgvecDimensionMismatchKeepsDataByDefault(t *testing.T) {
	dsn := integrationDSN(t)
	pool := resetDB(t, dsn)

	s, err := openStore(t, dsn, 3)
	if err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	mustIndex(t, s, ai.DocumentFromText("bleibt erhalten", nil))

	if _, err := openStore(t, dsn, 4); !errors.Is(err, ErrEmbeddingDimMismatch) {
		t.Fatalf("NewPgvecStore with new dimension: err = %v, want ErrEmbeddingDimMismatch", err)
	}
	if n := countDocs(t, pool); n != 1 {
		t.Fatalf("%d documents after refused migration, want 1", n)
	}
	if typ := columnType(t, pool); typ != "vector(3)" {
		t.Fatalf("column type = %s, want vector(3)", typ)
	}

	if _, err := openStore(t, dsn, 4, WithResetOnDimensionMismatch()); err != nil {
		t.Fatalf("NewPgvecStore with reset option: %v", err)
	}
	if n := countDocs(t, pool); n != 0 {
		t.Fatalf("%d documents after reset, want 0", n)
	}
	if typ := columnType(t, pool); typ != "vector(4)" {
		t.Fatalf("column type = %s, want vector(4)", typ)
	}
}

func TestIntegrationPgvecDimensionMismatchMigratesEmptyTable(t *testing.T) {
	dsn := integrationDSN(t)
	pool := resetDB(t, dsn)

	if _, err := openStore(t, dsn, 3); err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	s, err := openStore(t, dsn, 4)
	if err != nil {
		t.Fatalf("NewPgvecStore on empty table with new dimension: %v", err)
	}
	if typ := columnType(t, pool); typ != "vector(4)" {
		t.Fatalf("column type = %s, want vector(4)", typ)
	}
	mustIndex(t, s, ai.DocumentFromText("passt jetzt", nil))
}

func TestIntegrationPgvecRebuildsFTSIndexForNewLanguage(t *testing.T) {
	dsn := integrationDSN(t)
	pool := resetDB(t, dsn)
	ctx := context.Background()

	if _, err := openStore(t, dsn, 3); err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	if def := ftsIndexDef(t, pool); !strings.Contains(def, "'german'::regconfig") {
		t.Fatalf("fts index = %s, want german", def)
	}

	s, err := openStore(t, dsn, 3, WithTextSearchLanguage("english"))
	if err != nil {
		t.Fatalf("NewPgvecStore english: %v", err)
	}
	if def := ftsIndexDef(t, pool); !strings.Contains(def, "'english'::regconfig") {
		t.Fatalf("fts index = %s, want english", def)
	}

	// Reopening with the same language leaves the index alone.
	var oid1, oid2 uint32
	oidQ := `SELECT 'idx_documents_fts'::regclass::oid`
	if err := pool.QueryRow(ctx, oidQ).Scan(&oid1); err != nil {
		t.Fatal(err)
	}
	if _, err := openStore(t, dsn, 3, WithTextSearchLanguage("english")); err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if err := pool.QueryRow(ctx, oidQ).Scan(&oid2); err != nil {
		t.Fatal(err)
	}
	if oid1 != oid2 {
		t.Errorf("fts index rebuilt on reopen with the same language")
	}

	// The planner can use the index for KeywordSearchSQL's FTS predicate.
	conn, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Release()
	if _, err := conn.Exec(ctx, `SET enable_seqscan = off`); err != nil {
		t.Fatal(err)
	}
	rows, err := conn.Query(ctx, `EXPLAIN SELECT 1 FROM documents
		WHERE to_tsvector('`+s.TextSearchLanguage+`', content || ' ' || parent_text) @@ websearch_to_tsquery('`+s.TextSearchLanguage+`', $1)`, "library")
	if err != nil {
		t.Fatal(err)
	}
	var plan strings.Builder
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			t.Fatal(err)
		}
		plan.WriteString(line + "\n")
	}
	rows.Close()
	if !strings.Contains(plan.String(), "idx_documents_fts") {
		t.Errorf("FTS query does not use idx_documents_fts:\n%s", plan.String())
	}
}
