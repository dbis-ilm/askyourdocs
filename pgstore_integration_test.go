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

package askyourdocs

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
	"fmt"
	"os"
	"reflect"
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
		// In CI a missing database must fail the run, not skip silently.
		if os.Getenv("CI") != "" {
			t.Fatal("DATABASE_URL not set, but CI is: the Postgres service is missing")
		}
		t.Skip("DATABASE_URL not set")
	}
	return dsn
}

// resetDB drops the documents table so each test starts from scratch (and
// again when it ends), and returns a pool for direct checks.
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
	// Leave no table behind: an app's own integration tests on the same
	// database would otherwise find these tests' 3-dimensional vectors and
	// be refused with ErrEmbeddingDimMismatch. Registered after pool.Close,
	// so it runs before it.
	t.Cleanup(func() {
		if _, err := pool.Exec(context.Background(), `DROP TABLE IF EXISTS documents`); err != nil {
			t.Errorf("cleanup: drop table: %v", err)
		}
	})
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

func tsvMarker(t *testing.T, pool *pgxpool.Pool) string {
	t.Helper()
	var m *string
	err := pool.QueryRow(context.Background(), `
		SELECT col_description(attrelid, attnum) FROM pg_attribute
		WHERE attrelid = 'documents'::regclass AND attname = 'tsv' AND NOT attisdropped`).Scan(&m)
	if err != nil {
		t.Fatalf("tsv column: %v", err)
	}
	if m == nil {
		return ""
	}
	return *m
}

func indexExists(t *testing.T, pool *pgxpool.Pool, name string) bool {
	t.Helper()
	var ok bool
	if err := pool.QueryRow(context.Background(),
		`SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = $1)`, name).Scan(&ok); err != nil {
		t.Fatal(err)
	}
	return ok
}

// tsvAttnum identifies the tsv column; a rebuild drops and re-adds it,
// which gives it a new attribute number.
func tsvAttnum(t *testing.T, pool *pgxpool.Pool) int {
	t.Helper()
	var n int
	if err := pool.QueryRow(context.Background(), `
		SELECT attnum FROM pg_attribute
		WHERE attrelid = 'documents'::regclass AND attname = 'tsv' AND NOT attisdropped`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// contentsOf returns the chunk texts of docs, for readable failures.
func contentsOf(docs []*ai.Document) []string {
	out := make([]string, len(docs))
	for i, d := range docs {
		out[i] = DocText(d)
	}
	return out
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
	if !reflect.DeepEqual(s.TextSearchLanguages, DefaultTextSearchLanguages) {
		t.Errorf("TextSearchLanguages = %v, want %v", s.TextSearchLanguages, DefaultTextSearchLanguages)
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

	docs, err := s.KeywordSearch(ctx, "Wann öffnet die Mensa?", 10, testStopWords, "", nil)
	if err != nil {
		t.Fatalf("KeywordSearchSQL: %v", err)
	}
	if len(docs) != 1 || !strings.Contains(DocText(docs[0]), "Mensa") || docs[0].Metadata["parentText"] != "Campus" {
		t.Fatalf("KeywordSearchSQL = %v, want the Mensa chunk with its parentText", docs)
	}

	// Long keyword: ILIKE stage plus FTS; quoted phrase: phrase stage.
	docs, err = s.KeywordSearch(ctx, `"regelt die bachelorarbeit" Prüfungsordnung`, 10, testStopWords, "", nil)
	if err != nil || len(docs) != 2 {
		t.Fatalf("KeywordSearchSQL = %d docs, %v; want 2", len(docs), err)
	}

	// extraWhere placeholders start at $3.
	// The free function still works on the same table.
	docs, err = KeywordSearchSQL(ctx, s.Pool, "Prüfungsordnung", 10, "german", testStopWords,
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

func TestIntegrationPgvecMixedLanguageCorpus(t *testing.T) {
	dsn := integrationDSN(t)
	resetDB(t, dsn)
	ctx := context.Background()

	s, err := openStore(t, dsn, 3, WithTextSearchLanguages("german", "english"))
	if err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	mustIndex(t, s,
		ai.DocumentFromText("Die Studierenden schreiben ihre Klausuren im Hörsaal.", map[string]any{"url": "https://a.example/de"}),
		ai.DocumentFromText("The students are running experiments in the laboratory.", map[string]any{"url": "https://a.example/en"}),
		ai.DocumentFromText("Im Labor laufen experiments mit neuen Geräten.", map[string]any{"url": "https://a.example/mixed"}),
	)

	for _, tc := range []struct {
		query string
		want  []string // urls, any order
	}{
		{"Klausur", []string{"https://a.example/de"}},                               // German stemming: Klausuren -> klausur
		{"run", []string{"https://a.example/en"}},                                   // English stemming: running -> run
		{"experiment", []string{"https://a.example/en", "https://a.example/mixed"}}, // English word inside a German chunk
	} {
		docs, err := s.KeywordSearch(ctx, tc.query, 10, testStopWords, "", nil)
		if err != nil {
			t.Fatalf("KeywordSearch(%q): %v", tc.query, err)
		}
		got := map[string]bool{}
		for _, d := range docs {
			got[d.Metadata["url"].(string)] = true
		}
		if len(got) != len(tc.want) {
			t.Errorf("KeywordSearch(%q) = %q, want urls %v", tc.query, contentsOf(docs), tc.want)
			continue
		}
		for _, u := range tc.want {
			if !got[u] {
				t.Errorf("KeywordSearch(%q) = %q, missing %s", tc.query, contentsOf(docs), u)
			}
		}
	}

	// The column holds both languages' lexemes, so even a query stemmed
	// with German alone finds the English word.
	docs, err := KeywordSearchSQL(ctx, s.Pool, "experiment", 10, "german", testStopWords, "", nil)
	if err != nil || len(docs) != 2 {
		t.Errorf("German-only KeywordSearchSQL(experiment) = %q, %v; want 2", contentsOf(docs), err)
	}
}

func TestIntegrationPgvecMigratesLegacyFTSIndex(t *testing.T) {
	dsn := integrationDSN(t)
	pool := resetDB(t, dsn)
	ctx := context.Background()

	// The schema an earlier version created: no tsv column, an expression
	// index for one configuration.
	for _, stmt := range []string{
		`CREATE TABLE documents (
		  id SERIAL PRIMARY KEY, content TEXT NOT NULL, embedding vector(3),
		  metadata JSONB DEFAULT '{}', parent_text TEXT DEFAULT '', created_at TIMESTAMPTZ DEFAULT NOW())`,
		`CREATE INDEX idx_documents_fts ON documents USING gin (to_tsvector('german', content || ' ' || parent_text))`,
		`INSERT INTO documents (content, embedding, metadata) VALUES ('Alte Prüfungsordnung von früher', '[1,2,3]', '{"url":"https://a.example/alt"}')`,
	} {
		if _, err := pool.Exec(ctx, stmt); err != nil {
			t.Fatalf("legacy schema: %v", err)
		}
	}

	s, err := openStore(t, dsn, 3)
	if err != nil {
		t.Fatalf("NewPgvecStore on legacy schema: %v", err)
	}
	if indexExists(t, pool, "idx_documents_fts") {
		t.Error("legacy idx_documents_fts still exists")
	}
	if !indexExists(t, pool, "idx_documents_tsv") {
		t.Error("idx_documents_tsv missing")
	}
	if m := tsvMarker(t, pool); m != "askyourdocs:fts=german" {
		t.Errorf("tsv marker = %q", m)
	}
	// The existing row was kept and its tsv computed.
	docs, err := s.KeywordSearch(ctx, "früher", 10, testStopWords, "", nil)
	if err != nil || len(docs) != 1 {
		t.Fatalf("KeywordSearch on migrated row = %q, %v; want 1", contentsOf(docs), err)
	}
}

func TestIntegrationPgvecRebuildsFTSColumnForNewLanguages(t *testing.T) {
	dsn := integrationDSN(t)
	pool := resetDB(t, dsn)
	ctx := context.Background()

	s, err := openStore(t, dsn, 3)
	if err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	mustIndex(t, s, ai.DocumentFromText("The students are running experiments.", nil))
	before := tsvAttnum(t, pool)
	// German stemming alone doesn't reduce "running" to "run".
	if docs, err := s.KeywordSearch(ctx, "run", 10, testStopWords, "", nil); err != nil || len(docs) != 0 {
		t.Fatalf("German-only KeywordSearch(run) = %q, %v; want none", contentsOf(docs), err)
	}

	s, err = openStore(t, dsn, 3, WithTextSearchLanguages("german", "english"))
	if err != nil {
		t.Fatalf("NewPgvecStore german+english: %v", err)
	}
	if m := tsvMarker(t, pool); m != "askyourdocs:fts=german,english" {
		t.Fatalf("tsv marker = %q", m)
	}
	rebuilt := tsvAttnum(t, pool)
	if rebuilt == before {
		t.Error("tsv column not rebuilt for the new language set")
	}
	// The row indexed before the change is now found via English stemming.
	if docs, err := s.KeywordSearch(ctx, "run", 10, testStopWords, "", nil); err != nil || len(docs) != 1 {
		t.Fatalf("KeywordSearch(run) after rebuild = %q, %v; want 1", contentsOf(docs), err)
	}

	// Same set (duplicates ignored): no rebuild.
	if _, err := openStore(t, dsn, 3, WithTextSearchLanguages("german", "english", "german")); err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if tsvAttnum(t, pool) != rebuilt {
		t.Error("tsv column rebuilt on reopen with the same languages")
	}

	// The planner can use the GIN index for the full-text predicate.
	conn, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Release()
	if _, err := conn.Exec(ctx, `SET enable_seqscan = off`); err != nil {
		t.Fatal(err)
	}
	rows, err := conn.Query(ctx, `EXPLAIN SELECT 1 FROM documents WHERE tsv @@ `+tsqueryExpr(s.TextSearchLanguages, "$1"), "experiment")
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
	if !strings.Contains(plan.String(), "idx_documents_tsv") {
		t.Errorf("full-text query does not use idx_documents_tsv:\n%s", plan.String())
	}
}

func TestIntegrationPgvecConcurrentStartupsMigrateOnce(t *testing.T) {
	dsn := integrationDSN(t)
	pool := resetDB(t, dsn)

	if _, err := openStore(t, dsn, 3); err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	// Several instances starting at once with a new language set must not
	// trip over each other's ALTER TABLE.
	const n = 6
	errs := make(chan error, n)
	for range n {
		go func() {
			s, err := NewPgvecStore(context.Background(), genkit.Init(context.Background()), "test/pgvec",
				dimEmbedder{3}, dsn, 3, WithTextSearchLanguages("german", "english"))
			if err == nil {
				s.Close()
			}
			errs <- err
		}()
	}
	for range n {
		if err := <-errs; err != nil {
			t.Errorf("concurrent NewPgvecStore: %v", err)
		}
	}
	if m := tsvMarker(t, pool); m != "askyourdocs:fts=german,english" {
		t.Errorf("tsv marker = %q", m)
	}
}

func TestIntegrationKeywordSearchMatchesLikeWildcardsLiterally(t *testing.T) {
	dsn := integrationDSN(t)
	resetDB(t, dsn)
	ctx := context.Background()

	s, err := openStore(t, dsn, 3)
	if err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	mustIndex(t, s,
		ai.DocumentFromText("Treffpunkt Raum A_1 im Erdgeschoss.", map[string]any{"url": "https://a.example/underscore"}),
		ai.DocumentFromText("Treffpunkt Raum AB1 im Erdgeschoss.", map[string]any{"url": "https://a.example/letter"}),
		ai.DocumentFromText("Förderung zu 100% übernommen.", map[string]any{"url": "https://a.example/percent"}),
		ai.DocumentFromText("Förderung zu 1000 Euro übernommen.", map[string]any{"url": "https://a.example/digits"}),
		ai.DocumentFromText("Die Fahrzeugübergrößen sind genehmigungspflichtig.", map[string]any{"url": "https://a.example/umlaut"}),
	)

	for _, tc := range []struct{ query, want string }{
		{`"a_1"`, "https://a.example/underscore"},  // _ would match the B; "a", "1" are too short to be keywords
		{`"zu 100%"`, "https://a.example/percent"}, // % would match "1000 Euro"
		// The long keyword's ILIKE stem must stay valid UTF-8 (a byte cut
		// split the ß, and Postgres rejected the pattern).
		{"Fahrzeugübergröße", "https://a.example/umlaut"},
	} {
		docs, err := s.KeywordSearch(ctx, tc.query, 10, testStopWords, "", nil)
		if err != nil {
			t.Fatalf("KeywordSearch(%s): %v", tc.query, err)
		}
		if len(docs) != 1 || docs[0].Metadata["url"] != tc.want {
			t.Errorf("KeywordSearch(%s) = %q, want only %s", tc.query, contentsOf(docs), tc.want)
		}
	}

	// The ILIKE stage on its own, so FTS can't hide a failing pattern.
	rows, err := s.Pool.Query(ctx, `SELECT count(*) FROM documents WHERE LOWER(content || ' ' || parent_text) LIKE $1 ESCAPE '\'`,
		"%"+escapeLike(likeStem("fahrzeugübergröße"))+"%")
	if err != nil {
		t.Fatalf("ILIKE stage query: %v", err)
	}
	defer rows.Close()
	var n int
	if !rows.Next() || rows.Scan(&n) != nil || n != 1 {
		t.Fatalf("ILIKE stage matched %d rows (err %v), want 1", n, rows.Err())
	}
}

func TestIntegrationPgvecIndexBatchIsAtomic(t *testing.T) {
	dsn := integrationDSN(t)
	pool := resetDB(t, dsn)
	ctx := context.Background()

	s, err := openStore(t, dsn, 3)
	if err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	for _, idx := range []string{"idx_documents_url", "idx_documents_source", "idx_documents_trgm", "idx_documents_tsv", "idx_documents_embedding"} {
		if !indexExists(t, pool, idx) {
			t.Errorf("index %s missing", idx)
		}
	}

	docs := make([]*ai.Document, 250)
	for i := range docs {
		docs[i] = ai.DocumentFromText(fmt.Sprintf("Abschnitt %d", i), map[string]any{"source": "gross.pdf", "parentText": "Kapitel"})
	}
	mustIndex(t, s, docs...)
	if n := countDocs(t, pool); n != 250 {
		t.Fatalf("%d documents after batch, want 250", n)
	}
	var parent string
	if err := pool.QueryRow(ctx, `SELECT parent_text FROM documents LIMIT 1`).Scan(&parent); err != nil || parent != "Kapitel" {
		t.Fatalf("parent_text = %q, %v; want Kapitel", parent, err)
	}

	// Postgres rejects a NUL byte in text: the failing row in the middle
	// of a batch must leave none of the batch behind.
	bad := []*ai.Document{
		ai.DocumentFromText("davor", nil),
		ai.DocumentFromText("kaputt\x00", nil),
		ai.DocumentFromText("danach", nil),
	}
	err = s.Index(ctx, bad)
	if err == nil || !strings.Contains(err.Error(), "insert doc 1") {
		t.Fatalf("Index with a NUL byte: err = %v, want insert doc 1 failure", err)
	}
	if n := countDocs(t, pool); n != 250 {
		t.Fatalf("%d documents after failed batch, want 250 (rolled back)", n)
	}

	if n, err := s.DeleteBySource(ctx, "gross.pdf"); err != nil || n != 250 {
		t.Fatalf("DeleteBySource = %d, %v; want 250", n, err)
	}
}

func TestIntegrationPgvecRetrieverOptions(t *testing.T) {
	dsn := integrationDSN(t)
	resetDB(t, dsn)
	ctx := context.Background()

	s, err := openStore(t, dsn, 3)
	if err != nil {
		t.Fatalf("NewPgvecStore: %v", err)
	}
	docs := make([]*ai.Document, 12)
	for i := range docs {
		docs[i] = ai.DocumentFromText(fmt.Sprintf("Dokument %d", i), nil)
	}
	mustIndex(t, s, docs...)

	query := ai.DocumentFromText("Dokument", nil)
	for _, tc := range []struct {
		name string
		opts any
		want int
	}{
		{"none", nil, 10},
		{"typed", PgvecRetrieverOptions{K: 2}, 2},
		{"pointer", &PgvecRetrieverOptions{K: 3}, 3},
		{"map from the Dev UI", map[string]any{"k": float64(4)}, 4},
		{"zero K", PgvecRetrieverOptions{}, 10},
	} {
		resp, err := s.Retriever.Retrieve(ctx, &ai.RetrieverRequest{Query: query, Options: tc.opts})
		if err != nil {
			t.Errorf("%s: Retrieve: %v", tc.name, err)
			continue
		}
		if len(resp.Documents) != tc.want {
			t.Errorf("%s: got %d documents, want %d", tc.name, len(resp.Documents), tc.want)
		}
	}

	// Another store's options type is an error, not silently the default.
	type otherOptions struct{ K int }
	if _, err := s.Retriever.Retrieve(ctx, &ai.RetrieverRequest{Query: query, Options: otherOptions{K: 1}}); err == nil {
		t.Error("Retrieve with a foreign options type: got nil error")
	}
}
