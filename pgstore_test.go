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
	"errors"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/jackc/pgx/v5/pgxpool"
)

// unreachablePool returns a pool pointing at a port nothing listens on.
// pgxpool connects lazily, so creating it succeeds and every query fails.
func unreachablePool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	pool, err := pgxpool.New(context.Background(), "postgres://test@127.0.0.1:1/test?connect_timeout=1")
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

func TestCheckTextSearchLanguage(t *testing.T) {
	for _, ok := range []string{"german", "english", "simple", "pg_catalog_x", "my_config2"} {
		if err := checkTextSearchLanguage(ok); err != nil {
			t.Errorf("checkTextSearchLanguage(%q) = %v, want nil", ok, err)
		}
	}
	for _, bad := range []string{"", "German", "german'); DROP TABLE documents; --", "pg_catalog.german", "2english", "eng lish"} {
		if err := checkTextSearchLanguage(bad); err == nil {
			t.Errorf("checkTextSearchLanguage(%q) = nil, want error", bad)
		}
	}
}

func TestNewPgvecStoreRejectsInvalidLanguageBeforeConnecting(t *testing.T) {
	// The DSN is never used: the language check comes first.
	_, err := NewPgvecStore(context.Background(), nil, "test", &recordingEmbedder{}, "postgres://unused", 3,
		WithTextSearchLanguage("x'; --"))
	if err == nil || !strings.Contains(err.Error(), "text-search configuration") {
		t.Fatalf("NewPgvecStore error = %v, want invalid text-search configuration", err)
	}
}

func TestPgvecStoreIndexRejectsEmbeddingCountMismatch(t *testing.T) {
	// recordingEmbedder returns no embeddings at all. Index must fail
	// before touching the (nil) pool instead of indexing out of range.
	s := &PgvecStore{Embedder: &recordingEmbedder{}}
	docs := []*ai.Document{ai.DocumentFromText("a", nil), ai.DocumentFromText("b", nil)}
	err := s.Index(context.Background(), docs)
	if err == nil || !strings.Contains(err.Error(), "got 0 embeddings for 2 documents") {
		t.Fatalf("Index error = %v, want embedding count mismatch", err)
	}
}

func TestKeywordSearchSQLRejectsInvalidLanguage(t *testing.T) {
	_, err := KeywordSearchSQL(context.Background(), nil, "Prüfungsordnung", 10, "german'--", testStopWords, "", nil)
	if err == nil {
		t.Fatal("KeywordSearchSQL with invalid language: got nil error")
	}
}

func TestKeywordSearchSQLReportsErrorWhenEveryQueryFails(t *testing.T) {
	// One quoted phrase, one long keyword (ILIKE + FTS) and one short
	// keyword (FTS only): every stage runs, and every one fails.
	docs, err := KeywordSearchSQL(context.Background(), unreachablePool(t),
		`"bachelor arbeit" Prüfungsordnung Mensa`, 10, "german", testStopWords, "", nil)
	if err == nil {
		t.Fatalf("KeywordSearchSQL = %d docs, nil error; want error", len(docs))
	}
	if !strings.Contains(err.Error(), "all ") || !strings.Contains(err.Error(), "queries failed") {
		t.Errorf("error = %v, want it to say all queries failed", err)
	}
	for _, stage := range []string{"phrase query", "ILIKE query", "fts query"} {
		if !strings.Contains(err.Error(), stage) {
			t.Errorf("error = %v, want it to include the %s failure", err, stage)
		}
	}
}

func TestKeywordSearchSQLStopsOnCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := KeywordSearchSQL(ctx, unreachablePool(t), "Prüfungsordnung Mensa", 10, "german", testStopWords, "", nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("KeywordSearchSQL error = %v, want context.Canceled", err)
	}
}
