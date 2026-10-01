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
	"os"
	"path/filepath"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/localvec"
)

// Source identifies one logical indexed document — by URL (a web page) or
// Source (a named file, e.g. a PDF) — with its chunk count. It's the bare
// grouping identity every app needs for a "list what's indexed" admin view,
// before any app-specific metadata (ask-pdf's tag/datum/rank) is layered on
// top; an app wanting that layers its own, richer Sources method instead of
// using LocalvecStore's/PgvecStore's (see cmd/ask-pdf's pgvecStore).
type Source struct {
	URL    string `json:"url,omitempty"`
	Source string `json:"source,omitempty"`
	Chunks int    `json:"chunks"`
}

// VectorStore is the write-side contract IndexSource needs: persist
// documents and remove stale ones before a fresh index (upsert semantics).
// It is deliberately minimal — just indexing, not retrieval — so that an
// app wanting richer read-side behavior (filtered retrieval, keyword
// search, source administration) can define its own, larger interface:
// any concrete store satisfying that larger interface automatically
// satisfies this one too, as long as these four methods keep this exact
// signature. Retrieval itself is genkit's own ai.Retriever — LocalvecStore
// exposes one directly rather than wrapping it in something new.
type VectorStore interface {
	Index(ctx context.Context, docs []*ai.Document) error
	// DeleteByURL removes all indexed documents for the given source URL.
	// Returns the number of entries deleted.
	DeleteByURL(ctx context.Context, url string) (int, error)
	// DeleteBySource removes all indexed documents for the given document
	// name, regardless of any other metadata — the same upsert semantics
	// DeleteByURL gives web pages, so re-indexing a document replaces every
	// existing chunk for it rather than duplicating it. Returns the number
	// of entries deleted.
	DeleteBySource(ctx context.Context, sourceName string) (int, error)
	Close() error
}

// embedBatchSize caps how many documents go into one VectorStore.Index call
// (and therefore one embedding request). A single large document can
// produce well over a thousand chunks, and embedding them all in one
// request can crash a local embedding backend rather than just answering
// slowly — reproduced directly against Ollama's /api/embed, a 1469-text
// batch got back a bare EOF because the backend subprocess died mid-request.
// 200 texts per call succeeded reliably in testing; 400 already failed, so
// 100 leaves real headroom.
const embedBatchSize = 100

// IndexInBatches calls store.Index in batches of at most embedBatchSize
// documents instead of all at once. progress, if non-nil, is called after
// each successful batch with (batches done so far, total batches) — used to
// report incremental status on a long index job; pass nil when nothing is
// listening (e.g. a single web page, which rarely spans more than one
// batch).
func IndexInBatches(ctx context.Context, store VectorStore, docs []*ai.Document, progress func(done, total int)) error {
	total := (len(docs) + embedBatchSize - 1) / embedBatchSize
	done := 0
	for start := 0; start < len(docs); start += embedBatchSize {
		end := min(start+embedBatchSize, len(docs))
		if err := store.Index(ctx, docs[start:end]); err != nil {
			return fmt.Errorf("index batch %d-%d of %d: %w", start, end, len(docs), err)
		}
		done++
		if progress != nil {
			progress(done, total)
		}
	}
	return nil
}

// LocalvecStore is a ready-made VectorStore backed by genkit's localvec
// plugin — a file-based store with no external dependency (no database to
// run), suitable for a single-node app or a quick demo. Its Retriever field
// is genkit's own ai.Retriever; an app needing filtered or ranked retrieval
// builds that on top (see cmd/ask-pdf's pgvecStore, in the app that uses
// this engine, for a richer alternative implementing a larger interface
// against PostgreSQL instead).
type LocalvecStore struct {
	indexer   *localvec.DocStore
	Retriever ai.Retriever
	dbPath    string
}

// NewLocalvecStore creates a LocalvecStore. name identifies this store's
// genkit retriever/indexer action and its DB file (dir/__db_<name>.json) —
// use a different name per logical store sharing a genkit instance.
func NewLocalvecStore(g *genkit.Genkit, name string, emb ai.Embedder, dir string) (*LocalvecStore, error) {
	if err := localvec.Init(); err != nil {
		return nil, fmt.Errorf("localvec.Init: %w", err)
	}

	indexer, retriever, err := localvec.DefineRetriever(
		g, name, localvec.Config{
			Embedder: emb,
			Dir:      dir,
		},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("localvec.DefineRetriever: %w", err)
	}

	return &LocalvecStore{
		indexer:   indexer,
		Retriever: retriever,
		dbPath:    filepath.Join(dir, "__db_"+name+".json"),
	}, nil
}

func (s *LocalvecStore) Index(ctx context.Context, docs []*ai.Document) error {
	return localvec.Index(ctx, docs, s.indexer)
}

// deleteMatching removes every localvec entry whose metadata satisfies
// match. It backs DeleteByURL and DeleteBySource, which differ only in
// which metadata field they compare.
func (s *LocalvecStore) deleteMatching(match func(meta map[string]any) bool) (int, error) {
	raw, err := os.ReadFile(s.dbPath)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, fmt.Errorf("read db: %w", err)
	}

	var db map[string]json.RawMessage
	if err := json.Unmarshal(raw, &db); err != nil {
		return 0, fmt.Errorf("parse db: %w", err)
	}

	type entryMeta struct {
		Doc struct {
			Metadata map[string]any `json:"metadata"`
		} `json:"doc"`
	}
	deleted := 0
	for key, raw := range db {
		var e entryMeta
		if err := json.Unmarshal(raw, &e); err != nil {
			continue
		}
		if match(e.Doc.Metadata) {
			delete(db, key)
			deleted++
		}
	}
	if deleted == 0 {
		return 0, nil
	}

	out, err := json.Marshal(db)
	if err != nil {
		return 0, fmt.Errorf("marshal db: %w", err)
	}
	if err := os.WriteFile(s.dbPath, out, 0o644); err != nil {
		return 0, fmt.Errorf("write db: %w", err)
	}
	return deleted, nil
}

func (s *LocalvecStore) DeleteByURL(ctx context.Context, url string) (int, error) {
	return s.deleteMatching(func(meta map[string]any) bool {
		u, _ := meta["url"].(string)
		return u == url
	})
}

func (s *LocalvecStore) DeleteBySource(ctx context.Context, sourceName string) (int, error) {
	return s.deleteMatching(func(meta map[string]any) bool {
		src, _ := meta["source"].(string)
		return src == sourceName
	})
}

func (s *LocalvecStore) Close() error {
	return nil
}

// Sources returns aggregated source metadata across all stored documents,
// grouped by (url, source) alone.
func (s *LocalvecStore) Sources(ctx context.Context) ([]Source, error) {
	raw, err := os.ReadFile(s.dbPath)
	if err != nil {
		if os.IsNotExist(err) {
			return []Source{}, nil
		}
		return nil, fmt.Errorf("read db: %w", err)
	}

	var db map[string]localvecEntry
	if err := json.Unmarshal(raw, &db); err != nil {
		return nil, fmt.Errorf("parse db: %w", err)
	}

	type key struct{ url, source string }
	agg := make(map[key]*Source)
	for _, entry := range db {
		m := entry.Doc.Metadata
		if m == nil {
			continue
		}
		url, _ := m["url"].(string)
		source, _ := m["source"].(string)
		if url == "" && source == "" {
			continue
		}
		k := key{url, source}
		src, ok := agg[k]
		if !ok {
			src = &Source{URL: url, Source: source}
			agg[k] = src
		}
		src.Chunks++
	}

	sources := make([]Source, 0, len(agg))
	for _, src := range agg {
		sources = append(sources, *src)
	}
	return sources, nil
}

// localvecEntry mirrors the shape localvec writes its DB file in — just
// enough to read metadata back out.
type localvecEntry struct {
	Doc struct {
		Metadata map[string]any `json:"metadata"`
	} `json:"doc"`
}
