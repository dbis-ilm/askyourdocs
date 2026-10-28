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
// Source (a named file, e.g. a PDF) — with its chunk count. It's the
// grouping identity every app needs for a "list what's indexed" admin view.
// Metadata holds the descriptive fields a store was told to group by (see
// WithSourceMetaKeys and WithLocalvecSourceMetaKeys): chunks of one file
// that differ in, say, their "tag" are two sources until UpdateMetadata
// gives them the same one. A key missing from Metadata means the chunks
// don't carry it. In JSON, Metadata entries are fields of their own next
// to url, source and chunks.
type Source struct {
	URL      string
	Source   string
	Metadata map[string]string
	Chunks   int
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
// is genkit's own ai.Retriever; Retrieve, RetrieveFiltered and
// KeywordSearch cover the usual reads, and Sources, UpdateMetadata,
// DeleteSource and Reset the administration of what is indexed.
type LocalvecStore struct {
	indexer        *localvec.DocStore
	Retriever      ai.Retriever
	dbPath         string
	sourceMetaKeys []string
}

// LocalvecOption configures NewLocalvecStore.
type LocalvecOption func(*localvecConfig)

type localvecConfig struct {
	sourceMetaKeys []string
}

// WithLocalvecSourceMetaKeys is WithSourceMetaKeys for a LocalvecStore.
func WithLocalvecSourceMetaKeys(keys ...string) LocalvecOption {
	return func(c *localvecConfig) { c.sourceMetaKeys = keys }
}

// NewLocalvecStore creates a LocalvecStore. name identifies this store's
// genkit retriever/indexer action and its DB file (dir/__db_<name>.json) —
// use a different name per logical store sharing a genkit instance.
func NewLocalvecStore(g *genkit.Genkit, name string, emb ai.Embedder, dir string, opts ...LocalvecOption) (*LocalvecStore, error) {
	var cfg localvecConfig
	for _, o := range opts {
		o(&cfg)
	}
	if err := checkSourceMetaKeys(cfg.sourceMetaKeys); err != nil {
		return nil, err
	}
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
		indexer:        indexer,
		Retriever:      retriever,
		dbPath:         filepath.Join(dir, "__db_"+name+".json"),
		sourceMetaKeys: cfg.sourceMetaKeys,
	}, nil
}

func (s *LocalvecStore) Index(ctx context.Context, docs []*ai.Document) error {
	return localvec.Index(ctx, docs, s.indexer)
}

// deleteMatching removes every localvec entry whose metadata satisfies
// match. It backs DeleteByURL, DeleteBySource and DeleteSource, which
// differ only in which metadata fields they compare.
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

// DeleteSource removes exactly the chunks of src, compared on url, source
// and every key the store groups by — one entry of a Sources listing,
// without touching chunks of the same file that carry other metadata.
// Returns how many chunks were deleted.
func (s *LocalvecStore) DeleteSource(ctx context.Context, src Source) (int, error) {
	return s.deleteMatching(func(meta map[string]any) bool {
		return inSource(meta, src, s.sourceMetaKeys)
	})
}

// Reset deletes every indexed document by removing the DB file.
func (s *LocalvecStore) Reset(ctx context.Context) error {
	if err := os.Remove(s.dbPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove %s: %w", s.dbPath, err)
	}
	return nil
}

// Retrieve returns the k documents most similar to query.
func (s *LocalvecStore) Retrieve(ctx context.Context, query *ai.Document, k int) ([]*ai.Document, error) {
	return s.RetrieveFiltered(ctx, query, k, nil)
}

// localvecFilterFactor is how many times k RetrieveFiltered fetches when
// filtering: localvec ranks before the filter can see any metadata, so the
// filter needs candidates to choose from.
const localvecFilterFactor = 10

// RetrieveFiltered returns up to k documents similar to query whose
// metadata satisfies filter (nil keeps all). localvec cannot filter while
// ranking, so it fetches 10·k candidates and filters those: a filter that
// matches only a small part of the store can return fewer than k.
func (s *LocalvecStore) RetrieveFiltered(ctx context.Context, query *ai.Document, k int, filter func(meta map[string]any) bool) ([]*ai.Document, error) {
	want := k
	if filter != nil {
		want = k * localvecFilterFactor
	}
	// localvec reads k from *localvec.RetrieverOptions and silently falls
	// back to 3 for anything else.
	resp, err := s.Retriever.Retrieve(ctx, &ai.RetrieverRequest{
		Query:   query,
		Options: &localvec.RetrieverOptions{K: want},
	})
	if err != nil {
		return nil, err
	}
	if filter == nil {
		return resp.Documents, nil
	}
	docs := make([]*ai.Document, 0, k)
	for _, doc := range resp.Documents {
		if !filter(doc.Metadata) {
			continue
		}
		docs = append(docs, doc)
		if len(docs) >= k {
			break
		}
	}
	return docs, nil
}

// Sources returns one entry per indexed document, grouped by url, source
// and the store's source metadata keys, with its chunk count.
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

	agg := make(map[string]*Source)
	for _, entry := range db {
		src, ok := sourceOf(entry.Doc.Metadata, s.sourceMetaKeys)
		if !ok {
			continue
		}
		k := sourceGroupKey(src, s.sourceMetaKeys)
		if agg[k] == nil {
			agg[k] = &src
		}
		agg[k].Chunks++
	}

	sources := make([]Source, 0, len(agg))
	for _, src := range agg {
		sources = append(sources, *src)
	}
	return sources, nil
}

// UpdateMetadata sets the source metadata keys of every chunk of old
// (matched like DeleteSource) to newMetadata. A key missing or empty in
// newMetadata is removed from the chunks instead of stored as "", so they
// look like chunks that never had it. Other metadata is left alone, and
// keys the store doesn't group by are ignored in newMetadata. Returns how
// many chunks were updated.
func (s *LocalvecStore) UpdateMetadata(ctx context.Context, old Source, newMetadata map[string]string) (int, error) {
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

	// Entries are edited as raw JSON so that every field this code doesn't
	// know about (the embedding, the content parts) is written back as is.
	updated := 0
	for key, entryRaw := range db {
		var entry map[string]json.RawMessage
		if err := json.Unmarshal(entryRaw, &entry); err != nil {
			continue
		}
		var doc map[string]json.RawMessage
		if err := json.Unmarshal(entry["doc"], &doc); err != nil {
			continue
		}
		var meta map[string]any
		if metaRaw, ok := doc["metadata"]; ok {
			_ = json.Unmarshal(metaRaw, &meta)
		}
		if meta == nil {
			meta = map[string]any{}
		}
		if !inSource(meta, old, s.sourceMetaKeys) {
			continue
		}
		for _, k := range s.sourceMetaKeys {
			if v := newMetadata[k]; v == "" {
				delete(meta, k)
			} else {
				meta[k] = v
			}
		}

		if doc["metadata"], err = json.Marshal(meta); err != nil {
			return 0, fmt.Errorf("marshal metadata for %s: %w", key, err)
		}
		if entry["doc"], err = json.Marshal(doc); err != nil {
			return 0, fmt.Errorf("marshal doc for %s: %w", key, err)
		}
		if db[key], err = json.Marshal(entry); err != nil {
			return 0, fmt.Errorf("marshal entry for %s: %w", key, err)
		}
		updated++
	}
	if updated == 0 {
		return 0, nil
	}

	out, err := json.Marshal(db)
	if err != nil {
		return 0, fmt.Errorf("marshal db: %w", err)
	}
	if err := os.WriteFile(s.dbPath, out, 0o644); err != nil {
		return 0, fmt.Errorf("write db: %w", err)
	}
	return updated, nil
}

// localvecEntry mirrors the shape localvec writes its DB file in — just
// enough to read metadata back out.
type localvecEntry struct {
	Doc struct {
		Metadata map[string]any `json:"metadata"`
	} `json:"doc"`
}
