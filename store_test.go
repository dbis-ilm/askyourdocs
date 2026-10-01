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
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// fakeIndexStore is a minimal VectorStore for testing IndexInBatches: it
// records what Index was called with instead of actually indexing anything.
type fakeIndexStore struct {
	mu             sync.Mutex
	indexCallSizes []int
	indexedDocs    []*ai.Document
	indexErrOnCall int // 1-based; 0 means never fail
}

func (s *fakeIndexStore) Index(ctx context.Context, docs []*ai.Document) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.indexCallSizes = append(s.indexCallSizes, len(docs))
	if s.indexErrOnCall != 0 && len(s.indexCallSizes) == s.indexErrOnCall {
		return errors.New("simulated index failure")
	}
	s.indexedDocs = append(s.indexedDocs, docs...)
	return nil
}
func (s *fakeIndexStore) DeleteByURL(context.Context, string) (int, error)    { return 0, nil }
func (s *fakeIndexStore) DeleteBySource(context.Context, string) (int, error) { return 0, nil }
func (s *fakeIndexStore) Close() error                                        { return nil }

func makeTestDocs(n int) []*ai.Document {
	docs := make([]*ai.Document, n)
	for i := range docs {
		docs[i] = ai.DocumentFromText("chunk", nil)
	}
	return docs
}

func TestIndexInBatchesSplitsIntoBatchSizeGroups(t *testing.T) {
	store := &fakeIndexStore{}
	docs := makeTestDocs(2*embedBatchSize + 30) // two full batches + a partial one

	if err := IndexInBatches(context.Background(), store, docs, nil); err != nil {
		t.Fatalf("IndexInBatches: %v", err)
	}

	want := []int{embedBatchSize, embedBatchSize, 30}
	if len(store.indexCallSizes) != len(want) {
		t.Fatalf("Index called %d times with sizes %v, want %d calls", len(store.indexCallSizes), store.indexCallSizes, len(want))
	}
	for i := range want {
		if store.indexCallSizes[i] != want[i] {
			t.Errorf("call %d size = %d, want %d", i, store.indexCallSizes[i], want[i])
		}
	}
	if len(store.indexedDocs) != len(docs) {
		t.Errorf("indexed %d docs total, want %d", len(store.indexedDocs), len(docs))
	}
}

func TestIndexInBatchesSingleCallWhenUnderBatchSize(t *testing.T) {
	store := &fakeIndexStore{}
	docs := makeTestDocs(5)

	if err := IndexInBatches(context.Background(), store, docs, nil); err != nil {
		t.Fatalf("IndexInBatches: %v", err)
	}
	if len(store.indexCallSizes) != 1 || store.indexCallSizes[0] != 5 {
		t.Errorf("Index calls = %v, want a single call of 5", store.indexCallSizes)
	}
}

func TestIndexInBatchesEmptyDocsMakesNoCall(t *testing.T) {
	store := &fakeIndexStore{}
	if err := IndexInBatches(context.Background(), store, nil, nil); err != nil {
		t.Fatalf("IndexInBatches: %v", err)
	}
	if len(store.indexCallSizes) != 0 {
		t.Errorf("Index called %d times for empty input, want 0", len(store.indexCallSizes))
	}
}

func TestIndexInBatchesStopsAndReportsOnFailure(t *testing.T) {
	store := &fakeIndexStore{indexErrOnCall: 2}
	docs := makeTestDocs(2*embedBatchSize + 10)

	err := IndexInBatches(context.Background(), store, docs, nil)
	if err == nil {
		t.Fatal("expected an error when a batch fails")
	}
	if len(store.indexCallSizes) != 2 {
		t.Errorf("Index called %d times, want 2 (stopped after the failing batch)", len(store.indexCallSizes))
	}
	if len(store.indexedDocs) != embedBatchSize {
		t.Errorf("indexed %d docs, want only the first successful batch (%d)", len(store.indexedDocs), embedBatchSize)
	}
}

func TestIndexInBatchesReportsProgressPerBatch(t *testing.T) {
	store := &fakeIndexStore{}
	docs := makeTestDocs(2*embedBatchSize + 30) // 3 batches total

	type step struct{ done, total int }
	var steps []step
	err := IndexInBatches(context.Background(), store, docs, func(done, total int) {
		steps = append(steps, step{done, total})
	})
	if err != nil {
		t.Fatalf("IndexInBatches: %v", err)
	}

	want := []step{{1, 3}, {2, 3}, {3, 3}}
	if len(steps) != len(want) {
		t.Fatalf("progress calls = %v, want %v", steps, want)
	}
	for i := range want {
		if steps[i] != want[i] {
			t.Errorf("progress call %d = %+v, want %+v", i, steps[i], want[i])
		}
	}
}

func TestIndexInBatchesStopsReportingProgressAfterFailure(t *testing.T) {
	store := &fakeIndexStore{indexErrOnCall: 2}
	docs := makeTestDocs(2*embedBatchSize + 10)

	calls := 0
	_ = IndexInBatches(context.Background(), store, docs, func(done, total int) { calls++ })
	if calls != 1 {
		t.Errorf("progress called %d times, want 1 (only the successful first batch)", calls)
	}
}

func TestLocalvecStoreSourcesGroupsByURLAndSource(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "__db_test.json")
	db := map[string]any{
		"1": map[string]any{"doc": map[string]any{"metadata": map[string]any{"url": "https://a.example/"}}},
		"2": map[string]any{"doc": map[string]any{"metadata": map[string]any{"url": "https://a.example/"}}},
		"3": map[string]any{"doc": map[string]any{"metadata": map[string]any{"url": "https://b.example/"}}},
		"4": map[string]any{"doc": map[string]any{"metadata": map[string]any{"source": "a.pdf"}}},
		"5": map[string]any{"doc": map[string]any{"metadata": map[string]any{}}}, // neither url nor source: excluded
	}
	data, err := json.Marshal(db)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	s := &LocalvecStore{dbPath: path}
	sources, err := s.Sources(context.Background())
	if err != nil {
		t.Fatalf("Sources: %v", err)
	}

	byKey := map[string]int{}
	for _, src := range sources {
		byKey[src.URL+"|"+src.Source] = src.Chunks
	}
	if len(sources) != 3 {
		t.Fatalf("got %d sources, want 3: %+v", len(sources), sources)
	}
	if byKey["https://a.example/|"] != 2 {
		t.Errorf("a.example chunks = %d, want 2", byKey["https://a.example/|"])
	}
	if byKey["https://b.example/|"] != 1 {
		t.Errorf("b.example chunks = %d, want 1", byKey["https://b.example/|"])
	}
	if byKey["|a.pdf"] != 1 {
		t.Errorf("a.pdf chunks = %d, want 1", byKey["|a.pdf"])
	}
}

func TestLocalvecStoreSourcesMissingFileIsNotAnError(t *testing.T) {
	s := &LocalvecStore{dbPath: filepath.Join(t.TempDir(), "does-not-exist.json")}
	sources, err := s.Sources(context.Background())
	if err != nil {
		t.Fatalf("Sources: %v", err)
	}
	if len(sources) != 0 {
		t.Errorf("sources = %+v, want empty", sources)
	}
}
