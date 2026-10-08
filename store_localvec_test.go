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
	"os"
	"path/filepath"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// constEmbedder returns a fixed 3-dim vector per input; ranking is irrelevant
// for the store's index/delete mechanics.
type constEmbedder struct{}

func (constEmbedder) Name() string            { return "test/const" }
func (constEmbedder) Register(r api.Registry) {}
func (constEmbedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	resp := &ai.EmbedResponse{}
	for range req.Input {
		resp.Embeddings = append(resp.Embeddings, &ai.Embedding{Embedding: []float32{1, 0, 0}})
	}
	return resp, nil
}

func TestLocalvecStoreIndexAndDelete(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	g := genkit.Init(ctx)

	s, err := NewLocalvecStore(g, "store_test", constEmbedder{}, dir)
	if err != nil {
		t.Fatalf("NewLocalvecStore: %v", err)
	}
	defer s.Close()

	// Nothing indexed yet: deletes are a no-op, not an error.
	if n, err := s.DeleteByURL(ctx, "https://a.example/"); err != nil || n != 0 {
		t.Fatalf("DeleteByURL on empty store = %d, %v; want 0, nil", n, err)
	}

	docs := []*ai.Document{
		ai.DocumentFromText("eins", map[string]any{"url": "https://a.example/"}),
		ai.DocumentFromText("zwei", map[string]any{"url": "https://a.example/"}),
		ai.DocumentFromText("drei", map[string]any{"url": "https://b.example/"}),
		ai.DocumentFromText("vier", map[string]any{"source": "x.pdf"}),
	}
	if err := s.Index(ctx, docs); err != nil {
		t.Fatalf("Index: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "__db_store_test.json")); err != nil {
		t.Fatalf("db file not written: %v", err)
	}

	sources, err := s.Sources(ctx)
	if err != nil || len(sources) != 3 {
		t.Fatalf("Sources = %+v, %v; want 3 sources", sources, err)
	}

	if n, err := s.DeleteByURL(ctx, "https://a.example/"); err != nil || n != 2 {
		t.Errorf("DeleteByURL = %d, %v; want 2, nil", n, err)
	}
	if n, err := s.DeleteBySource(ctx, "x.pdf"); err != nil || n != 1 {
		t.Errorf("DeleteBySource = %d, %v; want 1, nil", n, err)
	}
	// Second delete of the same key finds nothing.
	if n, err := s.DeleteBySource(ctx, "x.pdf"); err != nil || n != 0 {
		t.Errorf("repeat DeleteBySource = %d, %v; want 0, nil", n, err)
	}

	sources, err = s.Sources(ctx)
	if err != nil || len(sources) != 1 || sources[0].URL != "https://b.example/" {
		t.Errorf("Sources after deletes = %+v, %v; want only b.example", sources, err)
	}
}

func TestLocalvecStoreDeleteCorruptDB(t *testing.T) {
	path := filepath.Join(t.TempDir(), "db.json")
	if err := os.WriteFile(path, []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &LocalvecStore{dbPath: path}
	if _, err := s.DeleteByURL(context.Background(), "u"); err == nil {
		t.Error("expected parse error for corrupt db")
	}
}
