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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// fakeIngestSource is an in-memory IngestSource whose Fetch/DeleteStale
// results are configurable.
type fakeIngestSource struct {
	paras         []Paragraph
	fetchErr      error
	deleteErr     error
	deleteCount   int
	deleteCalls   int
	documentsSeen int
}

func (f *fakeIngestSource) Fetch(context.Context) ([]Paragraph, error) { return f.paras, f.fetchErr }
func (f *fakeIngestSource) Documents(chunks []Chunk) []*ai.Document {
	f.documentsSeen = len(chunks)
	docs := make([]*ai.Document, len(chunks))
	for i, c := range chunks {
		docs[i] = ai.DocumentFromText(c.Text, nil)
	}
	return docs
}
func (f *fakeIngestSource) DeleteStale(context.Context, VectorStore) (int, error) {
	f.deleteCalls++
	return f.deleteCount, f.deleteErr
}
func (f *fakeIngestSource) Name() string { return "fake" }

type ingestResult struct {
	Indexed, Deleted int
	Progress         []string
}

// runIndexSource runs IndexSource inside a flow, since its steps use
// genkit.Run which needs a flow context.
func runIndexSource(t *testing.T, store VectorStore, src IngestSource) (ingestResult, error) {
	t.Helper()
	ctx := context.Background()
	g := genkit.Init(ctx)
	flow := genkit.DefineFlow(g, "ingestTest", func(ctx context.Context, _ string) (ingestResult, error) {
		var res ingestResult
		n, d, err := IndexSource(ctx, store, src, 200, 20, func(s string) { res.Progress = append(res.Progress, s) })
		res.Indexed, res.Deleted = n, d
		return res, err
	})
	return flow.Run(ctx, "")
}

func TestIndexSourceHappyPath(t *testing.T) {
	store := &fakeIndexStore{}
	src := &fakeIngestSource{
		paras:       []Paragraph{{Text: "Erster Absatz mit etwas Inhalt."}, {Text: "Zweiter Absatz mit mehr Inhalt."}},
		deleteCount: 3,
	}
	res, err := runIndexSource(t, store, src)
	if err != nil {
		t.Fatalf("IndexSource: %v", err)
	}
	if res.Indexed == 0 || res.Indexed != len(store.indexedDocs) {
		t.Errorf("indexed = %d, store has %d docs", res.Indexed, len(store.indexedDocs))
	}
	if res.Deleted != 3 {
		t.Errorf("deleted = %d, want 3", res.Deleted)
	}
	if src.deleteCalls != 1 {
		t.Errorf("DeleteStale calls = %d, want 1 (no cleanup on success)", src.deleteCalls)
	}
	if len(res.Progress) < 2 || !strings.Contains(res.Progress[0], "Chunks erzeugt") {
		t.Errorf("progress = %v, want chunk message then batch messages", res.Progress)
	}
}

func TestIndexSourceFetchErrorStopsEarly(t *testing.T) {
	store := &fakeIndexStore{}
	src := &fakeIngestSource{fetchErr: errors.New("boom")}
	if _, err := runIndexSource(t, store, src); err == nil {
		t.Fatal("expected fetch error")
	}
	if src.deleteCalls != 0 || len(store.indexCallSizes) != 0 {
		t.Errorf("nothing should be deleted or indexed after fetch failure (delete=%d index=%v)", src.deleteCalls, store.indexCallSizes)
	}
}

func TestIndexSourceDeleteStaleErrorIsNonFatal(t *testing.T) {
	store := &fakeIndexStore{}
	src := &fakeIngestSource{paras: []Paragraph{{Text: "Ein Absatz."}}, deleteErr: errors.New("nope")}
	res, err := runIndexSource(t, store, src)
	if err != nil {
		t.Fatalf("IndexSource: %v", err)
	}
	if res.Indexed == 0 {
		t.Error("indexing should proceed despite DeleteStale failure")
	}
}

func TestIndexSourceCleansUpAfterIndexFailure(t *testing.T) {
	store := &fakeIndexStore{indexErrOnCall: 1}
	src := &fakeIngestSource{paras: []Paragraph{{Text: "Ein Absatz."}}}
	if _, err := runIndexSource(t, store, src); err == nil {
		t.Fatal("expected index error")
	}
	if src.deleteCalls != 2 {
		t.Errorf("DeleteStale calls = %d, want 2 (before indexing + cleanup)", src.deleteCalls)
	}
}

func TestReindexCrawlData(t *testing.T) {
	dir := t.TempDir()
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("a.md", "---\nurl: https://a.example/\n---\n## Titel\n\nText A")
	write("b.md", "---\nurl: https://b.example/\n---\nText B")
	write("bad.md", "no frontmatter")
	write("ignored.txt", "x")
	if err := os.Mkdir(filepath.Join(dir, "sub.md"), 0o755); err != nil {
		t.Fatal(err)
	}

	got := map[string]int{}
	var lines []string
	res, err := ReindexCrawlData(context.Background(), dir, func(_ context.Context, u string, paras []Paragraph) (int, error) {
		if u == "https://b.example/" {
			return 0, errors.New("index failed")
		}
		got[u] = len(paras)
		return 1, nil
	}, func(s string) { lines = append(lines, s) })
	if err != nil {
		t.Fatalf("ReindexCrawlData: %v", err)
	}
	if res.FilesProcessed != 3 || res.PagesIndexed != 1 || len(res.Errors) != 2 {
		t.Errorf("result = %+v, want 3 files, 1 page, 2 errors", res)
	}
	if got["https://a.example/"] != 2 {
		t.Errorf("a.example paras = %d, want 2", got["https://a.example/"])
	}
	if len(lines) != 2 { // parse failure skips the progress call
		t.Errorf("progress lines = %v, want 2", lines)
	}

	// nil progress must be accepted.
	if _, err := ReindexCrawlData(context.Background(), dir, func(context.Context, string, []Paragraph) (int, error) { return 1, nil }, nil); err != nil {
		t.Errorf("nil progress: %v", err)
	}
}

func TestReindexCrawlDataMissingDir(t *testing.T) {
	_, err := ReindexCrawlData(context.Background(), filepath.Join(t.TempDir(), "nope"), nil, nil)
	if err == nil {
		t.Error("expected error for missing dir")
	}
}
