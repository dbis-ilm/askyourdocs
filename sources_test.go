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
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func TestSourceJSONIsFlat(t *testing.T) {
	src := Source{Source: "a.pdf", Metadata: map[string]string{"tag": "Senat", "rank": ""}, Chunks: 3}
	data, err := json.Marshal(src)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(data), `{"chunks":3,"source":"a.pdf","tag":"Senat"}`; got != want {
		t.Errorf("Marshal = %s, want %s", got, want)
	}

	var back Source
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	want := Source{Source: "a.pdf", Metadata: map[string]string{"tag": "Senat"}, Chunks: 3}
	if !reflect.DeepEqual(back, want) {
		t.Errorf("round trip = %+v, want %+v", back, want)
	}
}

func TestSourceJSONWithoutMetadata(t *testing.T) {
	data, _ := json.Marshal(Source{URL: "https://a.example/", Chunks: 1})
	if got, want := string(data), `{"chunks":1,"url":"https://a.example/"}`; got != want {
		t.Errorf("Marshal = %s, want %s", got, want)
	}
	var back Source
	if err := json.Unmarshal([]byte(`{"url":"https://a.example/","tag":null,"datum":""}`), &back); err != nil {
		t.Fatal(err)
	}
	if back.Metadata != nil {
		t.Errorf("Metadata = %v, want nil for null and empty fields", back.Metadata)
	}
}

func TestSourceUnmarshalRejectsNonStringMetadata(t *testing.T) {
	var s Source
	err := json.Unmarshal([]byte(`{"source":"a.pdf","tag":5}`), &s)
	if err == nil || !strings.Contains(err.Error(), "source.tag") {
		t.Fatalf("err = %v, want an error naming source.tag", err)
	}
}

func TestCheckSourceMetaKeys(t *testing.T) {
	if err := checkSourceMetaKeys([]string{"tag", "datum", "Rank_2"}); err != nil {
		t.Errorf("valid keys: %v", err)
	}
	for _, keys := range [][]string{{"x'; --"}, {""}, {"url"}, {"chunks"}, {"tag", "tag"}, {"2tag"}} {
		if err := checkSourceMetaKeys(keys); err == nil {
			t.Errorf("checkSourceMetaKeys(%q) = nil, want an error", keys)
		}
	}
}

func TestNewLocalvecStoreRejectsInvalidSourceMetaKeys(t *testing.T) {
	_, err := NewLocalvecStore(genkit.Init(context.Background()), "bad", constEmbedder{}, t.TempDir(),
		WithLocalvecSourceMetaKeys("source"))
	if err == nil || !strings.Contains(err.Error(), "reserved") {
		t.Fatalf("err = %v, want reserved key", err)
	}
}

// writeSourcesDB writes a localvec DB file with one entry per key of
// entries, holding that metadata, and returns its path.
func writeSourcesDB(t *testing.T, entries map[string]map[string]any) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "__db_test.json")
	db := make(map[string]any, len(entries))
	for key, meta := range entries {
		db[key] = map[string]any{
			"doc": map[string]any{
				"content":  []map[string]any{{"text": "chunk " + key}},
				"metadata": meta,
			},
			"embedding": []float32{1, 2, 3},
		}
	}
	data, err := json.Marshal(db)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// readLocalvecDB returns every entry of the DB file at path, as raw JSON.
func readLocalvecDB(t *testing.T, path string) map[string]map[string]any {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var db map[string]map[string]any
	if err := json.Unmarshal(raw, &db); err != nil {
		t.Fatal(err)
	}
	return db
}

func entryMeta(t *testing.T, db map[string]map[string]any, key string) map[string]any {
	t.Helper()
	e, ok := db[key]
	if !ok {
		t.Fatalf("entry %q missing", key)
	}
	meta, _ := e["doc"].(map[string]any)["metadata"].(map[string]any)
	return meta
}

func sortSources(s []Source) {
	sort.Slice(s, func(i, j int) bool {
		return sourceGroupKey(s[i], []string{"tag", "datum"}) < sourceGroupKey(s[j], []string{"tag", "datum"})
	})
}

func TestLocalvecStoreSourcesGroupsByMetaKeys(t *testing.T) {
	path := writeSourcesDB(t, map[string]map[string]any{
		"1": {"source": "a.pdf", "tag": "Senat", "datum": "2026-01-13", "page": "1"},
		"2": {"source": "a.pdf", "tag": "Senat", "datum": "2026-01-13", "page": "2"},
		"3": {"source": "a.pdf", "tag": "Ordnung"},
		"4": {"tag": "Lose"}, // no url or source, but a grouping key
		"5": {"page": "9"},   // nothing to group by
	})
	s := &LocalvecStore{dbPath: path, sourceMetaKeys: []string{"tag", "datum"}}
	got, err := s.Sources(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	sortSources(got)
	want := []Source{
		{Metadata: map[string]string{"tag": "Lose"}, Chunks: 1},
		{Source: "a.pdf", Metadata: map[string]string{"tag": "Ordnung"}, Chunks: 1},
		{Source: "a.pdf", Metadata: map[string]string{"tag": "Senat", "datum": "2026-01-13"}, Chunks: 2},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Sources = %+v\nwant %+v", got, want)
	}

	// Without keys a source is url and source alone, as before.
	s.sourceMetaKeys = nil
	got, _ = s.Sources(context.Background())
	if len(got) != 1 || got[0].Source != "a.pdf" || got[0].Chunks != 3 || got[0].Metadata != nil {
		t.Errorf("Sources without keys = %+v, want one a.pdf with 3 chunks", got)
	}
}

func TestLocalvecStoreUpdateMetadataRetagsOnlyTheGroup(t *testing.T) {
	path := writeSourcesDB(t, map[string]map[string]any{
		"1": {"source": "a.pdf", "tag": "Senat", "datum": "2026-01-13", "page": "1"},
		"2": {"source": "a.pdf", "tag": "Senat", "datum": "2026-01-13"},
		"3": {"source": "a.pdf", "tag": "Ordnung"},
		"4": {"source": "b.pdf", "tag": "Senat", "datum": "2026-01-13"},
	})
	s := &LocalvecStore{dbPath: path, sourceMetaKeys: []string{"tag", "datum"}}
	old := Source{Source: "a.pdf", Metadata: map[string]string{"tag": "Senat", "datum": "2026-01-13"}}
	n, err := s.UpdateMetadata(context.Background(), old, map[string]string{"tag": "Ordnung", "other": "ignored"})
	if err != nil || n != 2 {
		t.Fatalf("UpdateMetadata = %d, %v; want 2, nil", n, err)
	}
	db := readLocalvecDB(t, path)
	want := map[string]any{"source": "a.pdf", "tag": "Ordnung", "page": "1"}
	if got := entryMeta(t, db, "1"); !reflect.DeepEqual(got, want) {
		t.Errorf("entry 1 = %v, want %v (datum removed, page kept, other ignored)", got, want)
	}
	if got := entryMeta(t, db, "4"); got["tag"] != "Senat" {
		t.Errorf("entry 4 (other source) changed: %v", got)
	}
	if _, ok := db["1"]["embedding"]; !ok {
		t.Error("entry 1 lost its embedding")
	}
}

func TestLocalvecStoreUpdateMetadataNoMatchOrNoFile(t *testing.T) {
	path := writeSourcesDB(t, map[string]map[string]any{"1": {"source": "a.pdf", "tag": "Senat"}})
	s := &LocalvecStore{dbPath: path, sourceMetaKeys: []string{"tag"}}
	// a.pdf without a tag is a different group from a.pdf tagged Senat.
	if n, err := s.UpdateMetadata(context.Background(), Source{Source: "a.pdf"}, map[string]string{"tag": "X"}); err != nil || n != 0 {
		t.Errorf("UpdateMetadata = %d, %v; want 0, nil", n, err)
	}
	s.dbPath = filepath.Join(t.TempDir(), "missing.json")
	if n, err := s.UpdateMetadata(context.Background(), Source{Source: "a.pdf"}, nil); err != nil || n != 0 {
		t.Errorf("UpdateMetadata on missing file = %d, %v; want 0, nil", n, err)
	}
}

func TestLocalvecStoreDeleteSourceRemovesOnlyTheGroup(t *testing.T) {
	path := writeSourcesDB(t, map[string]map[string]any{
		"1": {"source": "a.pdf", "tag": "Senat"},
		"2": {"source": "a.pdf", "tag": "Senat"},
		"3": {"source": "a.pdf", "tag": "Ordnung"},
		"4": {"source": "a.pdf"},
	})
	s := &LocalvecStore{dbPath: path, sourceMetaKeys: []string{"tag"}}
	n, err := s.DeleteSource(context.Background(), Source{Source: "a.pdf", Metadata: map[string]string{"tag": "Senat"}})
	if err != nil || n != 2 {
		t.Fatalf("DeleteSource = %d, %v; want 2, nil", n, err)
	}
	if n, err := s.DeleteSource(context.Background(), Source{Source: "a.pdf"}); err != nil || n != 1 {
		t.Fatalf("DeleteSource untagged = %d, %v; want 1, nil", n, err)
	}
	db := readLocalvecDB(t, path)
	if len(db) != 1 || entryMeta(t, db, "3")["tag"] != "Ordnung" {
		t.Errorf("left %v, want only entry 3", db)
	}
}

func TestLocalvecStoreResetAndRetrieveFiltered(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := NewLocalvecStore(genkit.Init(ctx), "filtered", constEmbedder{}, dir, WithLocalvecSourceMetaKeys("tag"))
	if err != nil {
		t.Fatal(err)
	}
	var docs []*ai.Document
	for i, tag := range []string{"a", "b", "b", "a", "b"} {
		docs = append(docs, ai.DocumentFromText(strings.Repeat("x", i+1), map[string]any{"source": "s.pdf", "tag": tag}))
	}
	if err := s.Index(ctx, docs); err != nil {
		t.Fatal(err)
	}

	got, err := s.RetrieveFiltered(ctx, ai.DocumentFromText("q", nil), 2, func(m map[string]any) bool { return m["tag"] == "a" })
	if err != nil || len(got) != 2 {
		t.Fatalf("RetrieveFiltered = %d docs, %v; want 2", len(got), err)
	}
	for _, d := range got {
		if d.Metadata["tag"] != "a" {
			t.Errorf("RetrieveFiltered returned tag %v", d.Metadata["tag"])
		}
	}
	if got, err := s.Retrieve(ctx, ai.DocumentFromText("q", nil), 4); err != nil || len(got) != 4 {
		t.Errorf("Retrieve = %d docs, %v; want 4", len(got), err)
	}

	if err := s.Reset(ctx); err != nil {
		t.Fatal(err)
	}
	if srcs, err := s.Sources(ctx); err != nil || len(srcs) != 0 {
		t.Errorf("Sources after Reset = %v, %v; want none", srcs, err)
	}
	if err := s.Reset(ctx); err != nil {
		t.Errorf("second Reset: %v", err)
	}
}
