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
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

var testStopWords = map[string]bool{
	"was": true, "ist": true, "in": true, "der": true, "die": true, "das": true,
	"den": true, "über": true, "und": true,
}

func TestExtractKeywords(t *testing.T) {
	got := ExtractKeywords("Was ist in der Sitzung über den Haushalt beschlossen?", testStopWords)
	want := []string{"sitzung", "haushalt", "beschlossen"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ExtractKeywords = %v, want %v", got, want)
	}
}

func TestExtractKeywordsDeduplicates(t *testing.T) {
	got := ExtractKeywords("Haushalt Haushalt haushalt", nil)
	if len(got) != 1 || got[0] != "haushalt" {
		t.Errorf("ExtractKeywords = %v, want a single deduplicated entry", got)
	}
}

func TestExtractKeywordsNilStopWordsKeepsShortTokensFiltered(t *testing.T) {
	// No stop-word set at all: short tokens (<3 chars) are still dropped.
	got := ExtractKeywords("ja ok Haushalt", nil)
	want := []string{"haushalt"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ExtractKeywords = %v, want %v", got, want)
	}
}

func TestExtractQuotedPhrases(t *testing.T) {
	got := ExtractQuotedPhrases(`Suche nach "Digitalisierungsstrategie" und 'Senat 2026'`)
	want := []string{"digitalisierungsstrategie", "senat 2026"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ExtractQuotedPhrases = %v, want %v", got, want)
	}
}

func TestExtractQuotedPhrasesNone(t *testing.T) {
	if got := ExtractQuotedPhrases("keine Anführungszeichen hier"); got != nil {
		t.Errorf("ExtractQuotedPhrases = %v, want nil", got)
	}
}

// writeLocalvecDB writes entries in the shape LocalvecStore.KeywordSearch
// reads back, returning a *LocalvecStore pointed at the fixture file.
func writeLocalvecDB(t *testing.T, entries map[string]keywordDBEntry) *LocalvecStore {
	t.Helper()
	path := filepath.Join(t.TempDir(), "db.json")
	data, err := json.Marshal(entries)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return &LocalvecStore{dbPath: path}
}

func newKeywordEntry(text string, meta map[string]any) keywordDBEntry {
	var e keywordDBEntry
	e.Doc.Content = []struct {
		Text string `json:"text"`
	}{{Text: text}}
	e.Doc.Metadata = meta
	return e
}

func TestLocalvecKeywordSearchRanksPhraseAboveKeyword(t *testing.T) {
	store := writeLocalvecDB(t, map[string]keywordDBEntry{
		"1": newKeywordEntry(`Beschluss zur "Digitalisierungsstrategie" der Fakultät`, nil),
		"2": newKeywordEntry("Die Digitalisierung schreitet voran, aber ohne feste Strategie", nil),
	})

	docs, err := store.KeywordSearch(context.Background(), `"Digitalisierungsstrategie"`, 10, nil, nil)
	if err != nil {
		t.Fatalf("KeywordSearch: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("got %d docs, want 1 (only the exact phrase match)", len(docs))
	}
}

func TestLocalvecKeywordSearchAppliesFilter(t *testing.T) {
	store := writeLocalvecDB(t, map[string]keywordDBEntry{
		"1": newKeywordEntry("Haushalt beschlossen", map[string]any{"tag": "Senat"}),
		"2": newKeywordEntry("Haushalt beschlossen", map[string]any{"tag": "Fakultätsrat"}),
	})

	filter := func(meta map[string]any) bool {
		tag, _ := meta["tag"].(string)
		return tag == "Senat"
	}
	docs, err := store.KeywordSearch(context.Background(), "Haushalt beschlossen", 10, nil, filter)
	if err != nil {
		t.Fatalf("KeywordSearch: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("got %d docs, want 1 (filtered to tag Senat)", len(docs))
	}
}

func TestLocalvecKeywordSearchNoKeywordsReturnsNil(t *testing.T) {
	store := writeLocalvecDB(t, map[string]keywordDBEntry{
		"1": newKeywordEntry("irrelevant", nil),
	})
	docs, err := store.KeywordSearch(context.Background(), "und der die", 10, testStopWords, nil)
	if err != nil {
		t.Fatalf("KeywordSearch: %v", err)
	}
	if docs != nil {
		t.Errorf("docs = %v, want nil when the query has no usable keywords", docs)
	}
}

func TestLocalvecKeywordSearchRespectsMaxResults(t *testing.T) {
	entries := map[string]keywordDBEntry{}
	for i := 0; i < 5; i++ {
		entries[string(rune('a'+i))] = newKeywordEntry("Haushalt beschlossen", nil)
	}
	store := writeLocalvecDB(t, entries)

	docs, err := store.KeywordSearch(context.Background(), "Haushalt", 2, nil, nil)
	if err != nil {
		t.Fatalf("KeywordSearch: %v", err)
	}
	if len(docs) != 2 {
		t.Errorf("got %d docs, want capped at 2", len(docs))
	}
}

func TestLocalvecKeywordSearchSortsByScore(t *testing.T) {
	store := writeLocalvecDB(t, map[string]keywordDBEntry{
		"one-match":  newKeywordEntry("Haushalt", nil),
		"two-match":  newKeywordEntry("Haushalt und Digitalisierung", nil),
		"zero-match": newKeywordEntry("nichts davon", nil),
	})

	docs, err := store.KeywordSearch(context.Background(), "Haushalt Digitalisierung", 10, nil, nil)
	if err != nil {
		t.Fatalf("KeywordSearch: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("got %d docs, want 2", len(docs))
	}
	if DocText(docs[0]) != "Haushalt und Digitalisierung" {
		t.Errorf("top result = %q, want the two-keyword match ranked first", DocText(docs[0]))
	}
}

func TestLocalvecKeywordSearchMissingDBFileReturnsNil(t *testing.T) {
	store := &LocalvecStore{dbPath: filepath.Join(t.TempDir(), "does-not-exist.json")}
	docs, err := store.KeywordSearch(context.Background(), "Haushalt", 10, nil, nil)
	if err != nil {
		t.Fatalf("KeywordSearch: %v", err)
	}
	if docs != nil {
		t.Errorf("docs = %v, want nil for a missing DB file", docs)
	}
}

func TestKeywordSearchSQLNoKeywordsReturnsNilWithoutQueryingPool(t *testing.T) {
	// No pool is passed — if this reached the pool, it would panic on a nil
	// dereference, so a clean (nil, nil) proves the early return fires before
	// any query is attempted.
	docs, err := KeywordSearchSQL(context.Background(), nil, "und der die", 10, "german", testStopWords, "", nil)
	if err != nil {
		t.Fatalf("KeywordSearchSQL: %v", err)
	}
	if docs != nil {
		t.Errorf("docs = %v, want nil when the query has no usable keywords", docs)
	}
}
