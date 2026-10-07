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
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// testFeedbackEntry is the entry shape the store is exercised with — any
// type works as long as the store can read its ID and write its rating.
type testFeedbackEntry struct {
	ID       string `json:"id"`
	Question string `json:"question"`
	Answer   string `json:"answer"`
	Rating   string `json:"rating,omitempty"`
}

func newTestFeedbackStore(path string) *FeedbackStore[testFeedbackEntry] {
	return NewFeedbackStore(path,
		func(e testFeedbackEntry) string { return e.ID },
		func(e *testFeedbackEntry, r string) { e.Rating = r })
}

func TestFeedbackStoreAppendAndRate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "feedback.json")
	s := newTestFeedbackStore(path)

	if err := s.Append(testFeedbackEntry{ID: "a1", Question: "Wer war dabei?", Answer: "X"}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := s.Append(testFeedbackEntry{ID: "b2", Question: "Was wurde beschlossen?", Answer: "Y"}); err != nil {
		t.Fatalf("Append: %v", err)
	}

	found, err := s.Rate("a1", "up")
	if err != nil {
		t.Fatalf("Rate: %v", err)
	}
	if !found {
		t.Fatal("Rate returned found=false for an existing entry")
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}
	var entries []testFeedbackEntry
	if err := json.Unmarshal(raw, &entries); err != nil {
		t.Fatalf("parse file: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("got %d entries, want 2", len(entries))
	}
	var a1, b2 *testFeedbackEntry
	for i := range entries {
		switch entries[i].ID {
		case "a1":
			a1 = &entries[i]
		case "b2":
			b2 = &entries[i]
		}
	}
	if a1 == nil || a1.Rating != "up" {
		t.Errorf("a1 = %+v, want Rating=up", a1)
	}
	if b2 == nil || b2.Rating != "" {
		t.Errorf("b2 = %+v, want Rating unset (untouched)", b2)
	}
}

func TestFeedbackStoreAll(t *testing.T) {
	path := filepath.Join(t.TempDir(), "feedback.json")
	s := newTestFeedbackStore(path)
	if err := s.Append(testFeedbackEntry{ID: "a1"}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := s.Append(testFeedbackEntry{ID: "b2"}); err != nil {
		t.Fatalf("Append: %v", err)
	}

	entries, err := s.All()
	if err != nil {
		t.Fatalf("All: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("got %d entries, want 2", len(entries))
	}
}

func TestFeedbackStoreAllOnMissingFile(t *testing.T) {
	s := newTestFeedbackStore(filepath.Join(t.TempDir(), "does-not-exist.json"))
	entries, err := s.All()
	if err != nil {
		t.Fatalf("All: %v", err)
	}
	if entries != nil {
		t.Errorf("entries = %v, want nil", entries)
	}
}

func TestFeedbackStoreRateUnknownID(t *testing.T) {
	path := filepath.Join(t.TempDir(), "feedback.json")
	s := newTestFeedbackStore(path)
	if err := s.Append(testFeedbackEntry{ID: "a1"}); err != nil {
		t.Fatalf("Append: %v", err)
	}

	found, err := s.Rate("does-not-exist", "up")
	if err != nil {
		t.Fatalf("Rate: %v", err)
	}
	if found {
		t.Error("Rate returned found=true for a nonexistent ID")
	}
}

func TestFeedbackStoreRateOnMissingFile(t *testing.T) {
	s := newTestFeedbackStore(filepath.Join(t.TempDir(), "does-not-exist.json"))
	found, err := s.Rate("a1", "up")
	if err != nil {
		t.Fatalf("Rate: %v", err)
	}
	if found {
		t.Error("Rate returned found=true against an empty/missing log")
	}
}

func TestFeedbackStoreRateCanClearAPreviousRating(t *testing.T) {
	path := filepath.Join(t.TempDir(), "feedback.json")
	s := newTestFeedbackStore(path)
	if err := s.Append(testFeedbackEntry{ID: "a1"}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if _, err := s.Rate("a1", "down"); err != nil {
		t.Fatalf("Rate: %v", err)
	}
	if _, err := s.Rate("a1", ""); err != nil {
		t.Fatalf("Rate: %v", err)
	}

	raw, _ := os.ReadFile(path)
	var entries []testFeedbackEntry
	json.Unmarshal(raw, &entries)
	if len(entries) != 1 || entries[0].Rating != "" {
		t.Errorf("entries = %+v, want a single entry with Rating cleared", entries)
	}
}

func TestNewFeedbackIDReturnsDistinctValues(t *testing.T) {
	a := NewRandomHexID(8)
	b := NewRandomHexID(8)
	if a == "" || b == "" {
		t.Fatal("NewRandomHexID returned an empty string")
	}
	if a == b {
		t.Errorf("two calls returned the same ID %q", a)
	}
}
