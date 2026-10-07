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
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// FeedbackStore persists an app's feedback entries — one logged
// question/answer interaction each, optionally rated by the asker afterwards
// — to a JSON file, read-modify-write under a mutex (the same approach
// LocalvecStore uses for its DB file). Logging every interaction, not just
// rated ones, turns real usage into a ready-made eval dataset: questions and
// contexts as they actually occur, instead of ones an LLM invented from
// sampled chunks. Traffic is one append per question and one update per
// rating click, so this needs no database.
//
// E is the app's own entry type, so each app keeps the fields and JSON shape
// it needs (its own citation struct, extra metadata like tags). The store
// only needs to know how to read an entry's ID and write its rating, which
// the app supplies to NewFeedbackStore.
type FeedbackStore[E any] struct {
	mu        sync.Mutex
	path      string
	id        func(E) string
	setRating func(*E, string)
}

// NewFeedbackStore returns a store persisting to path (created, with its
// parent directories, on first write). id returns an entry's ID; setRating
// sets an entry's rating in place.
func NewFeedbackStore[E any](path string, id func(E) string, setRating func(*E, string)) *FeedbackStore[E] {
	return &FeedbackStore[E]{path: path, id: id, setRating: setRating}
}

func (s *FeedbackStore[E]) read() ([]E, error) {
	raw, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read feedback log: %w", err)
	}
	var entries []E
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("parse feedback log: %w", err)
	}
	return entries, nil
}

func (s *FeedbackStore[E]) write(entries []E) error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("create feedback log dir: %w", err)
	}
	out, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal feedback log: %w", err)
	}
	if err := os.WriteFile(s.path, out, 0o644); err != nil {
		return fmt.Errorf("write feedback log: %w", err)
	}
	return nil
}

// All returns every logged feedback entry.
func (s *FeedbackStore[E]) All() ([]E, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.read()
}

// Append records a new question/answer interaction, unrated.
func (s *FeedbackStore[E]) Append(entry E) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	entries, err := s.read()
	if err != nil {
		return err
	}
	entries = append(entries, entry)
	return s.write(entries)
}

// Rate sets the rating on the entry with the given ID, replacing whatever
// rating it had before. Returns false if no entry with that ID exists.
func (s *FeedbackStore[E]) Rate(id, rating string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entries, err := s.read()
	if err != nil {
		return false, err
	}
	found := false
	for i := range entries {
		if s.id(entries[i]) == id {
			s.setRating(&entries[i], rating)
			found = true
			break
		}
	}
	if !found {
		return false, nil
	}
	return true, s.write(entries)
}
