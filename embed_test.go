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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
)

// recordingEmbedder captures the text it is asked to embed.
type recordingEmbedder struct {
	seen []string
}

func (e *recordingEmbedder) Name() string            { return "test/recording" }
func (e *recordingEmbedder) Register(r api.Registry) {}

func (e *recordingEmbedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	for _, doc := range req.Input {
		var sb strings.Builder
		for _, part := range doc.Content {
			sb.WriteString(part.Text)
		}
		e.seen = append(e.seen, sb.String())
	}
	return &ai.EmbedResponse{}, nil
}

func TestPrefixEmbedder(t *testing.T) {
	rec := &recordingEmbedder{}
	emb := NewPrefixEmbedder(rec, "mxbai-embed-large:latest")
	if emb == ai.Embedder(rec) {
		t.Fatal("mxbai-embed-large should be wrapped")
	}

	passage := ai.DocumentFromText("Indexierter Absatz", map[string]any{"url": "https://example.com"})
	query := QueryDocument("Wonach wird gesucht?")

	if _, err := emb.Embed(context.Background(), &ai.EmbedRequest{Input: []*ai.Document{passage, query}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(rec.seen) != 2 {
		t.Fatalf("got %d embedded texts, want 2", len(rec.seen))
	}
	if rec.seen[0] != "Indexierter Absatz" {
		t.Errorf("passage text = %q, want unprefixed", rec.seen[0])
	}
	const queryPrefix = "Represent this sentence for searching relevant passages: "
	if !strings.HasPrefix(rec.seen[1], queryPrefix) {
		t.Errorf("query text = %q, want prefix %q", rec.seen[1], queryPrefix)
	}

	// The caller's documents must be untouched — they are what gets stored.
	if got := passage.Content[0].Text; got != "Indexierter Absatz" {
		t.Errorf("input document was mutated: %q", got)
	}

	// An unknown model needs no wrapper.
	if NewPrefixEmbedder(rec, "some-symmetric-model") != ai.Embedder(rec) {
		t.Error("unknown model should be returned unwrapped")
	}
}
