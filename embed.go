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

	"github.com/firebase/genkit/go/ai"
)

// embedKindKey marks a document as a search query rather than a passage.
// It is only read by prefixEmbedder and never reaches the vector store as
// meaningful metadata.
const (
	embedKindKey   = "embedKind"
	embedKindQuery = "query"
)

// QueryDocument builds a document that prefixEmbedder treats as a search
// query. Asymmetric embedding models expect a different instruction prefix
// for queries than for indexed passages; embedding a question like a passage
// costs noticeable recall.
//
// Note that HyDE output is deliberately NOT a query document: it is a
// hypothetical passage, and embedding it with the passage prefix is what
// makes HyDE bridge the query/document gap in the first place.
func QueryDocument(text string) *ai.Document {
	return ai.DocumentFromText(text, map[string]any{embedKindKey: embedKindQuery})
}

// embedPrefixes returns the (query, passage) instruction prefixes for an
// embedding model. Models that are symmetric — or whose prefixes we don't
// know — get empty prefixes and are left unwrapped.
func embedPrefixes(model string) (query, passage string) {
	name := strings.ToLower(model)
	if i := strings.IndexByte(name, ':'); i >= 0 {
		name = name[:i] // strip the ":latest" style tag
	}
	switch name {
	case "mxbai-embed-large":
		return "Represent this sentence for searching relevant passages: ", ""
	case "nomic-embed-text":
		return "search_query: ", "search_document: "
	case "multilingual-e5-large", "multilingual-e5-base", "multilingual-e5-small",
		"e5-large", "e5-base", "e5-small":
		return "query: ", "passage: "
	}
	return "", ""
}

// prefixEmbedder wraps an embedder and prepends the model's instruction
// prefix to every input document. Inputs are copied rather than mutated:
// the same documents are handed to the store for persistence, and the
// prefix must not end up in the stored content.
type prefixEmbedder struct {
	ai.Embedder
	queryPrefix   string
	passagePrefix string
}

// NewPrefixEmbedder wraps emb for models that need instruction prefixes and
// returns emb unchanged otherwise.
func NewPrefixEmbedder(emb ai.Embedder, model string) ai.Embedder {
	query, passage := embedPrefixes(model)
	if query == "" && passage == "" {
		return emb
	}
	return &prefixEmbedder{Embedder: emb, queryPrefix: query, passagePrefix: passage}
}

func (e *prefixEmbedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	if req == nil || len(req.Input) == 0 {
		return e.Embedder.Embed(ctx, req)
	}

	input := make([]*ai.Document, len(req.Input))
	for i, doc := range req.Input {
		prefix := e.passagePrefix
		if doc.Metadata != nil {
			if kind, _ := doc.Metadata[embedKindKey].(string); kind == embedKindQuery {
				prefix = e.queryPrefix
			}
		}
		input[i] = prefixDocument(doc, prefix)
	}

	next := *req
	next.Input = input
	return e.Embedder.Embed(ctx, &next)
}

// prefixDocument returns a copy of doc with prefix prepended to its text.
func prefixDocument(doc *ai.Document, prefix string) *ai.Document {
	if prefix == "" {
		return doc
	}
	var sb strings.Builder
	sb.WriteString(prefix)
	for _, part := range doc.Content {
		sb.WriteString(part.Text)
	}
	return ai.DocumentFromText(sb.String(), doc.Metadata)
}
