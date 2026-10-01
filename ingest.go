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
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// IngestSource abstracts fetching one document's paragraphs and turning them
// into indexable documents, so the extract → chunk → delete-stale → index →
// cleanup-on-failure pipeline (IndexSource) is written once instead of once
// per document type or per app. A PDF/.docx file and a crawled web page are
// two implementations; a different document store is a third.
type IngestSource interface {
	// Fetch returns the source's paragraphs.
	Fetch(ctx context.Context) ([]Paragraph, error)
	// Documents turns chunks into indexable documents carrying this source's
	// own metadata (e.g. pages/section/top/source name for a PDF; url/
	// breadcrumb for a web page) plus whatever else the app attaches.
	Documents(chunks []Chunk) []*ai.Document
	// DeleteStale removes this source's previously indexed chunks (upsert
	// semantics). Called both before indexing and, on failure, to clean up a
	// partial index — so a failed (re)index leaves nothing rather than a mix
	// of old and partial-new chunks.
	DeleteStale(ctx context.Context, store VectorStore) (int, error)
	// Name identifies the source for logging.
	Name() string
}

// IndexSource runs the shared extract/chunk/delete-stale/index/cleanup
// pipeline against any IngestSource. sendChunk, when non-nil, receives
// incremental progress messages for a streaming flow's HTTP adapter; pass
// nil when nothing is listening. Returns how many documents were indexed
// and how many stale ones were deleted beforehand.
func IndexSource(ctx context.Context, store VectorStore, src IngestSource, chunkSize, chunkOverlap int, sendChunk func(string)) (indexed, deleted int, err error) {
	paras, err := genkit.Run(ctx, "extract", func() ([]Paragraph, error) {
		return src.Fetch(ctx)
	})
	if err != nil {
		return 0, 0, err
	}

	docs, err := genkit.Run(ctx, "chunk", func() ([]*ai.Document, error) {
		chunks := BuildChunks(paras, chunkSize, chunkOverlap)
		return src.Documents(chunks), nil
	})
	if err != nil {
		return 0, 0, err
	}
	if sendChunk != nil {
		sendChunk(fmt.Sprintf("%d Chunks erzeugt, indexiere...", len(docs)))
	}

	deleted, err = src.DeleteStale(ctx, store)
	if err != nil {
		log.Printf("[IndexSource] deleteStale %s: %v", src.Name(), err)
	} else if deleted > 0 {
		log.Printf("[IndexSource] deleted %d stale chunks for %s", deleted, src.Name())
	}

	if err := IndexInBatches(ctx, store, docs, func(done, total int) {
		if sendChunk != nil {
			sendChunk(fmt.Sprintf("Batch %d/%d indexiert", done, total))
		}
	}); err != nil {
		if _, cleanupErr := src.DeleteStale(ctx, store); cleanupErr != nil {
			log.Printf("[IndexSource] cleanup after failed index for %s: %v", src.Name(), cleanupErr)
		}
		return 0, deleted, err
	}

	return len(docs), deleted, nil
}
