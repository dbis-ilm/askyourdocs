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

import "fmt"

// Progress is one step of a long-running IndexSource, RunCrawl or
// ReindexCrawlData call, reported to a ProgressFunc. The library only
// reports what happened; wording it for a user — in whatever language —
// is up to the app, e.g. with ProgressMessages and GermanProgress. Which
// fields are set depends on Stage; see each stage's comment.
type Progress struct {
	Stage ProgressStage `json:"stage"`
	Done  int           `json:"done,omitempty"`
	Total int           `json:"total,omitempty"`
	Depth int           `json:"depth,omitempty"`
	URL   string        `json:"url,omitempty"`
	File  string        `json:"file,omitempty"`
}

// ProgressStage says which step a Progress reports.
type ProgressStage string

const (
	// StageChunked: IndexSource has split the source into Total chunks
	// and starts indexing them.
	StageChunked ProgressStage = "chunked"
	// StageBatchIndexed: IndexSource has written Done of Total batches.
	StageBatchIndexed ProgressStage = "batch-indexed"
	// StageCrawlFetch: RunCrawl is fetching URL, the Done-th page visited
	// of at most Total, at link depth Depth.
	StageCrawlFetch ProgressStage = "crawl-fetch"
	// StageCrawlIndexing: RunCrawl has finished fetching and removing
	// boilerplate and starts indexing Total pages.
	StageCrawlIndexing ProgressStage = "crawl-indexing"
	// StageCrawlPageIndexed: RunCrawl has handed Done of Total pages to
	// indexPage, the last one being URL — whether indexing it succeeded is
	// in the CrawlResult, not here.
	StageCrawlPageIndexed ProgressStage = "crawl-page-indexed"
	// StageCrawlCheckGone: RunCrawl (WithRemovePage) is checking whether
	// URL, the Done-th of Total known pages not crawled this time, is gone.
	StageCrawlCheckGone ProgressStage = "crawl-check-gone"
	// StageReindexFile: ReindexCrawlData has processed Done of Total cached
	// files, the last one being File. A file that can't be read or parsed is
	// skipped without an event.
	StageReindexFile ProgressStage = "reindex-file"
)

// ProgressFunc receives Progress events. A nil ProgressFunc is valid
// wherever one is accepted and means nothing is listening.
type ProgressFunc func(Progress)

func (f ProgressFunc) report(p Progress) {
	if f != nil {
		f(p)
	}
}

// ProgressMessages adapts send, which takes lines of text — a streaming
// flow's chunk sender, a job's progress callback — to a ProgressFunc that
// words each event with format:
//
//	askyourdocs.IndexSource(ctx, store, src, size, overlap,
//		askyourdocs.ProgressMessages(sendChunk, askyourdocs.GermanProgress))
//
// A nil send gives a nil ProgressFunc. format returning "" drops the event.
func ProgressMessages(send func(string), format func(Progress) string) ProgressFunc {
	if send == nil {
		return nil
	}
	return func(p Progress) {
		if msg := format(p); msg != "" {
			send(msg)
		}
	}
}

// GermanProgress words p as the German status line the library used to
// send itself, for apps that keep showing those. An unknown stage gives "".
func GermanProgress(p Progress) string {
	switch p.Stage {
	case StageChunked:
		return fmt.Sprintf("%d Chunks erzeugt, indexiere...", p.Total)
	case StageBatchIndexed:
		return fmt.Sprintf("Batch %d/%d indexiert", p.Done, p.Total)
	case StageCrawlFetch:
		return fmt.Sprintf("Seite %d/%d (Tiefe %d): %s", p.Done, p.Total, p.Depth, p.URL)
	case StageCrawlIndexing:
		return fmt.Sprintf("Indexiere %d gecrawlte Seiten...", p.Total)
	case StageCrawlPageIndexed:
		return fmt.Sprintf("%d/%d Seiten indexiert", p.Done, p.Total)
	case StageCrawlCheckGone:
		return fmt.Sprintf("Prüfe nicht mehr gefundene Seite %d/%d: %s", p.Done, p.Total, p.URL)
	case StageReindexFile:
		return fmt.Sprintf("%d/%d Dateien verarbeitet (%s)", p.Done, p.Total, p.File)
	}
	return ""
}
