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
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
)

// PromptOptions builds the options for one prompt execution: the model, the
// provider config when there is one, and the input. It returns a fresh
// slice on every call, so it's safe to call concurrently (e.g. once per
// reranking batch goroutine).
func PromptOptions(model ai.Model, config any, input map[string]any) []ai.PromptExecuteOption {
	opts := make([]ai.PromptExecuteOption, 0, 3)
	opts = append(opts, ai.WithModel(model))
	if config != nil {
		opts = append(opts, ai.WithConfig(config))
	}
	return append(opts, ai.WithInput(input))
}

// DocText concatenates doc's content parts into a single string.
func DocText(doc *ai.Document) string {
	var sb strings.Builder
	for _, part := range doc.Content {
		sb.WriteString(part.Text)
	}
	return sb.String()
}

// GenerateHyDEText executes hydePrompt — expected to take {"question":
// question} and return a short hypothetical passage that could answer it —
// and returns its text. On error or an empty response it returns question
// unchanged: HyDE is a retrieval aid, not something retrieval should block
// on. The prompt's actual wording is the caller's own asset (e.g. a
// Genkit dotprompt loaded from the app's prompts directory) — its style
// should match the kind of documents the app indexes, which engine has no
// way to know.
func GenerateHyDEText(ctx context.Context, hydePrompt ai.Prompt, model ai.Model, config any, question string) string {
	resp, err := hydePrompt.Execute(ctx, PromptOptions(model, config, map[string]any{"question": question})...)
	if err != nil || resp.Text() == "" {
		return question
	}
	return resp.Text()
}

// RankedList is one retrieval source's results plus its fusion weight, for
// ReciprocalRankFusion.
type RankedList struct {
	Docs   []*ai.Document
	Weight float64
}

// FusedDoc is one document after ReciprocalRankFusion: its original
// *ai.Document, its concatenated text (the fusion/dedup key and what a
// reranker typically scores), and its fused score.
type FusedDoc struct {
	Doc   *ai.Document
	Text  string
	Score float64
}

// ReciprocalRankFusion merges several weighted, ranked retrieval result
// lists into one via Reciprocal Rank Fusion:
//
//	score(d) = Σ weight_i / (k + rank_i(d) + 1)
//
// Documents are deduplicated by their concatenated text content (DocText) —
// two sources returning the same chunk collapse into one FusedDoc whose
// score sums both contributions, rather than appearing twice. 60 is the
// standard RRF k; pass that unless tuning. Results are sorted by descending
// score; maxResults <= 0 means no cap.
func ReciprocalRankFusion(lists []RankedList, k, maxResults int) []FusedDoc {
	fused := make(map[string]*FusedDoc)
	for _, list := range lists {
		for rank, doc := range list.Docs {
			txt := DocText(doc)
			entry, ok := fused[txt]
			if !ok {
				entry = &FusedDoc{Doc: doc, Text: txt}
				fused[txt] = entry
			}
			entry.Score += list.Weight / float64(k+rank+1)
		}
	}

	merged := make([]FusedDoc, 0, len(fused))
	for _, entry := range fused {
		merged = append(merged, *entry)
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].Score > merged[j].Score })
	if maxResults > 0 && len(merged) > maxResults {
		merged = merged[:maxResults]
	}
	return merged
}

// RerankOptions tunes RerankScores. The zero value is usable — see each
// field's default below.
type RerankOptions struct {
	BatchSize    int // chunks per scoring call; default 5
	Parallel     int // max in-flight scoring calls; default 4
	DefaultScore int // score kept for a batch whose call failed; default 5
}

func (o RerankOptions) withDefaults() RerankOptions {
	if o.BatchSize <= 0 {
		o.BatchSize = 5
	}
	if o.Parallel <= 0 {
		o.Parallel = 4
	}
	if o.DefaultScore == 0 {
		o.DefaultScore = 5
	}
	return o
}

// RerankScores scores every text for relevance to question, in batches, via
// rerankPrompt (expected to take {"question": question, "chunks":
// [{"num": int, "text": string}, ...]} and reply with one "<num>:<score>"
// line per chunk — see parseRerankScores). The returned slice is aligned
// with texts. A batch whose call fails keeps opts.DefaultScore, so a
// partial failure costs ranking quality rather than dropping chunks.
//
// Candidate texts routinely repeat (e.g. neighbouring chunks of one section
// expanding to the same parent-text window) — each distinct text is scored
// once and every occurrence gets that score, which also shrinks the number
// of batches needed.
func RerankScores(ctx context.Context, rerankPrompt ai.Prompt, model ai.Model, config any, question string, texts []string, opts RerankOptions) []int {
	if len(texts) == 0 {
		return nil
	}
	opts = opts.withDefaults()

	unique := make([]string, 0, len(texts))
	seen := make(map[string]int, len(texts))
	slotOf := make([]int, len(texts))
	for i, text := range texts {
		idx, ok := seen[text]
		if !ok {
			idx = len(unique)
			seen[text] = idx
			unique = append(unique, text)
		}
		slotOf[i] = idx
	}
	if len(unique) < len(texts) {
		log.Printf("[RerankScores] %d candidates, %d distinct texts", len(texts), len(unique))
	}

	uniqueScores := make([]int, len(unique))
	for i := range uniqueScores {
		uniqueScores[i] = opts.DefaultScore
	}

	var wg sync.WaitGroup
	slot := make(chan struct{}, opts.Parallel)

	for start := 0; start < len(unique); start += opts.BatchSize {
		end := min(start+opts.BatchSize, len(unique))

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			slot <- struct{}{}
			defer func() { <-slot }()

			// The prompt numbers chunks from 1 within each batch; the offset
			// maps them back onto the candidate list.
			chunks := make([]map[string]any, 0, end-start)
			for i := start; i < end; i++ {
				chunks = append(chunks, map[string]any{
					"num":  i - start + 1,
					"text": unique[i],
				})
			}

			began := time.Now()
			resp, err := rerankPrompt.Execute(ctx,
				PromptOptions(model, config, map[string]any{
					"question": question,
					"chunks":   chunks,
				})...)
			if err != nil || resp.Text() == "" {
				log.Printf("[RerankScores] batch %d-%d failed after %s, keeping retrieval order (err=%v)",
					start+1, end, time.Since(began).Round(time.Millisecond), err)
				return
			}
			log.Printf("[RerankScores] batch %d-%d took %s", start+1, end, time.Since(began).Round(time.Millisecond))
			batch, scored := parseRerankScores(resp.Text(), end-start, opts.DefaultScore)
			if scored < end-start {
				log.Printf("[RerankScores] batch %d-%d: only %d of %d chunks scored, the rest keep %d",
					start+1, end, scored, end-start, opts.DefaultScore)
			}
			// Each batch writes its own slice of the shared array, so the
			// writes never overlap.
			copy(uniqueScores[start:end], batch)
		}(start, end)
	}

	wg.Wait()

	scores := make([]int, len(texts))
	for i, idx := range slotOf {
		scores[i] = uniqueScores[idx]
	}
	return scores
}

// parseRerankScores parses LLM reranking output like "1:8\n2:3\n..." into a
// score slice indexed by document position (0-based), and reports how many
// positions the model actually scored. Positions it left out keep
// defaultScore, so a caller that cares about the difference should check the
// count. Small models get this wrong often enough to matter: qwen2.5:3b
// answered a five-chunk batch with lines numbered up to 10, and another with
// a single line.
func parseRerankScores(text string, n, defaultScore int) ([]int, int) {
	scores := make([]int, n)
	for i := range scores {
		scores[i] = defaultScore
	}
	applied := make([]bool, n)
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		idx := 0
		for _, ch := range strings.TrimSpace(parts[0]) {
			if ch >= '0' && ch <= '9' {
				idx = idx*10 + int(ch-'0')
			}
		}
		idx-- // 1-based to 0-based
		if idx < 0 || idx >= n {
			continue
		}
		score := 0
		for _, ch := range strings.TrimSpace(parts[1]) {
			if ch >= '0' && ch <= '9' {
				score = score*10 + int(ch-'0')
			}
		}
		if score > 10 {
			score = 10
		}
		scores[idx] = score
		applied[idx] = true
	}
	count := 0
	for _, ok := range applied {
		if ok {
			count++
		}
	}
	return scores, count
}
