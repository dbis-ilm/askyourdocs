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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"time"
)

// CrossEncoder scores texts for relevance with a dedicated reranker model
// (a cross-encoder such as bge-reranker-v2-m3) behind a /rerank endpoint in
// the Jina/Cohere format, which GPUStack, vLLM and llama.cpp all serve:
//
//	POST <base>/rerank {"model", "query", "documents": [...]}
//	→ {"results": [{"index": 0, "relevance_score": 0.93}, ...]}
//
// It is an alternative to RerankScores: one request for all candidates
// instead of several LLM batches, and no reply format to parse. Get one
// from Providers.CrossEncoder.
type CrossEncoder struct {
	endpoint string // full URL of the /rerank endpoint
	apiKey   string
	model    string
	client   *http.Client
}

// NewCrossEncoder returns a CrossEncoder for model, served under the API
// base URL baseURL (the same URL as for chat models, e.g.
// https://gpustack.example/v1). timeout bounds each request; zero means 120s.
func NewCrossEncoder(baseURL, apiKey, model string, timeout time.Duration) *CrossEncoder {
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	return &CrossEncoder{
		endpoint: strings.TrimRight(baseURL, "/") + "/rerank",
		apiKey:   apiKey,
		model:    model,
		client:   &http.Client{Timeout: timeout},
	}
}

// Model is the reranker model's name.
func (c *CrossEncoder) Model() string { return c.model }

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n"`
}

type rerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

// Scores returns one relevance score in [0, 1] per text, aligned with
// texts. Servers differ in what they return: vLLM and GPUStack apply the
// model's sigmoid, llama.cpp returns raw logits. If any score of a reply
// lies outside [0, 1], the whole reply is taken as logits and passed
// through a sigmoid, so callers can use one threshold for both. Identical
// texts are sent once.
func (c *CrossEncoder) Scores(ctx context.Context, query string, texts []string) ([]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
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

	body, err := json.Marshal(rerankRequest{Model: c.model, Query: query, Documents: unique, TopN: len(unique)})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	began := time.Now()
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("rerank request: %s: %s", resp.Status, strings.TrimSpace(string(msg)))
	}
	var parsed rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("rerank response: %w", err)
	}
	slog.DebugContext(ctx, "CrossEncoder: scored", "model", c.model, "texts", len(unique), "took", time.Since(began).Round(time.Millisecond))

	uniqueScores := make([]float64, len(unique))
	got := make([]bool, len(unique))
	logits := false
	for _, r := range parsed.Results {
		if r.Index < 0 || r.Index >= len(unique) {
			return nil, fmt.Errorf("rerank response: index %d out of range (%d documents)", r.Index, len(unique))
		}
		uniqueScores[r.Index] = r.RelevanceScore
		got[r.Index] = true
		if r.RelevanceScore < 0 || r.RelevanceScore > 1 {
			logits = true
		}
	}
	for i, ok := range got {
		if !ok {
			return nil, fmt.Errorf("rerank response: no score for document %d of %d", i, len(unique))
		}
	}
	if logits {
		for i, s := range uniqueScores {
			uniqueScores[i] = 1 / (1 + math.Exp(-s))
		}
	}

	scores := make([]float64, len(texts))
	for i, idx := range slotOf {
		scores[i] = uniqueScores[idx]
	}
	return scores, nil
}

// ScaleScores maps scores in [0, 1] onto the integer 0–max scale RerankScores
// and RateConfidence use (max 10 for those), rounding to the nearest step.
func ScaleScores(scores []float64, max int) []int {
	out := make([]int, len(scores))
	for i, s := range scores {
		out[i] = int(math.Round(math.Min(math.Max(s, 0), 1) * float64(max)))
	}
	return out
}
