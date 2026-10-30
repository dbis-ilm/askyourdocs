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
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// rerankServer answers /rerank with score(doc) per document, in reverse
// order (servers sort by relevance, so the reply order is not the request
// order), and records the last request.
func rerankServer(t *testing.T, score func(doc string) float64, last *rerankRequest, auth *string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rerank" {
			http.NotFound(w, r)
			return
		}
		if auth != nil {
			*auth = r.Header.Get("Authorization")
		}
		var req rerankRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if last != nil {
			*last = req
		}
		var resp rerankResponse
		for i := len(req.Documents) - 1; i >= 0; i-- {
			resp.Results = append(resp.Results, struct {
				Index          int     `json:"index"`
				RelevanceScore float64 `json:"relevance_score"`
			}{i, score(req.Documents[i])})
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestCrossEncoderScores(t *testing.T) {
	var last rerankRequest
	var auth string
	srv := rerankServer(t, func(doc string) float64 {
		if strings.Contains(doc, "Mensa") {
			return 0.9
		}
		return 0.05
	}, &last, &auth)

	ce := NewCrossEncoder(srv.URL+"/v1/", "secret", "bge-reranker-v2-m3", 0)
	got, err := ce.Scores(context.Background(), "Wann hat die Mensa offen?", []string{"Mensa: 11–14 Uhr", "Bibliothek", "Mensa: 11–14 Uhr"})
	if err != nil {
		t.Fatal(err)
	}
	if want := []float64{0.9, 0.05, 0.9}; !reflect.DeepEqual(got, want) {
		t.Errorf("scores = %v, want %v", got, want)
	}
	if len(last.Documents) != 2 {
		t.Errorf("sent %d documents, want 2 (duplicates once)", len(last.Documents))
	}
	if last.Model != "bge-reranker-v2-m3" || last.Query != "Wann hat die Mensa offen?" || last.TopN != 2 {
		t.Errorf("request = %+v", last)
	}
	if auth != "Bearer secret" {
		t.Errorf("Authorization = %q", auth)
	}
}

func TestCrossEncoderLogits(t *testing.T) {
	srv := rerankServer(t, func(doc string) float64 {
		if doc == "relevant" {
			return 4
		}
		return -6
	}, nil, nil)
	got, err := NewCrossEncoder(srv.URL+"/v1", "", "m", 0).Scores(context.Background(), "q", []string{"relevant", "noise"})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got[0]-0.982) > 0.001 || got[1] > 0.01 {
		t.Errorf("scores = %v, want sigmoid of [4, -6]", got)
	}
}

func TestCrossEncoderErrors(t *testing.T) {
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	}))
	defer failing.Close()
	if _, err := NewCrossEncoder(failing.URL, "", "m", 0).Scores(context.Background(), "q", []string{"a"}); err == nil || !strings.Contains(err.Error(), "model not found") {
		t.Errorf("err = %v, want the server's message", err)
	}

	short := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"index":0,"relevance_score":0.5}]}`))
	}))
	defer short.Close()
	if _, err := NewCrossEncoder(short.URL, "", "m", 0).Scores(context.Background(), "q", []string{"a", "b"}); err == nil {
		t.Error("want an error for a reply missing a document's score")
	}

	if got, err := NewCrossEncoder(short.URL, "", "m", 0).Scores(context.Background(), "q", nil); err != nil || got != nil {
		t.Errorf("no texts: got %v, %v", got, err)
	}
}

func TestScaleScores(t *testing.T) {
	got := ScaleScores([]float64{0, 0.04, 0.15, 0.5, 0.96, 1, 1.2, -0.1}, 10)
	if want := []int{0, 0, 2, 5, 10, 10, 10, 0}; !reflect.DeepEqual(got, want) {
		t.Errorf("ScaleScores = %v, want %v", got, want)
	}
}

func TestProvidersCrossEncoder(t *testing.T) {
	p, err := NewProviders(ProviderConfig{GPUStackURL: "https://gpustack.example/v1/", GPUStackAPIKey: "k"}, nil, "gpustack/bge-reranker-v2-m3")
	if err != nil {
		t.Fatal(err)
	}
	ce, err := p.CrossEncoder("gpustack/bge-reranker-v2-m3")
	if err != nil {
		t.Fatal(err)
	}
	if ce.endpoint != "https://gpustack.example/v1/rerank" || ce.Model() != "bge-reranker-v2-m3" || ce.apiKey != "k" {
		t.Errorf("cross-encoder = %+v", ce)
	}
	if _, err := p.CrossEncoder("ollama/bge-reranker-v2-m3"); err == nil {
		t.Error("want an error for a non-gpustack spec")
	}
	empty, _ := NewProviders(ProviderConfig{}, nil)
	if _, err := empty.CrossEncoder("gpustack/x"); err == nil {
		t.Error("want an error for a spec not passed to NewProviders")
	}
}
