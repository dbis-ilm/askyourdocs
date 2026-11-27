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
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeOpenAI is an OpenAI-compatible server with one chat and one
// embedding model behind a bearer key.
func fakeOpenAI(t *testing.T, key string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	auth := func(w http.ResponseWriter, r *http.Request) bool {
		if r.Header.Get("Authorization") != "Bearer "+key {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return false
		}
		return true
	}
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		if auth(w, r) {
			w.Write([]byte(`{"object":"list","data":[{"id":"qwen3-8b"},{"id":"bge-m3"}]}`))
		}
	})
	mux.HandleFunc("POST /v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		if !auth(w, r) {
			return
		}
		var req struct{ Model string }
		json.NewDecoder(r.Body).Decode(&req)
		switch req.Model {
		case "bge-m3":
			w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3,0.4]}]}`))
		case "empty-embed":
			w.Write([]byte(`{"data":[]}`))
		default:
			http.Error(w, "model does not support embeddings", http.StatusBadRequest)
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestOpenAIModelDiscovery(t *testing.T) {
	ctx := context.Background()
	base := fakeOpenAI(t, "sk-test").URL + "/v1/"

	models, err := ListOpenAIModels(ctx, base, "sk-test")
	if err != nil || strings.Join(models, ",") != "qwen3-8b,bge-m3" {
		t.Fatalf("models = %v, %v", models, err)
	}
	if _, err := ListOpenAIModels(ctx, base, "wrong"); !errors.Is(err, ErrAPIKeyRejected) {
		t.Fatalf("wrong key: %v", err)
	}
	if dim, err := OpenAIEmbeddingDim(ctx, base, "sk-test", "bge-m3"); err != nil || dim != 4 {
		t.Fatalf("dim = %d, %v", dim, err)
	}
	_, err = OpenAIEmbeddingDim(ctx, base, "sk-test", "qwen3-8b")
	var se *ServerError
	if !errors.Is(err, ErrNotEmbeddingModel) || !errors.As(err, &se) || se.Model != "qwen3-8b" {
		t.Fatalf("chat model as embedder: %v", err)
	}
	if _, err := OpenAIEmbeddingDim(ctx, base, "sk-test", "empty-embed"); !errors.Is(err, ErrNoEmbedding) {
		t.Fatalf("empty embedding: %v", err)
	}
	if _, err := ListOpenAIModels(ctx, "http://127.0.0.1:1/v1", ""); !errors.Is(err, ErrUnreachable) {
		t.Fatalf("closed port: %v", err)
	}
}

func TestOllamaModelDiscovery(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/tags", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"models":[
			{"name":"gemma4:12b","size":8000000000,"capabilities":["completion","tools"]},
			{"name":"bge-m3:latest","size":1200000000,"capabilities":["embedding"]},
			{"name":"old-embed:latest","size":1},
			{"name":"minilm:latest","size":1,"details":{"family":"bert"}},
			{"name":"llama3:8b","size":1}]}`))
	})
	mux.HandleFunc("POST /api/show", func(w http.ResponseWriter, r *http.Request) {
		var req struct{ Model string }
		json.NewDecoder(r.Body).Decode(&req)
		switch req.Model {
		case "bge-m3:latest":
			w.Write([]byte(`{"model_info":{"general.architecture":"bert","bert.embedding_length":1024}}`))
		case "gemma4:12b":
			w.Write([]byte(`{"model_info":{"general.architecture":"gemma4"}}`))
		default:
			http.Error(w, `{"error":"model not found"}`, http.StatusNotFound)
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	ctx := context.Background()

	models, err := ListOllamaModels(ctx, srv.URL+"/")
	if err != nil || len(models) != 5 {
		t.Fatalf("models = %v, %v", models, err)
	}
	want := map[string]bool{"gemma4:12b": false, "bge-m3:latest": true, "old-embed:latest": true, "minilm:latest": true, "llama3:8b": false}
	for _, m := range models {
		if m.Embedding != want[m.Name] {
			t.Errorf("%s: embedding = %v", m.Name, m.Embedding)
		}
		if m.Tools != (m.Name == "gemma4:12b") {
			t.Errorf("%s: tools = %v", m.Name, m.Tools)
		}
	}
	if dim, err := OllamaEmbeddingDim(ctx, srv.URL, "bge-m3:latest"); err != nil || dim != 1024 {
		t.Fatalf("dim = %d, %v", dim, err)
	}
	for _, m := range []string{"gemma4:12b", "missing"} {
		if _, err := OllamaEmbeddingDim(ctx, srv.URL, m); !errors.Is(err, ErrNotEmbeddingModel) {
			t.Errorf("%s: %v", m, err)
		}
	}
	if _, err := ListOllamaModels(ctx, "http://127.0.0.1:1"); !errors.Is(err, ErrUnreachable) {
		t.Fatalf("closed port: %v", err)
	}
}

func TestBaseURLHelpers(t *testing.T) {
	for in, want := range map[string]string{
		"https://gpu.example/v1/":  "https://gpu.example/v1",
		" http://localhost:11434 ": "http://localhost:11434",
	} {
		if got, err := ValidateBaseURL(in); err != nil || got != want {
			t.Errorf("ValidateBaseURL(%q) = %q, %v", in, got, err)
		}
	}
	for _, bad := range []string{"gpu.example", "ftp://x", "https://"} {
		if _, err := ValidateBaseURL(bad); !errors.Is(err, ErrInvalidBaseURL) {
			t.Errorf("ValidateBaseURL(%q) = %v", bad, err)
		}
	}
	_, err := ValidateBaseURL("https://user:secret@gpu.example/v1")
	if !errors.Is(err, ErrCredentialsInURL) || strings.Contains(err.Error(), "secret") {
		t.Errorf("credentials: %v", err)
	}
	for base, local := range map[string]bool{
		"http://localhost:11434": true, "http://127.0.0.1:8080/v1": true, "http://[::1]:80": true,
		"https://gpu.example/v1": false, "http://192.168.1.20/v1": false,
	} {
		if IsLocalURL(base) != local {
			t.Errorf("IsLocalURL(%q) = %v", base, !local)
		}
	}
	if URLHost("https://gpu.example:8443/v1") != "gpu.example:8443" {
		t.Error("URLHost wrong")
	}
	if !LooksLikeEmbeddingModel("bge-m3") || !LooksLikeEmbeddingModel("nomic-embed-text") || LooksLikeEmbeddingModel("qwen3-8b") || LooksLikeEmbeddingModel("bge-reranker-v2") {
		t.Error("LooksLikeEmbeddingModel heuristics wrong")
	}
}
