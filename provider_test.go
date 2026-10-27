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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// fakeServer records every request body by path and answers with the fixed
// response registered for that path.
type fakeServer struct {
	*httptest.Server
	mu       sync.Mutex
	bodies   map[string]map[string]any
	headers  map[string]http.Header
	response map[string]string
}

func newFakeServer(t *testing.T, response map[string]string) *fakeServer {
	f := &fakeServer{bodies: map[string]map[string]any{}, headers: map[string]http.Header{}, response: response}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		f.mu.Lock()
		f.bodies[r.URL.Path] = body
		f.headers[r.URL.Path] = r.Header.Clone()
		resp, ok := f.response[r.URL.Path]
		f.mu.Unlock()
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, resp)
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeServer) body(path string) map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.bodies[path]
}

func TestParseModelSpec(t *testing.T) {
	for _, tc := range []struct{ spec, provider, name string }{
		{"ollama/gemma4:12b", "ollama", "gemma4:12b"},
		{"gpustack/qwen3-8b", "gpustack", "qwen3-8b"},
		{"googleai/gemini-2.5-flash", "googleai", "gemini-2.5-flash"},
		{"mxbai-embed-large", "ollama", "mxbai-embed-large"},
		{"hf.co/user/repo:Q4", "ollama", "hf.co/user/repo:Q4"},
	} {
		p, n := ParseModelSpec(tc.spec)
		if p != tc.provider || n != tc.name {
			t.Errorf("ParseModelSpec(%q) = %q, %q; want %q, %q", tc.spec, p, n, tc.provider, tc.name)
		}
	}
}

func TestNewProvidersRejects(t *testing.T) {
	cases := map[string]struct {
		cfg     ProviderConfig
		allowed []string
		spec    string
		want    string
	}{
		"disallowed provider":  {allowed: []string{ProviderOllama, ProviderGPUStack}, spec: "googleai/gemini-2.5-flash", want: "not allowed"},
		"gpustack without URL": {spec: "gpustack/qwen3-8b", want: "GPUSTACK_URL"},
		"no model name":        {spec: "ollama/", want: "names no model"},
	}
	for name, tc := range cases {
		_, err := NewProviders(tc.cfg, tc.allowed, tc.spec)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want it to mention %q", name, err, tc.want)
		}
	}
}

func TestProvidersPluginsOnlyForNamedProviders(t *testing.T) {
	p, err := NewProviders(ProviderConfig{GPUStackURL: "http://gpustack"}, nil, "ollama/a", "", "ollama/b", "mxbai-embed-large")
	if err != nil {
		t.Fatal(err)
	}
	if got := len(p.Plugins()); got != 1 {
		t.Fatalf("got %d plugins, want only ollama's", got)
	}
}

func TestProvidersModelNotPassedToNewProviders(t *testing.T) {
	p, err := NewProviders(ProviderConfig{}, nil, "ollama/a")
	if err != nil {
		t.Fatal(err)
	}
	g := genkit.Init(context.Background(), genkit.WithPlugins(p.Plugins()...))
	if _, err := p.Model(g, "gpustack/qwen3-8b"); err == nil {
		t.Error("expected an error for a provider NewProviders never saw")
	}
}

func TestProvidersGPUStack(t *testing.T) {
	srv := newFakeServer(t, map[string]string{
		"/v1/chat/completions": `{"id":"1","object":"chat.completion","created":1,"model":"qwen3-8b",
			"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"1:0.9"}}]}`,
		"/v1/embeddings": `{"object":"list","model":"nomic-embed-text-v1.5",
			"data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}]}`,
	})
	cfg := ProviderConfig{GPUStackURL: srv.URL + "/", GPUStackAPIKey: "secret"}
	p, err := NewProviders(cfg, []string{ProviderOllama, ProviderGPUStack}, "gpustack/qwen3-8b", "gpustack/nomic-embed-text-v1.5")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(p.Plugins()...))

	llm, err := p.Model(g, "gpustack/qwen3-8b")
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := p.Model(g, "gpustack/qwen3-8b"); again != llm {
		t.Error("resolving the same spec twice should return the same model")
	}
	resp, err := genkit.Generate(ctx, g, ai.WithModel(llm), ai.WithPrompt("rerank"),
		ai.WithConfig(GenerateConfig("gpustack/qwen3-8b", GenerateOptions{NumCtx: 8192, KeepAlive: "30m", MaxTokens: 120})))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Text() != "1:0.9" {
		t.Errorf("answer = %q", resp.Text())
	}
	body := srv.body("/v1/chat/completions")
	if body["model"] != "qwen3-8b" {
		t.Errorf("model = %v", body["model"])
	}
	if body["max_tokens"] != float64(120) {
		t.Errorf("max_tokens = %v, want 120", body["max_tokens"])
	}
	kwargs, _ := body["chat_template_kwargs"].(map[string]any)
	if kwargs["enable_thinking"] != false {
		t.Errorf("chat_template_kwargs = %v, want enable_thinking=false", body["chat_template_kwargs"])
	}
	for _, ollamaOnly := range []string{"num_ctx", "keep_alive", "options"} {
		if _, ok := body[ollamaOnly]; ok {
			t.Errorf("Ollama-only field %q reached GPUStack", ollamaOnly)
		}
	}
	if got := srv.headers["/v1/chat/completions"].Get("Authorization"); got != "Bearer secret" {
		t.Errorf("Authorization = %q", got)
	}

	emb, err := p.Embedder(g, "gpustack/nomic-embed-text-v1.5", 3)
	if err != nil {
		t.Fatal(err)
	}
	eresp, err := emb.Embed(ctx, &ai.EmbedRequest{Input: []*ai.Document{QueryDocument("Frage")}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(eresp.Embeddings) != 1 || len(eresp.Embeddings[0].Embedding) != 3 {
		t.Errorf("embeddings = %+v", eresp.Embeddings)
	}
	input, _ := srv.body("/v1/embeddings")["input"].([]any)
	if len(input) != 1 || input[0] != "search_query: Frage" {
		t.Errorf("embed input = %v, want the nomic query prefix", input)
	}
}

func TestProvidersGPUStackAPIPath(t *testing.T) {
	srv := newFakeServer(t, map[string]string{
		"/v1-openai/chat/completions": `{"id":"1","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}`,
	})
	p, err := NewProviders(ProviderConfig{GPUStackURL: srv.URL, GPUStackAPIPath: "/v1-openai/"}, nil, "gpustack/m")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(p.Plugins()...))
	llm, err := p.Model(g, "gpustack/m")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := genkit.Generate(ctx, g, ai.WithModel(llm), ai.WithPrompt("hi")); err != nil {
		t.Fatalf("Generate: %v", err)
	}
}

func TestProvidersOllama(t *testing.T) {
	srv := newFakeServer(t, map[string]string{
		"/api/chat":  `{"model":"gemma4:12b","message":{"role":"assistant","content":"ok"},"done":true}`,
		"/api/embed": `{"embeddings":[[0.1,0.2]]}`,
	})
	p, err := NewProviders(ProviderConfig{OllamaHost: srv.URL}, nil, "ollama/gemma4:12b", "mxbai-embed-large")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(p.Plugins()...))

	llm, err := p.Model(g, "ollama/gemma4:12b")
	if err != nil {
		t.Fatal(err)
	}
	opts := GenerateOptions{NumCtx: 8192, KeepAlive: "10m", MaxTokens: 120}
	if _, err := genkit.Generate(ctx, g, ai.WithModel(llm), ai.WithPrompt("hi"), ai.WithConfig(GenerateConfig("ollama/gemma4:12b", opts))); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	body := srv.body("/api/chat")
	options, _ := body["options"].(map[string]any)
	if options["num_ctx"] != float64(8192) || options["num_predict"] != float64(120) {
		t.Errorf("options = %v, want num_ctx=8192 num_predict=120", options)
	}
	if body["keep_alive"] != "10m" || body["think"] != false {
		t.Errorf("keep_alive = %v, think = %v", body["keep_alive"], body["think"])
	}

	emb, err := p.Embedder(g, "mxbai-embed-large", 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := emb.Embed(ctx, &ai.EmbedRequest{Input: []*ai.Document{QueryDocument("Frage")}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	// Ollama sends a single document as a bare string.
	input, _ := srv.body("/api/embed")["input"].(string)
	if !strings.HasPrefix(input, "Represent this sentence") {
		t.Errorf("embed input = %q, want the mxbai query prefix", input)
	}
}

func TestGenerateConfigGoogleAIIsNil(t *testing.T) {
	if cfg := GenerateConfig("googleai/gemini-2.5-flash", GenerateOptions{MaxTokens: 10}); cfg != nil {
		t.Errorf("got %#v, want nil", cfg)
	}
}

func TestProviderConfigFromEnv(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "http://o:1")
	t.Setenv("GPUSTACK_URL", "http://g")
	t.Setenv("GPUSTACK_API_PATH", "")
	t.Setenv("GPUSTACK_API_KEY", "k")
	t.Setenv("LLM_TIMEOUT", "")
	t.Setenv("OLLAMA_TIMEOUT", "300")
	cfg, err := ProviderConfigFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.OllamaHost != "http://o:1" || cfg.GPUStackURL != "http://g" || cfg.GPUStackAPIKey != "k" {
		t.Errorf("cfg = %+v", cfg)
	}
	if cfg.Timeout != 300*time.Second {
		t.Errorf("OLLAMA_TIMEOUT fallback: Timeout = %v", cfg.Timeout)
	}

	t.Setenv("LLM_TIMEOUT", "60")
	if cfg, _ := ProviderConfigFromEnv(); cfg.Timeout != 60*time.Second {
		t.Errorf("LLM_TIMEOUT should win: Timeout = %v", cfg.Timeout)
	}

	t.Setenv("LLM_TIMEOUT", "0")
	if _, err := ProviderConfigFromEnv(); err == nil || !strings.Contains(err.Error(), "LLM_TIMEOUT") {
		t.Errorf("err = %v, want it to name LLM_TIMEOUT", err)
	}
}

func TestGenerateOptionsFromEnv(t *testing.T) {
	t.Setenv("OLLAMA_NUM_CTX", "")
	t.Setenv("OLLAMA_KEEP_ALIVE", "")
	t.Setenv("OLLAMA_THINK", "")
	o, err := GenerateOptionsFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if o.NumCtx != 16384 || o.KeepAlive != "30m" || o.Think {
		t.Errorf("defaults = %+v", o)
	}

	t.Setenv("OLLAMA_NUM_CTX", "8192")
	t.Setenv("OLLAMA_KEEP_ALIVE", "10m")
	t.Setenv("OLLAMA_THINK", "1")
	o, err = GenerateOptionsFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if o.NumCtx != 8192 || o.KeepAlive != "10m" || !o.Think {
		t.Errorf("from env = %+v", o)
	}

	t.Setenv("OLLAMA_NUM_CTX", "-1")
	if _, err := GenerateOptionsFromEnv(); err == nil {
		t.Error("negative OLLAMA_NUM_CTX should be rejected")
	}
}

func TestGPUStackBaseURL(t *testing.T) {
	for _, tc := range []struct{ url, path, want string }{
		{"https://gpustack.example", "", "https://gpustack.example/v1"},
		{"https://gpustack.example/", "", "https://gpustack.example/v1"},
		{"https://gpustack.example/v1", "", "https://gpustack.example/v1"},
		{"https://gpustack.example/v1/", "", "https://gpustack.example/v1"},
		{"https://gpustack.example", "/v1-openai/", "https://gpustack.example/v1-openai"},
		{"https://gpustack.example/v1-openai", "v1-openai", "https://gpustack.example/v1-openai"},
	} {
		if got := gpustackBaseURL(tc.url, tc.path); got != tc.want {
			t.Errorf("gpustackBaseURL(%q, %q) = %q, want %q", tc.url, tc.path, got, tc.want)
		}
	}
}
