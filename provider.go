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
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/ollama"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// Model providers a spec can name. A spec is "<provider>/<model>", e.g.
// "ollama/gemma4:12b", "gpustack/qwen3-8b" or "googleai/gemini-2.5-flash".
const (
	ProviderOllama   = "ollama"
	ProviderGPUStack = "gpustack"
	ProviderGoogleAI = "googleai"
)

// ParseModelSpec splits a model spec into provider and model name. A spec
// without a known provider prefix is an Ollama model: EMBEDDING_MODEL has
// always been a bare Ollama name, and Ollama names may themselves contain a
// slash ("hf.co/<user>/<repo>"), so only the three known prefixes are split
// off.
func ParseModelSpec(spec string) (provider, name string) {
	if p, n, ok := strings.Cut(spec, "/"); ok {
		switch p {
		case ProviderOllama, ProviderGPUStack, ProviderGoogleAI:
			return p, n
		}
	}
	return ProviderOllama, spec
}

// ProviderConfig is how to reach each provider. Only the providers named by
// the specs passed to NewProviders are set up, so a field for an unused
// provider may stay empty.
type ProviderConfig struct {
	// OllamaHost is the Ollama server, default http://localhost:11434.
	OllamaHost string
	// GPUStackURL is the base URL of the OpenAI-compatible API, used as is:
	// e.g. https://gpustack.example/v1 for GPUStack, or
	// https://openwebui.example/api for Open WebUI. GPUStackAPIPath, if set,
	// is appended unless the URL already ends in it (older GPUStack
	// releases: server URL plus "/v1-openai").
	GPUStackURL     string
	GPUStackAPIPath string
	GPUStackAPIKey  string
	// GoogleAIAPIKey is optional; empty means the plugin reads GEMINI_API_KEY
	// or GOOGLE_API_KEY itself.
	GoogleAIAPIKey string
	// Timeout bounds every call to a self-hosted provider (Ollama, GPUStack).
	// It has to cover the slowest single call an app makes, typically a
	// rerank batch. Zero means 120s.
	Timeout time.Duration
}

// ProviderConfigFromEnv reads the conventional environment variables:
// OLLAMA_HOST, GPUSTACK_URL, GPUSTACK_API_PATH, GPUSTACK_API_KEY and
// LLM_TIMEOUT (seconds; OLLAMA_TIMEOUT is still honored when LLM_TIMEOUT is
// unset). The Google AI key is left to the plugin's own GEMINI_API_KEY
// lookup.
func ProviderConfigFromEnv() (ProviderConfig, error) {
	cfg := ProviderConfig{
		OllamaHost:      os.Getenv("OLLAMA_HOST"),
		GPUStackURL:     os.Getenv("GPUSTACK_URL"),
		GPUStackAPIPath: os.Getenv("GPUSTACK_API_PATH"),
		GPUStackAPIKey:  os.Getenv("GPUSTACK_API_KEY"),
	}
	name := "LLM_TIMEOUT"
	v := os.Getenv(name)
	if v == "" {
		name = "OLLAMA_TIMEOUT"
		v = os.Getenv(name)
	}
	if v != "" {
		secs, err := strconv.Atoi(v)
		if err != nil || secs <= 0 {
			return cfg, fmt.Errorf("%s must be a positive number of seconds, got %q", name, v)
		}
		cfg.Timeout = time.Duration(secs) * time.Second
	}
	return cfg, nil
}

// Providers sets up the Genkit plugins a set of model specs needs and
// resolves those specs to models and embedders. Typical use:
//
//	p, err := askyourdocs.NewProviders(cfg, nil, llmSpec, rerankSpec, embeddingSpec)
//	genk := genkit.Init(ctx, genkit.WithPlugins(p.Plugins()...))
//	llm, err := p.Model(genk, llmSpec)
//	emb, err := p.Embedder(genk, embeddingSpec, 1024)
type Providers struct {
	ollama   *ollama.Ollama
	gpustack *compat_oai.OpenAICompatible
	googleAI *googlegenai.GoogleAI

	gpustackBase   string // API base URL, for CrossEncoder
	gpustackAPIKey string
	timeout        time.Duration

	mu        sync.Mutex
	models    map[string]ai.Model
	embedders map[string]ai.Embedder
}

// NewProviders validates specs and prepares a plugin for each provider they
// name. allowed restricts which providers may be named at all — e.g. an app
// whose data must not leave the own infrastructure passes
// []string{ProviderOllama, ProviderGPUStack}; nil allows every provider.
// Empty specs are skipped, so an optional setting can be passed as is.
func NewProviders(cfg ProviderConfig, allowed []string, specs ...string) (*Providers, error) {
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	p := &Providers{
		models:    map[string]ai.Model{},
		embedders: map[string]ai.Embedder{},
		timeout:   timeout,
	}
	for _, spec := range specs {
		if spec == "" {
			continue
		}
		provider, name := ParseModelSpec(spec)
		if name == "" {
			return nil, fmt.Errorf("model spec %q names no model", spec)
		}
		if allowed != nil && !contains(allowed, provider) {
			return nil, fmt.Errorf("model spec %q: provider %s is not allowed here (allowed: %s)", spec, provider, strings.Join(allowed, ", "))
		}
		switch provider {
		case ProviderOllama:
			if p.ollama == nil {
				host := cfg.OllamaHost
				if host == "" {
					host = "http://localhost:11434"
				}
				p.ollama = &ollama.Ollama{ServerAddress: host, Timeout: int(timeout / time.Second)}
			}
		case ProviderGPUStack:
			if p.gpustack == nil {
				if cfg.GPUStackURL == "" {
					return nil, fmt.Errorf("model spec %q needs GPUSTACK_URL", spec)
				}
				p.gpustackBase = gpustackBaseURL(cfg.GPUStackURL, cfg.GPUStackAPIPath)
				p.gpustackAPIKey = cfg.GPUStackAPIKey
				p.gpustack = &compat_oai.OpenAICompatible{
					Provider: ProviderGPUStack,
					BaseURL:  p.gpustackBase,
					APIKey:   cfg.GPUStackAPIKey,
					Opts:     []option.RequestOption{option.WithRequestTimeout(timeout)},
				}
			}
		case ProviderGoogleAI:
			if p.googleAI == nil {
				p.googleAI = &googlegenai.GoogleAI{APIKey: cfg.GoogleAIAPIKey}
			}
		}
	}
	return p, nil
}

// gpustackBaseURL is the API base URL: serverURL as is, with apiPath
// appended only when one is set and serverURL doesn't already end in it.
// Nothing is appended by default: the GPUStack UI shows the API endpoint
// with /v1, and other OpenAI-compatible servers (Open WebUI: /api) use
// other paths.
func gpustackBaseURL(serverURL, apiPath string) string {
	base := strings.TrimRight(serverURL, "/")
	path := strings.Trim(apiPath, "/")
	if path == "" || strings.HasSuffix(base, "/"+path) {
		return base
	}
	return base + "/" + path
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// Plugins returns the plugins to hand to genkit.Init, one per provider the
// specs named.
func (p *Providers) Plugins() []api.Plugin {
	var plugins []api.Plugin
	if p.ollama != nil {
		plugins = append(plugins, p.ollama)
	}
	if p.gpustack != nil {
		plugins = append(plugins, p.gpustack)
	}
	if p.googleAI != nil {
		plugins = append(plugins, p.googleAI)
	}
	return plugins
}

// Model resolves a chat model spec. g must have been initialized with
// Plugins(). Resolving the same spec twice returns the same model, so an app
// can resolve its answer and rerank model without checking whether they are
// equal.
func (p *Providers) Model(g *genkit.Genkit, spec string) (ai.Model, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if m, ok := p.models[spec]; ok {
		return m, nil
	}
	provider, name := ParseModelSpec(spec)
	var m ai.Model
	switch {
	case provider == ProviderOllama && p.ollama != nil:
		m = p.ollama.DefineModel(g, ollama.ModelDefinition{Name: name, Type: "chat"}, nil)
	case provider == ProviderGPUStack && p.gpustack != nil:
		// GPUStackConfig rather than the SDK's params (NewModel): those have
		// no field for chat_template_kwargs, which GenerateConfig needs to
		// switch thinking off.
		action := compat_oai.NewChatModel[GPUStackConfig](p.gpustack, name, ai.ModelOptions{Supports: &compat_oai.BasicText})
		genkit.RegisterAction(g, action)
		m = action
	case provider == ProviderGoogleAI && p.googleAI != nil:
		m = genkit.LookupModel(g, api.NewName(ProviderGoogleAI, name))
	default:
		return nil, fmt.Errorf("model spec %q was not passed to NewProviders", spec)
	}
	if m == nil {
		return nil, fmt.Errorf("model %q could not be defined", spec)
	}
	p.models[spec] = m
	return m, nil
}

// CrossEncoder resolves a reranker model spec to a CrossEncoder. Only
// gpustack specs are supported: Ollama has no rerank endpoint, Google AI no
// cross-encoder models. The spec must have been passed to NewProviders. The
// server behind GPUSTACK_URL has to serve <base>/rerank — GPUStack does, an
// Open WebUI proxy does not.
func (p *Providers) CrossEncoder(spec string) (*CrossEncoder, error) {
	provider, name := ParseModelSpec(spec)
	if provider != ProviderGPUStack {
		return nil, fmt.Errorf("cross-encoder spec %q: only %s/<model> is supported", spec, ProviderGPUStack)
	}
	if p.gpustack == nil {
		return nil, fmt.Errorf("cross-encoder spec %q was not passed to NewProviders", spec)
	}
	return NewCrossEncoder(p.gpustackBase, p.gpustackAPIKey, name, p.timeout), nil
}

// Embedder resolves an embedding model spec to an embedder producing dim-sized
// vectors, wrapped with NewPrefixEmbedder for models that expect instruction
// prefixes. g must have been initialized with Plugins().
func (p *Providers) Embedder(g *genkit.Genkit, spec string, dim int) (ai.Embedder, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if e, ok := p.embedders[spec]; ok {
		return e, nil
	}
	if dim <= 0 {
		return nil, fmt.Errorf("embedder %q: dimension must be positive, got %d", spec, dim)
	}
	provider, name := ParseModelSpec(spec)
	var e ai.Embedder
	switch {
	case provider == ProviderOllama && p.ollama != nil:
		e = p.ollama.DefineEmbedder(g, name, dim, nil)
	case provider == ProviderGPUStack && p.gpustack != nil:
		action := p.gpustack.NewEmbedder(name, &ai.EmbedderOptions{Dimensions: dim})
		genkit.RegisterAction(g, action)
		e = action
	case provider == ProviderGoogleAI && p.googleAI != nil:
		if e = genkit.LookupEmbedder(g, api.NewName(ProviderGoogleAI, name)); e != nil {
			e = &outputDimEmbedder{Embedder: e, dim: dim}
		}
	default:
		return nil, fmt.Errorf("embedder spec %q was not passed to NewProviders", spec)
	}
	if e == nil {
		return nil, fmt.Errorf("embedder %q could not be defined", spec)
	}
	e = NewPrefixEmbedder(e, name)
	p.embedders[spec] = e
	return e, nil
}

// GenerateOptions are the per-request settings the RAG pipeline cares about,
// independent of provider. Zero values leave the provider's default.
type GenerateOptions struct {
	// NumCtx is Ollama's context window. One value for every call of an app,
	// deliberately — Ollama reloads the model whenever num_ctx changes.
	// GPUStack fixes the context when the model is deployed; ignored there.
	NumCtx int
	// KeepAlive is how long Ollama holds the model in memory after a
	// request, e.g. "30m". Ignored by other providers.
	KeepAlive string
	// Think enables the model's reasoning phase. Off by default: every
	// prompt in these pipelines asks for a fixed shape and reasoning tokens
	// are never read, while a thinking model can spend thousands of tokens
	// on a five-line rerank batch.
	Think bool
	// MaxTokens caps the generated tokens, e.g. for a rerank call that only
	// has to emit one short line per chunk.
	MaxTokens int
}

// GenerateOptionsFromEnv reads OLLAMA_NUM_CTX (default 16384),
// OLLAMA_KEEP_ALIVE (default 30m) and OLLAMA_THINK (any non-empty value
// enables thinking).
func GenerateOptionsFromEnv() (GenerateOptions, error) {
	o := GenerateOptions{NumCtx: 16384, KeepAlive: "30m", Think: os.Getenv("OLLAMA_THINK") != ""}
	if v := os.Getenv("OLLAMA_NUM_CTX"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			return o, fmt.Errorf("OLLAMA_NUM_CTX must be a positive number of tokens, got %q", v)
		}
		o.NumCtx = n
	}
	if v := os.Getenv("OLLAMA_KEEP_ALIVE"); v != "" {
		o.KeepAlive = v
	}
	return o, nil
}

// GenerateConfig turns o into the config type spec's provider expects, for
// ai.WithConfig. It returns nil for Google AI: its defaults are fine, and a
// token cap there would also count thinking tokens and could cut the answer
// off.
func GenerateConfig(spec string, o GenerateOptions) any {
	provider, _ := ParseModelSpec(spec)
	switch provider {
	case ProviderOllama:
		cfg := &ollama.GenerateContentConfig{
			KeepAlive: o.KeepAlive,
			Think:     ollama.ThinkEnabled(o.Think),
		}
		if o.NumCtx > 0 {
			cfg.NumCtx = ollama.Ptr(o.NumCtx)
		}
		if o.MaxTokens > 0 {
			cfg.NumPredict = ollama.Ptr(o.MaxTokens)
		}
		return cfg
	case ProviderGPUStack:
		think := o.Think
		return &GPUStackConfig{MaxOutputTokens: o.MaxTokens, EnableThinking: &think}
	}
	return nil
}

// outputDimEmbedder asks a Google AI embedder for dim-sized vectors. The
// API returns the model's default size (3072 for gemini-embedding-001)
// unless the request sets outputDimensionality; the plugin's
// EmbedderOptions.Dimensions only describes the embedder and isn't sent.
// A request that already carries options is passed on unchanged.
type outputDimEmbedder struct {
	ai.Embedder
	dim int
}

func (e *outputDimEmbedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	if req != nil && req.Options == nil {
		next := *req
		// The plugin converts the map into its genai.EmbedContentConfig.
		next.Options = map[string]any{"outputDimensionality": e.dim}
		req = &next
	}
	return e.Embedder.Embed(ctx, req)
}

// GPUStackConfig is the request config of a GPUStack chat model, what
// GenerateConfig returns for a gpustack spec. GPUStack serves through vLLM,
// SGLang or llama.cpp; all three take max_tokens, and switch thinking via
// the chat template's enable_thinking flag (Qwen3 and others). A template
// without that flag ignores it. Anything else a backend accepts goes in
// RequestConfig.Extra, keyed by its wire name.
type GPUStackConfig struct {
	compat_oai.RequestConfig

	// MaxOutputTokens caps the generated tokens, sent as max_tokens; 0
	// leaves it to the server.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=0" jsonschema_description:"Maximum number of tokens to generate, sent as max_tokens. 0 leaves it to the server."`
	// EnableThinking switches the model's thinking on or off, sent as
	// chat_template_kwargs.enable_thinking; nil leaves the template default.
	EnableThinking *bool `json:"enableThinking,omitempty" jsonschema_description:"Switches thinking on or off, sent as chat_template_kwargs.enable_thinking. Unset leaves the chat template's default."`
}

// ApplyToChatCompletion implements compat_oai.ChatConfig.
func (c GPUStackConfig) ApplyToChatCompletion(params *openai.ChatCompletionNewParams) {
	c.ApplyVersion(params)
	if c.MaxOutputTokens > 0 {
		params.MaxTokens = openai.Int(int64(c.MaxOutputTokens))
	}
	if c.EnableThinking != nil {
		compat_oai.AddExtraFields(params, map[string]any{
			"chat_template_kwargs": map[string]any{"enable_thinking": *c.EnableThinking},
		})
	}
}
