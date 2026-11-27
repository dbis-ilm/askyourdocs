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
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
)

// Model discovery: what a model server offers, before any spec is resolved.
// A settings screen lists the models an Ollama or OpenAI-compatible server
// (GPUStack, LM Studio, llama.cpp, Open WebUI) has, tells embedding models
// apart, and asks for the embedding dimension the index must be created
// with. The errors are English; a ServerError's Kind and the URL sentinels
// let an app show them in its own language.

var (
	// ErrInvalidBaseURL: not an http(s) URL with a host.
	ErrInvalidBaseURL = errors.New("not a valid address (http:// or https:// with a host)")
	// ErrCredentialsInURL: user:password in the URL instead of an API key.
	ErrCredentialsInURL = errors.New("credentials do not belong in the address; use an API key")

	// ServerError kinds.
	ErrUnreachable       = errors.New("not reachable")
	ErrAPIKeyRejected    = errors.New("API key rejected")
	ErrServerStatus      = errors.New("unexpected server status")
	ErrNotEmbeddingModel = errors.New("not an embedding model or not available")
	ErrNoEmbedding       = errors.New("server returned no embedding")
)

// ServerError is a failed call to a model server. Kind is one of
// ErrUnreachable, ErrAPIKeyRejected, ErrServerStatus, ErrNotEmbeddingModel or
// ErrNoEmbedding; errors.Is matches it, and Err (if any) is the cause.
type ServerError struct {
	Kind   error
	URL    string // the endpoint called
	Model  string // the model asked about, if any
	Status string // HTTP status for ErrServerStatus and ErrAPIKeyRejected
	Err    error
}

func (e *ServerError) Error() string {
	var b strings.Builder
	if e.Model != "" {
		b.WriteString(e.Model + ": ")
	}
	b.WriteString(e.Kind.Error())
	if e.Status != "" {
		b.WriteString(" (" + e.Status + ")")
	}
	if e.Err != nil {
		b.WriteString(": " + e.Err.Error())
	}
	return b.String()
}

func (e *ServerError) Unwrap() []error {
	if e.Err == nil {
		return []error{e.Kind}
	}
	return []error{e.Kind, e.Err}
}

// listClient answers quickly or not at all: a settings screen should not
// hang on a server that is down. probeClient waits longer, since the first
// embedding request may load the model.
var (
	listClient  = &http.Client{Timeout: 5 * time.Second}
	probeClient = &http.Client{Timeout: 60 * time.Second}
)

// ValidateBaseURL checks an http(s) URL with a host, as a user enters it for
// Ollama or an OpenAI-compatible server, and returns it trimmed and without
// a trailing slash. The error wraps ErrInvalidBaseURL or ErrCredentialsInURL.
func ValidateBaseURL(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", fmt.Errorf("%q: %w", raw, ErrInvalidBaseURL)
	}
	if u.User != nil {
		return "", fmt.Errorf("%q: %w", u.Redacted(), ErrCredentialsInURL)
	}
	return strings.TrimRight(u.String(), "/"), nil
}

// IsLocalURL reports whether a base URL points at this machine (localhost or
// a loopback address). Anything else, a server in the LAN included,
// receives document text over the network — the point where an app should
// ask for consent.
func IsLocalURL(base string) bool {
	u, err := url.Parse(base)
	if err != nil {
		return false
	}
	h := u.Hostname()
	if h == "localhost" {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

// URLHost returns the host[:port] of a base URL, e.g. to name the server in
// a consent dialog; an unparsable URL comes back unchanged.
func URLHost(base string) string {
	if u, err := url.Parse(base); err == nil && u.Host != "" {
		return u.Host
	}
	return base
}

// OllamaModel is one model installed in Ollama.
type OllamaModel struct {
	Name      string `json:"name"`
	Size      int64  `json:"size"` // bytes on disk
	Embedding bool   `json:"embedding"`
	Tools     bool   `json:"tools"` // can call tools (agents); false if Ollama reports no capabilities
}

// ListOllamaModels lists the models installed in the Ollama at host. A
// model counts as an embedding model when Ollama reports the "embedding"
// capability; older Ollama releases report none, and then the name ("embed")
// or a BERT family decides. An error means Ollama is not reachable.
func ListOllamaModels(ctx context.Context, host string) ([]OllamaModel, error) {
	var resp struct {
		Models []struct {
			Name         string   `json:"name"`
			Size         int64    `json:"size"`
			Capabilities []string `json:"capabilities"`
			Details      struct {
				Family string `json:"family"`
			} `json:"details"`
		} `json:"models"`
	}
	endpoint := strings.TrimRight(host, "/") + "/api/tags"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	if err := doJSON(listClient, req, "", &resp); err != nil {
		return nil, err
	}
	out := make([]OllamaModel, 0, len(resp.Models))
	for _, m := range resp.Models {
		emb := slices.Contains(m.Capabilities, "embedding") ||
			(len(m.Capabilities) == 0 && (strings.Contains(m.Name, "embed") || m.Details.Family == "bert"))
		out = append(out, OllamaModel{Name: m.Name, Size: m.Size, Embedding: emb, Tools: slices.Contains(m.Capabilities, "tools")})
	}
	return out, nil
}

// OllamaEmbeddingDim returns the embedding length Ollama reports for model
// (its "*.embedding_length" model info). A model without one is not an
// embedding model (ErrNotEmbeddingModel).
func OllamaEmbeddingDim(ctx context.Context, host, model string) (int, error) {
	payload, _ := json.Marshal(map[string]string{"model": model})
	endpoint := strings.TrimRight(host, "/") + "/api/show"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	var info struct {
		ModelInfo map[string]any `json:"model_info"`
	}
	if err := doJSON(listClient, req, "", &info); err != nil {
		var se *ServerError
		if errors.As(err, &se) && se.Kind == ErrServerStatus {
			se.Kind, se.Model = ErrNotEmbeddingModel, model // Ollama answers 404 for an unknown model
		}
		return 0, err
	}
	for k, v := range info.ModelInfo {
		if strings.HasSuffix(k, ".embedding_length") {
			if f, ok := v.(float64); ok && f > 0 {
				return int(f), nil
			}
		}
	}
	return 0, &ServerError{Kind: ErrNotEmbeddingModel, URL: endpoint, Model: model}
}

// OllamaCapabilities returns what Ollama reports a model can do, e.g.
// "completion", "tools", "vision", "thinking", "embedding" (/api/show).
func OllamaCapabilities(ctx context.Context, host, model string) ([]string, error) {
	payload, _ := json.Marshal(map[string]string{"model": model})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(host, "/")+"/api/show", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	var info struct {
		Capabilities []string `json:"capabilities"`
	}
	if err := doJSON(listClient, req, "", &info); err != nil {
		return nil, err
	}
	return info.Capabilities, nil
}

// ListOpenAIModels lists the model IDs an OpenAI-compatible server offers
// under base (e.g. https://gpustack.example/v1). Such servers do not say
// what a model is for; see LooksLikeEmbeddingModel and OpenAIEmbeddingDim.
func ListOpenAIModels(ctx context.Context, base, apiKey string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/models", nil)
	if err != nil {
		return nil, err
	}
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := doJSON(listClient, req, apiKey, &body); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(body.Data))
	for _, m := range body.Data {
		if m.ID != "" {
			out = append(out, m.ID)
		}
	}
	return out, nil
}

// OpenAIEmbeddingDim embeds a short probe text with model and returns the
// vector length. That also proves model is an embedding model: a chat model
// fails with ErrNotEmbeddingModel.
func OpenAIEmbeddingDim(ctx context.Context, base, apiKey, model string) (int, error) {
	payload, _ := json.Marshal(map[string]any{"model": model, "input": "probe"})
	endpoint := strings.TrimRight(base, "/") + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	var body struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := doJSON(probeClient, req, apiKey, &body); err != nil {
		var se *ServerError
		if errors.As(err, &se) && se.Kind == ErrServerStatus {
			se.Kind, se.Model = ErrNotEmbeddingModel, model
		}
		return 0, err
	}
	if len(body.Data) == 0 || len(body.Data[0].Embedding) == 0 {
		return 0, &ServerError{Kind: ErrNoEmbedding, URL: endpoint, Model: model}
	}
	return len(body.Data[0].Embedding), nil
}

// LooksLikeEmbeddingModel guesses from a model name whether it is an
// embedding model (bge, e5, gte, nomic, …; rerankers excluded) — enough to
// sort a list from ListOpenAIModels. OpenAIEmbeddingDim gives certainty.
func LooksLikeEmbeddingModel(name string) bool {
	n := strings.ToLower(name)
	for _, k := range []string{"embed", "bge-", "bge_", "e5-", "gte-", "minilm", "jina-emb", "nomic"} {
		if strings.Contains(n, k) {
			return !strings.Contains(n, "rerank")
		}
	}
	return false
}

// doJSON sends req (with a bearer key, if any) and decodes a JSON answer
// into v. Failures come back as *ServerError.
func doJSON(client *http.Client, req *http.Request, apiKey string, v any) error {
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	endpoint := req.URL.Redacted()
	res, err := client.Do(req)
	if err != nil {
		return &ServerError{Kind: ErrUnreachable, URL: endpoint, Err: err}
	}
	defer res.Body.Close()
	switch {
	case res.StatusCode == http.StatusUnauthorized || res.StatusCode == http.StatusForbidden:
		return &ServerError{Kind: ErrAPIKeyRejected, URL: endpoint, Status: res.Status}
	case res.StatusCode != http.StatusOK:
		return &ServerError{Kind: ErrServerStatus, URL: endpoint, Status: res.Status}
	}
	if err := json.NewDecoder(res.Body).Decode(v); err != nil {
		return &ServerError{Kind: ErrServerStatus, URL: endpoint, Status: res.Status, Err: fmt.Errorf("unexpected answer: %w", err)}
	}
	return nil
}
