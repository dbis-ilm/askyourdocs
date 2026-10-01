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
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func newTestToolContext() *ai.ToolContext {
	return &ai.ToolContext{Context: context.Background()}
}

func TestMCPModeRequestedDetectsFlag(t *testing.T) {
	origArgs := os.Args
	defer func() { os.Args = origArgs }()

	os.Args = []string{"app"}
	if MCPModeRequested() {
		t.Error("MCPModeRequested() = true without -mcp/--mcp, want false")
	}
	os.Args = []string{"app", "-mcp"}
	if !MCPModeRequested() {
		t.Error("MCPModeRequested() = false with -mcp, want true")
	}
	os.Args = []string{"app", "--mcp"}
	if !MCPModeRequested() {
		t.Error("MCPModeRequested() = false with --mcp, want true")
	}
	os.Args = []string{"app", "-other"}
	if MCPModeRequested() {
		t.Error("MCPModeRequested() = true with an unrelated flag, want false")
	}
}

func TestJSONToolHandlerMarshalsResult(t *testing.T) {
	type result struct {
		Answer string `json:"answer"`
	}
	handler := JSONToolHandler(func(ctx context.Context, input string) (result, error) {
		return result{Answer: "got: " + input}, nil
	})

	out, err := handler(newTestToolContext(), "question")
	if err != nil {
		t.Fatalf("handler: %v", err)
	}

	var parsed result
	if err := json.Unmarshal([]byte(out), &parsed); err != nil {
		t.Fatalf("result is not valid JSON: %v (result=%q)", err, out)
	}
	if parsed.Answer != "got: question" {
		t.Errorf("answer = %q", parsed.Answer)
	}
}

func TestJSONToolHandlerPropagatesError(t *testing.T) {
	handler := JSONToolHandler(func(ctx context.Context, input string) (string, error) {
		return "", errors.New("failed")
	})

	out, err := handler(newTestToolContext(), "x")
	if err == nil {
		t.Fatal("expected an error to propagate")
	}
	if out != "" {
		t.Errorf("result = %q, want empty on error", out)
	}
}
