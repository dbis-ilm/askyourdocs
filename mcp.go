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
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/mcp"
)

// MCPModeRequested reports whether the process was started with -mcp or
// --mcp — the usual way an app offers "run as an MCP stdio server instead of
// the HTTP server" without needing a full flag-parsing setup just for this
// one switch.
func MCPModeRequested() bool {
	for _, arg := range os.Args[1:] {
		if arg == "-mcp" || arg == "--mcp" {
			return true
		}
	}
	return false
}

// JSONToolHandler wraps run so its result is returned as a JSON string.
// Genkit's MCP server stringifies any non-string tool result with Go's "%v"
// (e.g. "map[chunks:83 source:...]") instead of JSON, so every MCP tool
// needs this same marshal-to-string wrapping to give clients valid,
// parseable JSON back — this is that wrapping, written once.
func JSONToolHandler[In, Out any](run func(ctx context.Context, input In) (Out, error)) func(*ai.ToolContext, In) (string, error) {
	return func(ctx *ai.ToolContext, input In) (string, error) {
		out, err := run(ctx, input)
		if err != nil {
			return "", err
		}
		b, err := json.Marshal(out)
		return string(b), err
	}
}

// ServeMCPStdio serves g's registered tools (see genkit.DefineTool) over
// stdio, blocking until the client disconnects.
func ServeMCPStdio(g *genkit.Genkit, name, version string) error {
	srv := mcp.NewMCPServer(g, mcp.MCPServerOptions{
		Name:    name,
		Version: version,
	})
	return srv.ServeStdio()
}
