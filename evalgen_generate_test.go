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
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// defineFakeLLM registers a model that answers every prompt with reply.
func defineFakeLLM(g *genkit.Genkit, name, reply string) ai.Model {
	return genkit.DefineModel(g, name, &ai.ModelOptions{Supports: &ai.ModelSupports{}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Request: req,
				Message: ai.NewModelTextMessage(reply),
			}, nil
		})
}

// longCrawlMD returns crawl-data Markdown with a body long enough (>200
// chars) to pass GenerateEvalSet's minimum-content filter.
func longCrawlMD() string {
	return "---\nurl: https://a.example/\n---\n" + strings.Repeat("Dies ist ein ausreichend langer Satz über das Studium. ", 12)
}

func TestGenerateEvalSetWritesDataset(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	llm := defineFakeLLM(g, "test/qa", `Hier: {"question": "Was ist das Studium?", "answer": "Ein Studium."}`)

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "page.md"), []byte(longCrawlMD()), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "eval.json")

	res, err := GenerateEvalSet(ctx, g, []ai.GenerateOption{ai.WithModel(llm)}, 300, 30,
		GenerateEvalSetInput{Dir: dir, SampleRate: 1, MaxPerDoc: 2, OutPath: out},
		func(q string) string { return "IN:" + q })
	if err != nil {
		t.Fatalf("GenerateEvalSet: %v", err)
	}
	if res.Written == 0 || res.Written > 2 || res.Documents != 1 || res.Skipped != 0 || res.OutPath != out {
		t.Fatalf("result = %+v", res)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var cases []EvalCase[string]
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("dataset not valid JSON: %v", err)
	}
	if len(cases) != res.Written {
		t.Fatalf("cases = %d, want %d", len(cases), res.Written)
	}
	c := cases[0]
	if c.Input != "IN:Was ist das Studium?" || c.Reference != "Ein Studium." || !strings.HasPrefix(c.TestCaseId, "page#") {
		t.Errorf("case = %+v", c)
	}
}

func TestGenerateEvalSetCountsUnparsableAsSkipped(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	llm := defineFakeLLM(g, "test/bad", "kein json")

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "page.md"), []byte(longCrawlMD()), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "eval.json")

	res, err := GenerateEvalSet(ctx, g, []ai.GenerateOption{ai.WithModel(llm)}, 300, 30,
		GenerateEvalSetInput{Dir: dir, SampleRate: 1, OutPath: out},
		func(q string) string { return q })
	if err != nil {
		t.Fatalf("GenerateEvalSet: %v", err)
	}
	if res.Written != 0 || res.Skipped == 0 {
		t.Errorf("result = %+v, want 0 written and >0 skipped", res)
	}
}

func TestGenerateEvalSetNoDocumentsIsAnError(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	_, err := GenerateEvalSet(ctx, g, nil, 300, 30,
		GenerateEvalSetInput{Dir: t.TempDir(), OutPath: filepath.Join(t.TempDir(), "o.json")},
		func(q string) string { return q })
	if err == nil {
		t.Error("expected error for a directory without documents")
	}
}
