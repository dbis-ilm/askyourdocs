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
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// EvalCase is one entry of a genkit eval:flow dataset: a question (Input, in
// whatever shape the flow being evaluated expects) and a reference answer to
// score the flow's output against.
type EvalCase[In any] struct {
	TestCaseId string `json:"testCaseId"`
	Input      In     `json:"input"`
	Reference  string `json:"reference"`
}

// GenerateEvalSetInput configures GenerateEvalSet: which documents to draw
// questions from and how densely to sample their chunks.
type GenerateEvalSetInput struct {
	Dir        string `json:"dir,omitempty"`        // directory of .pdf/.docx files to read; default "./pdf-data"
	SampleRate int    `json:"sampleRate,omitempty"` // take every Nth chunk per document; default 4
	MaxPerDoc  int    `json:"maxPerDoc,omitempty"`  // cap questions generated per document; default 8
	OutPath    string `json:"outPath,omitempty"`    // dataset output path; default "testdata/eval_dataset_generated.json"
}

type GenerateEvalSetResult struct {
	Written   int    `json:"written"`
	OutPath   string `json:"outPath"`
	Documents int    `json:"documents"`
	Skipped   int    `json:"skipped"` // chunks where the LLM response could not be parsed
}

// GenerateEvalSet turns the PDFs and .docx files in cfg.Dir into a
// question/reference-answer dataset for `genkit eval:flow`. For a sample of
// chunks per document it asks the LLM (via genOpts — typically
// []ai.GenerateOption{ai.WithModel(llm), ai.WithConfig(modelConfig)}) for one
// question that chunk answers plus a short reference answer, grounded only
// in that chunk's parent text — the same context window BuildChunks hands to
// retrieval, so generated questions match what the flow actually has to work
// with. buildInput turns each generated question into the input shape the
// flow being evaluated expects (e.g. a plain string, or a struct carrying
// extra fields like tags) — EvalCase.Input must match that flow's input type
// for `genkit eval:flow` to run against it.
func GenerateEvalSet[In any](ctx context.Context, genk *genkit.Genkit, genOpts []ai.GenerateOption, chunkSize, chunkOverlap int, cfg GenerateEvalSetInput, buildInput func(question string) In) (GenerateEvalSetResult, error) {
	dir := cfg.Dir
	if dir == "" {
		dir = "./pdf-data"
	}
	sampleRate := cfg.SampleRate
	if sampleRate <= 0 {
		sampleRate = 4
	}
	maxPerDoc := cfg.MaxPerDoc
	if maxPerDoc <= 0 {
		maxPerDoc = 8
	}
	outPath := cfg.OutPath
	if outPath == "" {
		outPath = "testdata/eval_dataset_generated.json"
	}

	paths, err := globDocuments(dir)
	if err != nil {
		return GenerateEvalSetResult{}, err
	}
	if len(paths) == 0 {
		return GenerateEvalSetResult{}, fmt.Errorf("no PDFs or .docx files found in %s", dir)
	}

	var cases []EvalCase[In]
	skipped := 0
	for _, path := range paths {
		docName := filepath.Base(path)
		paras, err := ExtractParagraphs(path)
		if err != nil {
			log.Printf("[GenerateEvalSet] %s: read failed: %v", docName, err)
			continue
		}
		chunks := BuildChunks(paras, chunkSize, chunkOverlap)

		generated := 0
		for i := 0; i < len(chunks) && generated < maxPerDoc; i += sampleRate {
			text := chunks[i].ParentText
			if text == "" {
				text = chunks[i].Text
			}
			if len(strings.TrimSpace(text)) < 200 {
				continue // too little content to ask a meaningful question
			}

			question, answer, err := generateQAPair(ctx, genk, genOpts, text)
			if err != nil {
				log.Printf("[GenerateEvalSet] %s chunk %d: %v", docName, i, err)
				skipped++
				continue
			}

			cases = append(cases, EvalCase[In]{
				TestCaseId: fmt.Sprintf("%s#%d", strings.TrimSuffix(docName, filepath.Ext(docName)), i),
				Input:      buildInput(question),
				Reference:  answer,
			})
			generated++
		}
		log.Printf("[GenerateEvalSet] %s: %d questions generated", docName, generated)
	}

	data, err := json.MarshalIndent(cases, "", "  ")
	if err != nil {
		return GenerateEvalSetResult{}, fmt.Errorf("marshal dataset: %w", err)
	}
	if err := os.WriteFile(outPath, data, 0o644); err != nil {
		return GenerateEvalSetResult{}, fmt.Errorf("write %s: %w", outPath, err)
	}

	return GenerateEvalSetResult{
		Written:   len(cases),
		OutPath:   outPath,
		Documents: len(paths),
		Skipped:   skipped,
	}, nil
}

// globDocuments returns every .pdf and .docx path directly inside dir,
// sorted — the same two extensions ExtractParagraphs knows how to read.
func globDocuments(dir string) ([]string, error) {
	pdfPaths, err := filepath.Glob(filepath.Join(dir, "*.pdf"))
	if err != nil {
		return nil, fmt.Errorf("glob %s: %w", dir, err)
	}
	docxPaths, err := filepath.Glob(filepath.Join(dir, "*.docx"))
	if err != nil {
		return nil, fmt.Errorf("glob %s: %w", dir, err)
	}
	paths := append(pdfPaths, docxPaths...)
	sort.Strings(paths)
	return paths, nil
}

// generateQAPair asks the LLM for one question the given text answers,
// together with a short reference answer grounded only in that text.
func generateQAPair(ctx context.Context, genk *genkit.Genkit, genOpts []ai.GenerateOption, text string) (question, answer string, err error) {
	prompt := fmt.Sprintf(`Du bekommst einen Ausschnitt aus einem Dokument. Formuliere GENAU EINE Frage,
die sich ausschließlich mit den Informationen aus diesem Ausschnitt beantworten lässt,
plus eine kurze, präzise Referenzantwort (1-3 Sätze) mit den relevanten Fakten.

Die Frage soll klingen, als würde sie eine Person stellen, die das Dokument nicht kennt
(keine Formulierungen wie "im Ausschnitt" oder "im Text"), muss aber spezifisch genug
sein, dass nur dieser Textabschnitt sie beantwortet.

AUSSCHNITT:
%s

Antworte NUR mit einem JSON-Objekt: {"question": "...", "answer": "..."}`, text)

	opts := append(append([]ai.GenerateOption{}, genOpts...), ai.WithPrompt(prompt))

	resp, err := genkit.Generate(ctx, genk, opts...)
	if err != nil {
		return "", "", fmt.Errorf("generate: %w", err)
	}

	return parseGeneratedQA(resp.Text())
}

// parseGeneratedQA extracts {"question","answer"} from the LLM response,
// tolerating extra text around the JSON object the way parseJudgeResponse
// does for evaluator scores.
func parseGeneratedQA(text string) (question, answer string, err error) {
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start < 0 || end <= start {
		return "", "", fmt.Errorf("no JSON object in response: %q", text)
	}

	var result struct {
		Question string `json:"question"`
		Answer   string `json:"answer"`
	}
	if err := json.Unmarshal([]byte(text[start:end+1]), &result); err != nil {
		return "", "", fmt.Errorf("parse response: %w", err)
	}
	if strings.TrimSpace(result.Question) == "" || strings.TrimSpace(result.Answer) == "" {
		return "", "", fmt.Errorf("empty question or answer in response")
	}
	return strings.TrimSpace(result.Question), strings.TrimSpace(result.Answer), nil
}
