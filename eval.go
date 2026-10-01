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
	"fmt"
	"strconv"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// RegisterEvaluators defines custom LLM-as-judge evaluators for RAG quality:
// custom/faithfulness and custom/answerRelevancy, usable with
// `genkit eval:flow <flow> --evaluators=custom/faithfulness,custom/answerRelevancy`
// against any flow whose output is a string or a map/struct with an
// "answer" field, and whose input is a string or a map/struct with a
// "question" field.
func RegisterEvaluators(g *genkit.Genkit, modelName string) {
	// ── Faithfulness: is the answer grounded in the retrieved context? ──
	genkit.DefineEvaluator(g, api.NewName("custom", "faithfulness"),
		&ai.EvaluatorOptions{
			DisplayName: "Faithfulness",
			Definition:  "Checks whether the answer is supported by the retrieved context. Score 0.0–1.0.",
		},
		func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
			answer, err := extractAnswer(req.Input.Output)
			if err != nil {
				return nil, err
			}

			contextStr := formatContext(req.Input.Context)
			// Fallback: read retrieval chunks from the flow output when Genkit
			// hasn't captured them automatically (custom retriever, no ai.Retrieve).
			if contextStr == "" {
				if m, ok := req.Input.Output.(map[string]any); ok {
					if chunks, ok := m["context"].([]any); ok {
						contextStr = formatContext(chunks)
					}
				}
			}
			if contextStr == "" {
				return &ai.EvaluatorCallbackResponse{
					TestCaseId: req.Input.TestCaseId,
					Evaluation: []ai.Score{{
						Id:    "faithfulness",
						Score: 0.0,
						Details: map[string]any{
							"reason": "no context available",
						},
					}},
				}, nil
			}

			prompt := fmt.Sprintf(`You are an impartial judge evaluating faithfulness of an AI answer.

CONTEXT (retrieved documents):
%s

ANSWER:
%s

Task: Determine what fraction of the claims in the ANSWER are supported by the CONTEXT.
- 1.0 = every claim is fully supported
- 0.0 = no claims are supported
- Values in between reflect partial support

Respond with ONLY a JSON object: {"score": <float>, "reason": "<brief explanation>"}`, contextStr, answer)

			resp, err := genkit.Generate(ctx, g,
				ai.WithModelName(modelName),
				ai.WithPrompt(prompt),
			)
			if err != nil {
				return nil, fmt.Errorf("faithfulness judge failed: %w", err)
			}

			score, reason := parseJudgeResponse(resp.Text())
			return &ai.EvaluatorCallbackResponse{
				TestCaseId: req.Input.TestCaseId,
				Evaluation: []ai.Score{{
					Id:    "faithfulness",
					Score: score,
					Details: map[string]any{
						"reason": reason,
					},
				}},
			}, nil
		},
	)

	// ── Answer Relevancy: does the answer actually address the question? ──
	genkit.DefineEvaluator(g, api.NewName("custom", "answerRelevancy"),
		&ai.EvaluatorOptions{
			DisplayName: "Answer Relevancy",
			Definition:  "Checks whether the answer addresses the original question. Score 0.0–1.0.",
		},
		func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
			answer, err := extractAnswer(req.Input.Output)
			if err != nil {
				return nil, err
			}

			question := extractQuestion(req.Input.Input)

			prompt := fmt.Sprintf(`You are an impartial judge evaluating answer relevancy.

QUESTION:
%s

ANSWER:
%s

Task: Determine how well the ANSWER addresses the QUESTION.
- 1.0 = the answer directly and completely addresses the question
- 0.0 = the answer is completely unrelated to the question
- Values in between reflect partial relevancy

Respond with ONLY a JSON object: {"score": <float>, "reason": "<brief explanation>"}`, question, answer)

			resp, err := genkit.Generate(ctx, g,
				ai.WithModelName(modelName),
				ai.WithPrompt(prompt),
			)
			if err != nil {
				return nil, fmt.Errorf("relevancy judge failed: %w", err)
			}

			score, reason := parseJudgeResponse(resp.Text())
			return &ai.EvaluatorCallbackResponse{
				TestCaseId: req.Input.TestCaseId,
				Evaluation: []ai.Score{{
					Id:    "answerRelevancy",
					Score: score,
					Details: map[string]any{
						"reason": reason,
					},
				}},
			}, nil
		},
	)
}

// extractQuestion pulls the question text out of an eval record's input,
// which arrives as a map[string]any matching the evaluated flow's JSON
// input shape.
func extractQuestion(input any) string {
	if m, ok := input.(map[string]any); ok {
		if q, ok := m["question"].(string); ok {
			return q
		}
	}
	return fmt.Sprintf("%v", input)
}

// extractAnswer pulls the "answer" string from the flow output. A flow
// returning a struct (e.g. {Answer, Sources, ...}) arrives here as
// map[string]any.
func extractAnswer(output any) (string, error) {
	if output == nil {
		return "", errors.New("output is nil")
	}
	switch v := output.(type) {
	case string:
		return v, nil
	case map[string]any:
		if a, ok := v["answer"].(string); ok {
			return a, nil
		}
		// Fall back to JSON representation.
		b, _ := json.Marshal(v)
		return string(b), nil
	default:
		b, _ := json.Marshal(v)
		return string(b), nil
	}
}

// formatContext joins the context entries (retrieved documents) into a single string.
func formatContext(ctx []any) string {
	if len(ctx) == 0 {
		return ""
	}
	var parts []string
	for _, c := range ctx {
		switch v := c.(type) {
		case string:
			parts = append(parts, v)
		case map[string]any:
			if text, ok := v["text"].(string); ok {
				parts = append(parts, text)
			} else {
				b, _ := json.Marshal(v)
				parts = append(parts, string(b))
			}
		default:
			b, _ := json.Marshal(v)
			parts = append(parts, string(b))
		}
	}
	return strings.Join(parts, "\n---\n")
}

// parseJudgeResponse extracts score and reason from the LLM judge response.
// Expected format: {"score": 0.8, "reason": "..."}
func parseJudgeResponse(text string) (float64, string) {
	// Try to find JSON in the response (LLM might add extra text).
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start >= 0 && end > start {
		text = text[start : end+1]
	}

	var result struct {
		Score  any    `json:"score"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(text), &result); err == nil {
		score := toFloat64(result.Score)
		return clampScore(score), result.Reason
	}

	// Fallback: try to find a bare number.
	for _, word := range strings.Fields(text) {
		if f, err := strconv.ParseFloat(strings.TrimRight(word, ".,;"), 64); err == nil && f >= 0 && f <= 1 {
			return f, text
		}
	}

	return 0.0, "could not parse judge response: " + text
}

func toFloat64(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case json.Number:
		f, _ := n.Float64()
		return f
	case string:
		f, _ := strconv.ParseFloat(n, 64)
		return f
	default:
		return 0
	}
}

func clampScore(s float64) float64 {
	if s < 0 {
		return 0
	}
	if s > 1 {
		return 1
	}
	return s
}
