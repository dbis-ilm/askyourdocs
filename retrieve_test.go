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
	"errors"
	"fmt"
	"iter"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// fakePrompt is a test double for ai.Prompt: it ignores every option and
// returns a canned reply, or a canned error on a chosen call number.
type fakePrompt struct {
	mu       sync.Mutex
	calls    int
	reply    string
	replyFn  func(calls int) string
	failCall int // 1-based; 0 means never fail
}

func (p *fakePrompt) Name() string { return "fake" }

func (p *fakePrompt) Execute(ctx context.Context, opts ...ai.PromptExecuteOption) (*ai.ModelResponse, error) {
	p.mu.Lock()
	p.calls++
	n := p.calls
	p.mu.Unlock()

	if p.failCall != 0 && n == p.failCall {
		return nil, errors.New("fake failure")
	}
	text := p.reply
	if p.replyFn != nil {
		text = p.replyFn(n)
	}
	return &ai.ModelResponse{
		Message: &ai.Message{
			Content: []*ai.Part{ai.NewTextPart(text)},
		},
	}, nil
}

func (p *fakePrompt) ExecuteStream(ctx context.Context, opts ...ai.PromptExecuteOption) iter.Seq2[*ai.ModelStreamValue, error] {
	return func(yield func(*ai.ModelStreamValue, error) bool) {}
}

func (p *fakePrompt) Render(ctx context.Context, input any) (*ai.GenerateActionOptions, error) {
	return nil, errors.New("fakePrompt.Render not implemented")
}

func (p *fakePrompt) callCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

func doc(text string) *ai.Document {
	return ai.DocumentFromText(text, nil)
}

func TestGenerateHyDETextReturnsResponse(t *testing.T) {
	p := &fakePrompt{reply: "hypothetical passage"}
	got := GenerateHyDEText(context.Background(), p, nil, nil, "what is x?")
	if got != "hypothetical passage" {
		t.Errorf("got %q", got)
	}
}

func TestGenerateHyDETextFallsBackToQuestionOnError(t *testing.T) {
	p := &fakePrompt{failCall: 1}
	got := GenerateHyDEText(context.Background(), p, nil, nil, "what is x?")
	if got != "what is x?" {
		t.Errorf("got %q, want fallback to question", got)
	}
}

func TestGenerateHyDETextFallsBackToQuestionOnEmptyReply(t *testing.T) {
	p := &fakePrompt{reply: ""}
	got := GenerateHyDEText(context.Background(), p, nil, nil, "what is x?")
	if got != "what is x?" {
		t.Errorf("got %q, want fallback to question", got)
	}
}

func TestReciprocalRankFusionOrdersByScore(t *testing.T) {
	a, b, c := doc("a"), doc("b"), doc("c")
	lists := []RankedList{
		{Docs: []*ai.Document{a, b, c}, Weight: 1},
	}
	got := ReciprocalRankFusion(lists, 60, 0)
	if len(got) != 3 || got[0].Text != "a" || got[1].Text != "b" || got[2].Text != "c" {
		t.Fatalf("got %+v", got)
	}
}

func TestReciprocalRankFusionMergesDuplicateText(t *testing.T) {
	a1, a2, b := doc("a"), doc("a"), doc("b")
	lists := []RankedList{
		{Docs: []*ai.Document{b, a1}, Weight: 1}, // a1 rank 1
		{Docs: []*ai.Document{a2}, Weight: 1},    // a2 rank 0, same text as a1
	}
	got := ReciprocalRankFusion(lists, 60, 0)
	if len(got) != 2 {
		t.Fatalf("want 2 distinct docs after merge, got %d: %+v", len(got), got)
	}
	// "a" appears at rank 1 (weight 1) and rank 0 (weight 1); its score should
	// exceed "b"'s (rank 0, weight 1, counted once).
	var aScore, bScore float64
	for _, d := range got {
		if d.Text == "a" {
			aScore = d.Score
		} else {
			bScore = d.Score
		}
	}
	if aScore <= bScore {
		t.Errorf("merged doc should outscore single-source doc: a=%v b=%v", aScore, bScore)
	}
}

func TestReciprocalRankFusionRespectsMaxResults(t *testing.T) {
	lists := []RankedList{
		{Docs: []*ai.Document{doc("a"), doc("b"), doc("c")}, Weight: 1},
	}
	got := ReciprocalRankFusion(lists, 60, 2)
	if len(got) != 2 {
		t.Fatalf("want 2 results, got %d", len(got))
	}
}

func TestReciprocalRankFusionWeightsSourcesIndependently(t *testing.T) {
	lists := []RankedList{
		{Docs: []*ai.Document{doc("low")}, Weight: 0.1},
		{Docs: []*ai.Document{doc("high")}, Weight: 1.0},
	}
	got := ReciprocalRankFusion(lists, 60, 0)
	if got[0].Text != "high" {
		t.Fatalf("want higher-weight source to rank first, got %+v", got)
	}
}

func TestParseRerankScoresBasic(t *testing.T) {
	scores, n := parseRerankScores("1:8\n2:3\n3:10", 3, 5)
	if n != 3 {
		t.Fatalf("want 3 scored, got %d", n)
	}
	want := []int{8, 3, 10}
	for i, s := range want {
		if scores[i] != s {
			t.Errorf("scores[%d] = %d, want %d", i, scores[i], s)
		}
	}
}

func TestParseRerankScoresClampsAboveTen(t *testing.T) {
	scores, _ := parseRerankScores("1:99", 1, 5)
	if scores[0] != 10 {
		t.Errorf("want clamped to 10, got %d", scores[0])
	}
}

func TestParseRerankScoresKeepsDefaultForMissingLines(t *testing.T) {
	scores, n := parseRerankScores("1:8", 3, 5)
	if n != 1 {
		t.Fatalf("want 1 scored, got %d", n)
	}
	if scores[1] != 5 || scores[2] != 5 {
		t.Errorf("unscored positions should keep default: %v", scores)
	}
}

func TestParseRerankScoresIgnoresOutOfRangeLines(t *testing.T) {
	scores, n := parseRerankScores("1:8\n10:3\n0:1", 2, 5)
	if n != 1 {
		t.Fatalf("want 1 scored (out-of-range lines ignored), got %d", n)
	}
	if scores[0] != 8 {
		t.Errorf("scores[0] = %d, want 8", scores[0])
	}
}

func TestParseRerankScoresIgnoresGarbageLines(t *testing.T) {
	got, scored := parseRerankScores("1:9\n2:0\nMüll ohne Doppelpunkt\n4:12\n99:7\n", 5, 5)
	if scored != 3 {
		t.Errorf("scored = %d, want 3 (positions 1, 2 and 4)", scored)
	}
	want := []int{9, 0, 5, 10, 5} // 3 missing -> default, 12 clamped to 10, 99 out of range
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("parseRerankScores = %v, want %v", got, want)
			break
		}
	}
}

func TestRerankScoresBatching(t *testing.T) {
	texts := make([]string, 12)
	for i := range texts {
		texts[i] = fmt.Sprintf("chunk-%d", i)
	}
	var calls int32
	p := &fakePrompt{replyFn: func(n int) string {
		atomic.AddInt32(&calls, 1)
		return "1:7\n2:7\n3:7\n4:7\n5:7"
	}}
	scores := RerankScores(context.Background(), p, nil, nil, "q", texts, RerankOptions{BatchSize: 5, Parallel: 4})
	if len(scores) != 12 {
		t.Fatalf("want 12 scores, got %d", len(scores))
	}
	for i, s := range scores {
		if s != 7 {
			t.Errorf("scores[%d] = %d, want 7", i, s)
		}
	}
	// 12 distinct texts in batches of 5 => 3 calls.
	if got := p.callCount(); got != 3 {
		t.Errorf("want 3 batch calls, got %d", got)
	}
}

func TestRerankScoresSurvivesAFailedBatch(t *testing.T) {
	texts := []string{"a", "b", "c", "d", "e", "f"}
	// 6 texts, batch size 5 => 2 batches (5 + 1); whichever one wins the race
	// to fail keeps DefaultScore, the other gets the real score — the
	// ordering between concurrent batches isn't guaranteed, so this asserts
	// the order-independent invariant: failure stays isolated to its own
	// batch instead of taking down every score.
	p := &fakePrompt{failCall: 1, replyFn: func(n int) string { return "1:9\n2:9\n3:9\n4:9\n5:9" }}
	scores := RerankScores(context.Background(), p, nil, nil, "q", texts, RerankOptions{BatchSize: 5, Parallel: 1, DefaultScore: 4})
	if len(scores) != 6 {
		t.Fatalf("want 6 scores, got %d", len(scores))
	}
	var defaults, nines int
	for _, s := range scores {
		switch s {
		case 4:
			defaults++
		case 9:
			nines++
		default:
			t.Errorf("unexpected score %d in %v", s, scores)
		}
	}
	if defaults == 0 || defaults == 6 {
		t.Errorf("want exactly one batch defaulted, not all-or-nothing: %v", scores)
	}
	if defaults+nines != 6 {
		t.Errorf("want every score to be either default or scored, got %v", scores)
	}
}

func TestRerankScoresDeduplicates(t *testing.T) {
	texts := []string{"x", "x", "x", "y", "y", "z", "x", "y", "z", "x", "y", "z"}
	p := &fakePrompt{reply: "1:9\n2:6\n3:3"}
	scores := RerankScores(context.Background(), p, nil, nil, "q", texts, RerankOptions{})
	if got := p.callCount(); got != 1 {
		t.Fatalf("want 1 call for 3 distinct texts, got %d", got)
	}
	if len(scores) != len(texts) {
		t.Fatalf("want %d scores, got %d", len(texts), len(scores))
	}
}

func TestRerankScoresEmptyInput(t *testing.T) {
	p := &fakePrompt{}
	scores := RerankScores(context.Background(), p, nil, nil, "q", nil, RerankOptions{})
	if scores != nil {
		t.Errorf("want nil for empty input, got %v", scores)
	}
	if p.callCount() != 0 {
		t.Errorf("want no calls for empty input")
	}
}
