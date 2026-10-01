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

import "testing"

func TestParseJudgeResponse(t *testing.T) {
	cases := []struct {
		name       string
		text       string
		wantScore  float64
		wantReason string
	}{
		{"clean json", `{"score": 0.8, "reason": "mostly grounded"}`, 0.8, "mostly grounded"},
		{"json with surrounding prose", "Sure, here you go: {\"score\": 1, \"reason\": \"perfect\"} thanks!", 1.0, "perfect"},
		{"integer score", `{"score": 1, "reason": "ok"}`, 1.0, "ok"},
		{"score above range clamps to 1", `{"score": 1.5, "reason": "over"}`, 1.0, "over"},
		{"score below range clamps to 0", `{"score": -0.3, "reason": "under"}`, 0.0, "under"},
		{"bare number fallback", "The score is 0.7 based on my analysis.", 0.7, "The score is 0.7 based on my analysis."},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			score, reason := parseJudgeResponse(c.text)
			if score != c.wantScore {
				t.Errorf("score = %v, want %v", score, c.wantScore)
			}
			if reason != c.wantReason {
				t.Errorf("reason = %q, want %q", reason, c.wantReason)
			}
		})
	}
}

func TestParseJudgeResponseUnparsable(t *testing.T) {
	score, reason := parseJudgeResponse("no json and no number here")
	if score != 0.0 {
		t.Errorf("score = %v, want 0.0 for unparsable input", score)
	}
	if reason == "" {
		t.Error("reason should explain the parse failure, got empty string")
	}
}

func TestToFloat64(t *testing.T) {
	cases := []struct {
		in   any
		want float64
	}{
		{float64(0.5), 0.5},
		{float32(0.25), 0.25},
		{int(3), 3.0},
		{"0.9", 0.9},
		{"not a number", 0},
		{nil, 0},
	}
	for _, c := range cases {
		if got := toFloat64(c.in); got != c.want {
			t.Errorf("toFloat64(%#v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestClampScore(t *testing.T) {
	cases := map[float64]float64{
		-1:  0,
		0:   0,
		0.5: 0.5,
		1:   1,
		2:   1,
	}
	for in, want := range cases {
		if got := clampScore(in); got != want {
			t.Errorf("clampScore(%v) = %v, want %v", in, got, want)
		}
	}
}

func TestExtractAnswer(t *testing.T) {
	cases := []struct {
		name    string
		input   any
		want    string
		wantErr bool
	}{
		{"nil input errors", nil, "", true},
		{"plain string", "the answer", "the answer", false},
		{"map with answer field", map[string]any{"answer": "found it", "context": []any{"x"}}, "found it", false},
		{"map without answer field falls back to json", map[string]any{"foo": "bar"}, `{"foo":"bar"}`, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := extractAnswer(c.input)
			if (err != nil) != c.wantErr {
				t.Fatalf("err = %v, wantErr = %v", err, c.wantErr)
			}
			if got != c.want {
				t.Errorf("answer = %q, want %q", got, c.want)
			}
		})
	}
}

func TestExtractQuestion(t *testing.T) {
	cases := []struct {
		name  string
		input any
		want  string
	}{
		{"current qaInput shape", map[string]any{"question": "Wie viele TOPs?", "tags": []any{"Senat"}}, "Wie viele TOPs?"},
		{"legacy plain string dataset", "Wie viele TOPs?", "Wie viele TOPs?"},
		{"map without question key falls back to %v", map[string]any{"foo": "bar"}, "map[foo:bar]"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := extractQuestion(c.input); got != c.want {
				t.Errorf("extractQuestion(%#v) = %q, want %q", c.input, got, c.want)
			}
		})
	}
}

func TestFormatContext(t *testing.T) {
	if got := formatContext(nil); got != "" {
		t.Errorf("formatContext(nil) = %q, want empty", got)
	}

	ctx := []any{
		"plain string chunk",
		map[string]any{"text": "chunk with text field"},
		map[string]any{"other": "no text field"},
	}
	got := formatContext(ctx)
	want := "plain string chunk\n---\nchunk with text field\n---\n{\"other\":\"no text field\"}"
	if got != want {
		t.Errorf("formatContext = %q, want %q", got, want)
	}
}
