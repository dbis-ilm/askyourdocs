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
	"testing"

	"github.com/firebase/genkit/go/genkit"
)

func TestDefineProgressFlowReturnsOutputAndCollectsProgress(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)

	flow := DefineProgressFlow(g, "testProgressFlow",
		func(ctx context.Context, input string, progress func(string)) (string, error) {
			progress("step 1")
			progress("step 2: " + input)
			return "done: " + input, nil
		})

	// .Run discards the stream — only the final output is observable this way.
	out, err := flow.Run(ctx, "hello")
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if out != "done: hello" {
		t.Errorf("Run output = %q, want %q", out, "done: hello")
	}

	// .Stream surfaces progress() calls as intermediate values.
	var seen []string
	for value, err := range flow.Stream(ctx, "world") {
		if err != nil {
			t.Fatalf("Stream returned error: %v", err)
		}
		if value.Stream != "" {
			seen = append(seen, value.Stream)
		}
		if value.Done {
			if value.Output != "done: world" {
				t.Errorf("final output = %q, want %q", value.Output, "done: world")
			}
		}
	}
	want := []string{"step 1", "step 2: world"}
	if len(seen) != len(want) {
		t.Fatalf("progress messages = %v, want %v", seen, want)
	}
	for i := range want {
		if seen[i] != want[i] {
			t.Errorf("progress[%d] = %q, want %q", i, seen[i], want[i])
		}
	}
}

func TestDefineProgressFlowPropagatesError(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)

	wantErr := context.DeadlineExceeded
	flow := DefineProgressFlow(g, "testProgressFlowError",
		func(ctx context.Context, input string, progress func(string)) (string, error) {
			return "", wantErr
		})

	_, err := flow.Run(ctx, "x")
	if err == nil {
		t.Fatal("want error, got nil")
	}
}
