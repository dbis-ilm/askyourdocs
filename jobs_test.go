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
	"testing"
	"time"

	"github.com/firebase/genkit/go/core"
)

// waitForJob polls until the job with id finishes (done or failed) or the
// timeout elapses.
func waitForJob(t *testing.T, s *JobStore, id string) Job {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		j, ok := s.Get(id)
		if !ok {
			t.Fatalf("job %s disappeared", id)
		}
		if j.Status == JobDone || j.Status == JobFailed {
			return j
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("job %s did not finish within timeout", id)
	return Job{}
}

func TestJobStoreEnqueueRunsWorkAndStoresResult(t *testing.T) {
	s := NewJobStore()
	j, err := s.Enqueue("indexPDF", func(progress func(string)) (any, error) {
		return map[string]any{"documentsIndexed": 3}, nil
	})
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}

	done := waitForJob(t, s, j.ID)
	if done.Status != JobDone {
		t.Fatalf("Status = %q, want done (error=%q)", done.Status, done.Error)
	}
	m, ok := done.Result.(map[string]any)
	if !ok || m["documentsIndexed"] != 3 {
		t.Errorf("Result = %+v, want {documentsIndexed: 3}", done.Result)
	}
	if done.StartedAt == nil || done.EndedAt == nil {
		t.Error("StartedAt/EndedAt should be set for a finished job")
	}
	if done.Kind != "indexPDF" {
		t.Errorf("Kind = %q, want indexPDF", done.Kind)
	}
}

func TestJobStoreEnqueueCapturesError(t *testing.T) {
	s := NewJobStore()
	j, err := s.Enqueue("crawl", func(progress func(string)) (any, error) {
		return nil, errors.New("boom")
	})
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}

	done := waitForJob(t, s, j.ID)
	if done.Status != JobFailed {
		t.Fatalf("Status = %q, want failed", done.Status)
	}
	if done.Error != "boom" {
		t.Errorf("Error = %q, want %q", done.Error, "boom")
	}
	if done.Result != nil {
		t.Errorf("Result = %v, want nil on failure", done.Result)
	}
}

func TestJobStoreProgressIsVisibleBeforeCompletion(t *testing.T) {
	s := NewJobStore()
	release := make(chan struct{})
	progressSeen := make(chan struct{})

	j, err := s.Enqueue("crawl", func(progress func(string)) (any, error) {
		progress("Seite 1/5")
		close(progressSeen)
		<-release
		return "done", nil
	})
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}

	<-progressSeen
	deadline := time.Now().Add(time.Second)
	var got Job
	for time.Now().Before(deadline) {
		got, _ = s.Get(j.ID)
		if got.Progress == "Seite 1/5" {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if got.Progress != "Seite 1/5" {
		t.Fatalf("Progress = %q, want %q while the job is still running", got.Progress, "Seite 1/5")
	}
	if got.Status != JobRunning {
		t.Errorf("Status = %q, want running", got.Status)
	}

	close(release)
	waitForJob(t, s, j.ID)
}

func TestJobStoreGetUnknownID(t *testing.T) {
	s := NewJobStore()
	if _, ok := s.Get("does-not-exist"); ok {
		t.Error("Get returned ok=true for an unknown ID")
	}
}

func TestJobStoreListNewestFirstAndRespectsLimit(t *testing.T) {
	s := NewJobStore()
	var ids []string
	for i := 0; i < 5; i++ {
		j, err := s.Enqueue("test", func(progress func(string)) (any, error) { return nil, nil })
		if err != nil {
			t.Fatalf("Enqueue: %v", err)
		}
		waitForJob(t, s, j.ID) // force sequential completion so order is deterministic
		ids = append(ids, j.ID)
	}

	all := s.List(0)
	if len(all) != 5 {
		t.Fatalf("List(0) returned %d jobs, want 5", len(all))
	}
	for i, j := range all {
		want := ids[len(ids)-1-i]
		if j.ID != want {
			t.Errorf("List()[%d].ID = %q, want %q (newest first)", i, j.ID, want)
		}
	}

	limited := s.List(2)
	if len(limited) != 2 || limited[0].ID != ids[4] || limited[1].ID != ids[3] {
		t.Errorf("List(2) = %+v, want the 2 most recent jobs", limited)
	}
}

func TestJobStoreCapsHistory(t *testing.T) {
	s := NewJobStore()
	var ids []string
	for i := 0; i < maxJobHistory+10; i++ {
		j, err := s.Enqueue("test", func(progress func(string)) (any, error) { return nil, nil })
		if err != nil {
			t.Fatalf("Enqueue: %v", err)
		}
		ids = append(ids, j.ID)
	}

	// Eviction happens synchronously inside Enqueue, independent of whether
	// the worker has run the job yet, so no need to wait for completion here.
	all := s.List(0)
	if len(all) > maxJobHistory {
		t.Fatalf("List(0) returned %d jobs, want at most %d", len(all), maxJobHistory)
	}
	if _, ok := s.Get(ids[0]); ok {
		t.Error("oldest job should have been evicted from history")
	}
	lastID := ids[len(ids)-1]
	if _, ok := s.Get(lastID); !ok {
		t.Error("most recently enqueued job was evicted; eviction should drop the oldest, not the newest")
	}
}

func TestNewJobIDReturnsDistinctValues(t *testing.T) {
	a := newJobID()
	b := newJobID()
	if a == "" || b == "" {
		t.Fatal("newJobID returned an empty string")
	}
	if a == b {
		t.Errorf("two calls returned the same ID %q", a)
	}
}

func TestRunStreamingFlowAsJobCollectsProgressAndResult(t *testing.T) {
	flow := core.NewStreamingFlow("test-flow", func(ctx context.Context, input string, sendChunk core.StreamCallback[string]) (string, error) {
		sendChunk(ctx, "step 1")
		sendChunk(ctx, "step 2")
		return "final:" + input, nil
	})

	var progressed []string
	result, err := RunStreamingFlowAsJob(context.Background(), flow, "hello", func(msg string) {
		progressed = append(progressed, msg)
	})
	if err != nil {
		t.Fatalf("RunStreamingFlowAsJob: %v", err)
	}
	if result != "final:hello" {
		t.Errorf("result = %v, want %q", result, "final:hello")
	}
	want := []string{"step 1", "step 2"}
	if len(progressed) != len(want) {
		t.Fatalf("progressed = %v, want %v", progressed, want)
	}
	for i := range want {
		if progressed[i] != want[i] {
			t.Errorf("progressed[%d] = %q, want %q", i, progressed[i], want[i])
		}
	}
}

func TestRunStreamingFlowAsJobPropagatesError(t *testing.T) {
	flow := core.NewStreamingFlow("test-flow-err", func(ctx context.Context, input string, sendChunk core.StreamCallback[string]) (string, error) {
		return "", errors.New("boom")
	})

	_, err := RunStreamingFlowAsJob(context.Background(), flow, "x", func(string) {})
	if err == nil {
		t.Fatal("expected an error")
	}
}
