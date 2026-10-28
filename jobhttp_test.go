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
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core"
)

func TestFlowJobHandlerRunsFlowAsJob(t *testing.T) {
	flow := core.NewStreamingFlow("test-flow-job", func(ctx context.Context, in struct {
		Name string `json:"name"`
	}, send core.StreamCallback[string]) (string, error) {
		send(ctx, "working")
		return "hello " + in.Name, nil
	})
	jobs := NewJobStore()
	w := httptest.NewRecorder()
	FlowJobHandler(jobs, context.Background(), "greet", flow)(w,
		httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"data":{"name":"world"}}`)))

	if w.Code != http.StatusAccepted || w.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("status = %d, content type %q, body %s", w.Code, w.Header().Get("Content-Type"), w.Body)
	}
	var resp struct{ JobID string }
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil || resp.JobID == "" {
		t.Fatalf("response %v, err %v; want a jobId", resp, err)
	}
	j := waitForJob(t, jobs, resp.JobID)
	if j.Status != JobDone || j.Result != "hello world" || j.Kind != "greet" || j.Progress != "working" {
		t.Errorf("job = %+v, want done, result hello world, kind greet, progress working", j)
	}
}

func TestFlowJobHandlerEmptyBodyUsesZeroInput(t *testing.T) {
	flow := core.NewStreamingFlow("test-flow-job-empty", func(ctx context.Context, in string, send core.StreamCallback[string]) (string, error) {
		return "got " + in + ".", nil
	})
	jobs := NewJobStore()
	w := httptest.NewRecorder()
	FlowJobHandler(jobs, context.Background(), "k", flow)(w, httptest.NewRequest(http.MethodPost, "/", nil))
	var resp struct{ JobID string }
	json.NewDecoder(w.Body).Decode(&resp)
	if j := waitForJob(t, jobs, resp.JobID); j.Result != "got ." {
		t.Errorf("result = %v, want the zero input", j.Result)
	}
}

func TestFlowJobHandlerRejectsBadBodies(t *testing.T) {
	flow := core.NewStreamingFlow("test-flow-job-bad", func(ctx context.Context, in string, send core.StreamCallback[string]) (string, error) {
		t.Error("flow ran for a rejected request")
		return "", nil
	})
	h := FlowJobHandler(NewJobStore(), context.Background(), "k", flow)
	huge := `{"data":"` + strings.Repeat("x", maxJobInputBytes) + `"}`
	for name, body := range map[string]string{"not JSON": "not json", "over 1 MiB": huge} {
		w := httptest.NewRecorder()
		h(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
		if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "invalid JSON body") {
			t.Errorf("%s: status %d, body %s; want 400 invalid JSON body", name, w.Code, w.Body)
		}
	}
}

func TestJobHandler(t *testing.T) {
	jobs := NewJobStore()
	j, err := jobs.Enqueue("k", func(progress func(string)) (any, error) { return "ok", nil })
	if err != nil {
		t.Fatal(err)
	}
	waitForJob(t, jobs, j.ID)
	h := JobHandler(jobs, GermanUnknownJob)

	req := httptest.NewRequest(http.MethodGet, "/api/jobs/"+j.ID, nil)
	req.SetPathValue("id", j.ID)
	w := httptest.NewRecorder()
	h(w, req)
	var got Job
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil || w.Code != http.StatusOK || got.Result != "ok" {
		t.Errorf("status %d, job %+v, err %v; want 200 with result ok", w.Code, got, err)
	}

	req = httptest.NewRequest(http.MethodGet, "/api/jobs/nope", nil)
	req.SetPathValue("id", "nope")
	w = httptest.NewRecorder()
	h(w, req)
	if w.Code != http.StatusNotFound || !strings.Contains(w.Body.String(), `"unbekannte Job-ID"`) {
		t.Errorf("unknown ID: status %d, body %s; want 404 with the given message", w.Code, w.Body)
	}
}

func TestJobListHandlerRespectsLimit(t *testing.T) {
	jobs := NewJobStore()
	for range 3 {
		j, err := jobs.Enqueue("k", func(progress func(string)) (any, error) { return nil, nil })
		if err != nil {
			t.Fatal(err)
		}
		waitForJob(t, jobs, j.ID)
	}
	for url, want := range map[string]int{"/api/jobs?limit=2": 2, "/api/jobs": 3, "/api/jobs?limit=x": 3} {
		w := httptest.NewRecorder()
		JobListHandler(jobs)(w, httptest.NewRequest(http.MethodGet, url, nil))
		var resp struct{ Jobs []Job }
		if err := json.NewDecoder(bytes.NewReader(w.Body.Bytes())).Decode(&resp); err != nil || len(resp.Jobs) != want {
			t.Errorf("%s: %d jobs, err %v; want %d", url, len(resp.Jobs), err, want)
		}
	}
}
