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
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestEvalDatasetHandlerServesOnlyDatasetFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "eval_dataset_injection.json"), []byte(`[{"testCaseId":"a"}]`), 0o644)
	os.WriteFile(filepath.Join(dir, "secret.json"), []byte(`{"k":"v"}`), 0o644)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/eval/datasets/{name}", EvalDatasetHandler(dir))

	cases := []struct {
		name string
		path string
		want int
	}{
		{"dataset", "/api/eval/datasets/eval_dataset_injection.json", 200},
		{"other json file", "/api/eval/datasets/secret.json", 404},
		{"missing dataset", "/api/eval/datasets/eval_dataset_nope.json", 404},
		{"traversal", "/api/eval/datasets/..%2Fsecret.json", 404},
		{"traversal in name", "/api/eval/datasets/eval_dataset_..%2F..%2Fx.json", 404},
	}
	for _, c := range cases {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest("GET", c.path, nil))
		if rec.Code != c.want {
			t.Errorf("%s: status = %d, want %d", c.name, rec.Code, c.want)
		}
	}

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("GET", "/api/eval/datasets/eval_dataset_injection.json", nil))
	if got := rec.Body.String(); got != `[{"testCaseId":"a"}]` {
		t.Errorf("body = %q", got)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q", ct)
	}
}

func TestRestrictEvalPaths(t *testing.T) {
	ok := []string{"", "pdf-data", "./pdf-data/sub", "crawl-data"}
	for _, d := range ok {
		dir, out := d, "/etc/passwd"
		if err := RestrictEvalPaths(&dir, &out); err != nil {
			t.Errorf("dir %q rejected: %v", d, err)
		}
		if out != "" {
			t.Errorf("outPath %q was not cleared", out)
		}
	}
	bad := []string{"/etc", "../secret", "a/../../b", ".."}
	for _, d := range bad {
		dir := d
		if err := RestrictEvalPaths(&dir, nil); err == nil {
			t.Errorf("dir %q accepted, want error", d)
		}
	}
	if err := RestrictEvalPaths(nil, nil); err != nil {
		t.Errorf("nil, nil: %v", err)
	}
}

type evalTestInput struct {
	Dir     string `json:"dir"`
	OutPath string `json:"outPath"`
}

func waitDone(t *testing.T, jobs *JobStore, id string) Job {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		j, _ := jobs.Get(id)
		if j.Status == JobDone || j.Status == JobFailed {
			return j
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("job did not finish")
	return Job{}
}

func TestEvalJobHandlerRunsFlowAsJobWithRestrictedInput(t *testing.T) {
	jobs := NewJobStore()
	var got evalTestInput
	h := EvalJobHandler(jobs, context.Background(), "evalGenerated",
		func(ctx context.Context, in evalTestInput) (any, error) {
			got = in
			return map[string]string{"outPath": "testdata/eval_dataset_x.json"}, nil
		},
		func(in *evalTestInput) error { return RestrictEvalPaths(&in.Dir, &in.OutPath) })

	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest("POST", "/api/eval/generated", strings.NewReader(`{"data":{"dir":"pdf-data","outPath":"/tmp/evil.json"}}`)))
	if rec.Code != http.StatusAccepted {
		t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
	}
	var resp struct {
		JobID string `json:"jobId"`
	}
	json.Unmarshal(rec.Body.Bytes(), &resp)
	j := waitDone(t, jobs, resp.JobID)
	if j.Status != JobDone || j.Kind != "evalGenerated" {
		t.Errorf("job = %+v", j)
	}
	if got.Dir != "pdf-data" || got.OutPath != "" {
		t.Errorf("flow saw %+v, want dir kept and outPath cleared", got)
	}
}

func TestEvalJobHandlerRejectsBadInput(t *testing.T) {
	jobs := NewJobStore()
	called := false
	h := EvalJobHandler(jobs, context.Background(), "k",
		func(ctx context.Context, in evalTestInput) (any, error) { called = true; return nil, nil },
		func(in *evalTestInput) error { return RestrictEvalPaths(&in.Dir, &in.OutPath) })

	for name, body := range map[string]string{
		"absolute dir": `{"data":{"dir":"/etc"}}`,
		"bad json":     `{"data":`,
	} {
		rec := httptest.NewRecorder()
		h(rec, httptest.NewRequest("POST", "/x", strings.NewReader(body)))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400", name, rec.Code)
		}
	}
	if called {
		t.Error("flow ran despite rejected input")
	}
}

func TestEvalJobHandlerEmptyBodyUsesDefaultsAndReportsFlowError(t *testing.T) {
	jobs := NewJobStore()
	h := EvalJobHandler(jobs, context.Background(), "k",
		func(ctx context.Context, in evalTestInput) (any, error) { return nil, errors.New("boom") }, nil)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/x", nil)
	h(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("status = %d", rec.Code)
	}
	var resp struct {
		JobID string `json:"jobId"`
	}
	json.Unmarshal(rec.Body.Bytes(), &resp)
	if j := waitDone(t, jobs, resp.JobID); j.Status != JobFailed || j.Error != "boom" {
		t.Errorf("job = %+v, want failed with the flow's error", j)
	}
}
