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

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	engine "github.com/dbis-ilm/askyourdocs"
)

// newTestApp stands up the server side of the contract with the engine's own
// handlers, behind Basic Auth on every route, and writes the dataset the fake
// flow "produces" into a temp dir.
func newTestApp(t *testing.T, flow func(ctx context.Context, in map[string]any) (any, error)) *httptest.Server {
	t.Helper()
	dir := t.TempDir()
	jobs := engine.NewJobStore()

	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/eval/{kind}", func(w http.ResponseWriter, r *http.Request) {
		engine.EvalJobHandler(jobs, context.Background(), r.PathValue("kind"),
			func(ctx context.Context, in map[string]any) (any, error) {
				res, err := flow(ctx, in)
				if err == nil {
					os.WriteFile(filepath.Join(dir, "eval_dataset_"+r.PathValue("kind")+".json"), []byte(`[{"testCaseId":"t1"}]`), 0o644)
				}
				return res, err
			}, nil)(w, r)
	})
	mux.HandleFunc("GET /api/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {
		j, ok := jobs.Get(r.PathValue("id"))
		if !ok {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(j)
	})
	mux.HandleFunc("GET /api/eval/datasets/{name}", engine.EvalDatasetHandler(dir))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, pw, ok := r.BasicAuth(); !ok || pw != "s3cret" {
			w.Header().Set("WWW-Authenticate", `Basic realm="x"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		mux.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func env(m map[string]string) func(string) string { return func(k string) string { return m[k] } }

func TestRunFetchesDatasetEndToEnd(t *testing.T) {
	var seen map[string]any
	srv := newTestApp(t, func(ctx context.Context, in map[string]any) (any, error) {
		seen = in
		return map[string]any{"written": 1, "outPath": "testdata/eval_dataset_injection.json"}, nil
	})
	out := filepath.Join(t.TempDir(), "sub", "ds.json")

	var stdout, stderr bytes.Buffer
	err := run([]string{"-kind", "injection", "-url", srv.URL, "-out", out, "-poll", "1ms", "-user", "admin", "-data", `{"sampleRate":8}`},
		&stdout, &stderr, env(map[string]string{"EVALSET_PASSWORD": "s3cret"}))
	if err != nil {
		t.Fatalf("run: %v\nstderr: %s", err, stderr.String())
	}

	got, err := os.ReadFile(out)
	if err != nil || string(got) != `[{"testCaseId":"t1"}]` {
		t.Errorf("saved file = %q, err %v", got, err)
	}
	if strings.TrimSpace(stdout.String()) != out {
		t.Errorf("stdout = %q, want the saved path", stdout.String())
	}
	if seen["sampleRate"] != float64(8) {
		t.Errorf("flow input = %v, want -data forwarded", seen)
	}
}

func TestRunDefaultsOutputToTestdataWithServerName(t *testing.T) {
	srv := newTestApp(t, func(ctx context.Context, in map[string]any) (any, error) {
		return map[string]any{"outPath": "testdata/eval_dataset_feedback.json"}, nil
	})
	t.Chdir(t.TempDir())

	var stdout, stderr bytes.Buffer
	err := run([]string{"-kind", "feedback", "-url", srv.URL, "-poll", "1ms", "-user", "admin"},
		&stdout, &stderr, env(map[string]string{"EVALSET_PASSWORD": "s3cret"}))
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if _, err := os.Stat(filepath.Join("testdata", "eval_dataset_feedback.json")); err != nil {
		t.Errorf("default output missing: %v", err)
	}
}

func TestRunReportsFailedJob(t *testing.T) {
	srv := newTestApp(t, func(ctx context.Context, in map[string]any) (any, error) {
		return nil, context.DeadlineExceeded
	})
	err := run([]string{"-kind", "generated", "-url", srv.URL, "-poll", "1ms", "-user", "admin"},
		&bytes.Buffer{}, &bytes.Buffer{}, env(map[string]string{"EVALSET_PASSWORD": "s3cret"}))
	if err == nil || !strings.Contains(err.Error(), "job failed") {
		t.Errorf("err = %v, want a job-failed error", err)
	}
}

func TestRunHintsAtCredentialsOn401(t *testing.T) {
	srv := newTestApp(t, func(ctx context.Context, in map[string]any) (any, error) { return nil, nil })
	err := run([]string{"-kind", "injection", "-url", srv.URL}, &bytes.Buffer{}, &bytes.Buffer{}, env(nil))
	if err == nil || !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "EVALSET_PASSWORD") {
		t.Errorf("err = %v, want a 401 with a credentials hint", err)
	}
}

func TestRunTimesOutWhileJobStaysRunning(t *testing.T) {
	block := make(chan struct{})
	t.Cleanup(func() { close(block) })
	srv := newTestApp(t, func(ctx context.Context, in map[string]any) (any, error) {
		<-block
		return nil, nil
	})
	err := run([]string{"-kind", "generated", "-url", srv.URL, "-poll", "5ms", "-timeout", "60ms", "-user", "admin"},
		&bytes.Buffer{}, &bytes.Buffer{}, env(map[string]string{"EVALSET_PASSWORD": "s3cret"}))
	if err == nil || !strings.Contains(err.Error(), "gave up") {
		t.Errorf("err = %v, want a give-up error", err)
	}
}

func TestRunLoginPostsCredentialsAndKeepsCookie(t *testing.T) {
	var loginBody map[string]string
	var cookieSeen bool
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/login", func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&loginBody)
		http.SetCookie(w, &http.Cookie{Name: "session", Value: "tok", Path: "/"})
	})
	mux.HandleFunc("POST /api/eval/{kind}", func(w http.ResponseWriter, r *http.Request) {
		if c, err := r.Cookie("session"); err == nil && c.Value == "tok" {
			cookieSeen = true
		}
		w.WriteHeader(http.StatusAccepted)
		w.Write([]byte(`{"jobId":"j1"}`))
	})
	mux.HandleFunc("GET /api/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"status":"done","result":{"outPath":"testdata/eval_dataset_injection.json"}}`))
	})
	mux.HandleFunc("GET /api/eval/datasets/{name}", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`[]`)) })
	srv := httptest.NewServer(mux)
	defer srv.Close()

	out := filepath.Join(t.TempDir(), "x.json")
	err := run([]string{"-kind", "injection", "-url", srv.URL, "-out", out, "-poll", "1ms", "-user", "kai", "-login"},
		&bytes.Buffer{}, &bytes.Buffer{}, env(map[string]string{"EVALSET_PASSWORD": "pw"}))
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if loginBody["username"] != "kai" || loginBody["password"] != "pw" {
		t.Errorf("login body = %v", loginBody)
	}
	if !cookieSeen {
		t.Error("session cookie was not sent on later requests")
	}
}

func TestRunValidatesFlags(t *testing.T) {
	cases := map[string][]string{
		"no kind":      {},
		"unknown kind": {"-kind", "nonsense"},
		"bad data":     {"-kind", "injection", "-data", "{oops"},
		"login no pw":  {"-kind", "injection", "-login", "-user", "kai"},
	}
	for name, args := range cases {
		if err := run(args, &bytes.Buffer{}, &bytes.Buffer{}, env(nil)); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}
