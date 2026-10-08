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

// Command evalset fetches an eval dataset from a running app over HTTP and
// saves it locally — for apps in Docker, where `genkit flow:run` can't reach
// the flow and the dataset would otherwise stay inside the container.
//
//	evalset -kind injection -url http://localhost:3400 -out testdata/eval_dataset_injection.json
//
// It speaks the contract documented in the engine's evalhttp.go: start the
// flow with POST /api/eval/{kind}, poll GET /api/jobs/{id}, then download the
// file the job wrote with GET /api/eval/datasets/{name}. Kinds: generated
// (questions generated from the indexed documents), feedback (from logged
// real usage), injection (fixed prompt-injection attacks).
//
// Credentials come from -user and the EVALSET_PASSWORD environment variable
// (kept off the command line, where other users' ps would show it). They are
// sent as HTTP Basic Auth; with -login they are also posted to /api/login
// first, for apps that use a session cookie instead.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr, os.Getenv); err != nil {
		fmt.Fprintln(os.Stderr, "evalset:", err)
		os.Exit(1)
	}
}

type options struct {
	baseURL  string
	kind     string
	data     string
	out      string
	user     string
	password string
	login    bool
	poll     time.Duration
	timeout  time.Duration
}

func run(args []string, stdout, stderr io.Writer, getenv func(string) string) error {
	fs := flag.NewFlagSet("evalset", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var o options
	fs.StringVar(&o.baseURL, "url", firstNonEmpty(getenv("EVALSET_URL"), "http://localhost:3400"), "base URL of the app (env EVALSET_URL)")
	fs.StringVar(&o.kind, "kind", "", "dataset kind: generated, feedback or injection (required)")
	fs.StringVar(&o.data, "data", "", `flow input as JSON, e.g. '{"sampleRate":8,"maxPerDoc":5}' (generated: also "dir")`)
	fs.StringVar(&o.out, "out", "", "where to save the dataset (default testdata/<name the app gave it>)")
	fs.StringVar(&o.user, "user", getenv("EVALSET_USER"), "user name for Basic Auth / login (env EVALSET_USER); the password is read from EVALSET_PASSWORD")
	fs.BoolVar(&o.login, "login", false, "also log in via POST /api/login and keep the session cookie")
	fs.DurationVar(&o.poll, "poll", 2*time.Second, "how often to poll the job")
	fs.DurationVar(&o.timeout, "timeout", 30*time.Minute, "give up after this long")
	if err := fs.Parse(args); err != nil {
		return err
	}
	o.password = getenv("EVALSET_PASSWORD")

	switch o.kind {
	case "generated", "feedback", "injection":
	case "":
		return errors.New("-kind is required (generated, feedback or injection)")
	default:
		return fmt.Errorf("unknown -kind %q (generated, feedback or injection)", o.kind)
	}
	if o.data != "" && !json.Valid([]byte(o.data)) {
		return errors.New("-data is not valid JSON")
	}
	o.baseURL = strings.TrimRight(o.baseURL, "/")

	ctx, cancel := context.WithTimeout(context.Background(), o.timeout)
	defer cancel()
	err := fetch(ctx, o, stdout, stderr)
	if err != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return fmt.Errorf("gave up after %s: %w", o.timeout, err)
	}
	return err
}

func fetch(ctx context.Context, o options, stdout, stderr io.Writer) error {
	jar, _ := cookiejar.New(nil)
	c := &client{http: &http.Client{Jar: jar}, o: o}

	if o.login {
		if o.user == "" || o.password == "" {
			return errors.New("-login needs -user and EVALSET_PASSWORD")
		}
		body, _ := json.Marshal(map[string]string{"username": o.user, "password": o.password})
		if _, err := c.do(ctx, "POST", "/api/login", body); err != nil {
			return fmt.Errorf("login: %w", err)
		}
	}

	payload := []byte(`{"data":{}}`)
	if o.data != "" {
		payload = []byte(`{"data":` + o.data + `}`)
	}
	resp, err := c.do(ctx, "POST", "/api/eval/"+o.kind, payload)
	if err != nil {
		return fmt.Errorf("start %s job: %w", o.kind, err)
	}
	var started struct {
		JobID string `json:"jobId"`
	}
	if err := json.Unmarshal(resp, &started); err != nil || started.JobID == "" {
		return fmt.Errorf("start %s job: unexpected response %q", o.kind, truncate(string(resp)))
	}
	fmt.Fprintf(stderr, "job %s started\n", started.JobID)

	var result map[string]any
	last := ""
	for {
		raw, err := c.do(ctx, "GET", "/api/jobs/"+started.JobID, nil)
		if err != nil {
			return fmt.Errorf("poll job: %w", err)
		}
		var j struct {
			Status   string         `json:"status"`
			Progress string         `json:"progress"`
			Error    string         `json:"error"`
			Result   map[string]any `json:"result"`
		}
		if err := json.Unmarshal(raw, &j); err != nil {
			return fmt.Errorf("poll job: unexpected response %q", truncate(string(raw)))
		}
		if j.Progress != "" && j.Progress != last {
			fmt.Fprintf(stderr, "  %s\n", j.Progress)
			last = j.Progress
		}
		if j.Status == "failed" {
			return fmt.Errorf("job failed: %s", j.Error)
		}
		if j.Status == "done" {
			result = j.Result
			break
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("job %s still running: %w", started.JobID, ctx.Err())
		case <-time.After(o.poll):
		}
	}

	outPath, _ := result["outPath"].(string)
	name := path.Base(filepath.ToSlash(outPath))
	if outPath == "" || name == "." || name == "/" {
		return fmt.Errorf("job finished without an outPath: %v", result)
	}
	if w, ok := result["written"]; ok {
		fmt.Fprintf(stderr, "job wrote %v cases\n", w)
	}

	data, err := c.do(ctx, "GET", "/api/eval/datasets/"+name, nil)
	if err != nil {
		return fmt.Errorf("download %s: %w", name, err)
	}
	dest := o.out
	if dest == "" {
		dest = filepath.Join("testdata", name)
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(dest, data, 0o644); err != nil {
		return err
	}
	fmt.Fprintf(stderr, "saved %d bytes\n", len(data))
	fmt.Fprintln(stdout, dest)
	return nil
}

type client struct {
	http *http.Client
	o    options
}

// do sends one request and returns the body of a 2xx response; any other
// status becomes an error carrying the server's message.
func (c *client) do(ctx context.Context, method, p string, body []byte) ([]byte, error) {
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.o.baseURL+p, rd)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.o.user != "" && c.o.password != "" {
		req.SetBasicAuth(c.o.user, c.o.password)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		hint := ""
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusServiceUnavailable {
			hint = " (admin credentials needed? set -user and EVALSET_PASSWORD)"
		}
		return nil, fmt.Errorf("%s %s: %s: %s%s", method, p, resp.Status, truncate(string(data)), hint)
	}
	return data, nil
}

func truncate(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 200 {
		return s[:200] + "…"
	}
	return s
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
