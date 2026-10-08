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
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// HTTP building blocks for triggering eval-dataset flows remotely — from the
// evalset command (cmd/evalset), for instance, against an app running in
// Docker, where `genkit flow:run` can't reach the flow and the container's
// files aren't on the host.
//
// The contract, shared by every app that adopts it:
//
//	POST /api/eval/{kind}              body {"data": {...}} -> 202 {"jobId": "..."}
//	GET  /api/jobs/{id}                -> the Job; its result carries "outPath"
//	GET  /api/eval/datasets/{name}     -> the dataset file the job wrote
//
// with kind one of "generated", "feedback", "injection". All three are admin
// routes: the first starts LLM work and reads server files, the last returns
// logged user questions.

// evalDatasetNameRE is the only file-name shape EvalDatasetHandler serves.
var evalDatasetNameRE = regexp.MustCompile(`^eval_dataset_[A-Za-z0-9_-]+\.json$`)

// EvalDatasetHandler serves eval dataset files from dir by name, taken from
// the request's {name} path value. Only names like eval_dataset_<x>.json are
// served, so the route can't be used to read anything else in dir (or
// outside it).
func EvalDatasetHandler(dir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		name := r.PathValue("name")
		if !evalDatasetNameRE.MatchString(name) {
			http.Error(w, `{"error":"unknown dataset"}`, http.StatusNotFound)
			return
		}
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			http.Error(w, `{"error":"unknown dataset"}`, http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(data)
	}
}

// RestrictEvalPaths prepares the paths of an HTTP-supplied eval request.
// outPath is cleared so the flow writes to its own default location —
// otherwise a client could make the server write a file anywhere it can. dir,
// if non-nil, must stay relative and inside the working directory: absolute
// paths and ".." are rejected. Either argument may be nil.
func RestrictEvalPaths(dir, outPath *string) error {
	if outPath != nil {
		*outPath = ""
	}
	if dir != nil && *dir != "" {
		clean := filepath.Clean(*dir)
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return errors.New("dir must be a relative path inside the working directory")
		}
		*dir = clean
	}
	return nil
}

// EvalJobHandler returns a handler for POST /api/eval/{kind}: it decodes
// {"data": In}, lets restrict adjust or reject the input (typically via
// RestrictEvalPaths; nil means no check), and runs the flow as a background
// job, answering 202 {"jobId": ...}. kind is the job's kind label. run
// executes the flow, e.g. func(ctx, in) (any, error) { return flow.Run(ctx,
// in) }. ctx must be the app's long-lived context, not the request's.
func EvalJobHandler[In any](jobs *JobStore, ctx context.Context, kind string, run func(ctx context.Context, in In) (any, error), restrict func(*In) error) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Data In `json:"data"`
		}
		if r.Body != nil && r.ContentLength != 0 {
			defer r.Body.Close()
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				writeEvalJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
				return
			}
		}
		if restrict != nil {
			if err := restrict(&body.Data); err != nil {
				writeEvalJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
				return
			}
		}

		j, err := jobs.Enqueue(kind, func(progress func(string)) (any, error) {
			progress("running")
			return run(ctx, body.Data)
		})
		if err != nil {
			writeEvalJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
			return
		}
		writeEvalJSON(w, http.StatusAccepted, map[string]string{"jobId": j.ID})
	}
}

func writeEvalJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
