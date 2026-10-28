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
	"context"
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/firebase/genkit/go/core"
)

// maxJobInputBytes caps the request body FlowJobHandler and EvalJobHandler
// decode. A job's input is a few fields, never a document.
const maxJobInputBytes = 1 << 20

// GermanUnknownJob is the message JobHandler answers an unknown job ID with,
// for apps that keep showing it.
const GermanUnknownJob = "unbekannte Job-ID"

// FlowJobHandler returns a handler that starts flow as a background job: it
// decodes {"data": In} from the request body (an empty body leaves In's zero
// value), enqueues the flow with RunStreamingFlowAsJob so its stream becomes
// the job's progress, and answers 202 {"jobId": ...}. A malformed or larger
// than 1 MiB body gets 400, a full queue 503. kind is the job's kind label.
// ctx must be the app's long-lived context, not the request's: the job
// outlives the request.
func FlowJobHandler[In, Out any](jobs *JobStore, ctx context.Context, kind string, flow *core.Flow[In, Out, string]) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		in, ok := decodeJobInput[In](w, r)
		if !ok {
			return
		}
		enqueueJob(w, jobs, kind, func(progress func(string)) (any, error) {
			return RunStreamingFlowAsJob(ctx, flow, in, progress)
		})
	}
}

// JobHandler returns a handler for GET .../{id} that answers with the job's
// JSON, or 404 {"error": unknownJob} for an ID the store doesn't know (or
// no longer keeps). unknownJob is shown to the user, in the app's language;
// GermanUnknownJob is "unbekannte Job-ID".
func JobHandler(jobs *JobStore, unknownJob string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		j, ok := jobs.Get(r.PathValue("id"))
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": unknownJob})
			return
		}
		writeJSON(w, http.StatusOK, j)
	}
}

// JobListHandler returns a handler that answers {"jobs": [...]} with the
// newest jobs first, at most ?limit=N of them (default 50).
func JobListHandler(jobs *JobStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		limit := 50
		if n, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && n > 0 {
			limit = n
		}
		writeJSON(w, http.StatusOK, map[string]any{"jobs": jobs.List(limit)})
	}
}

// decodeJobInput reads {"data": In} from r's body. An empty body gives In's
// zero value. On a bad body it answers 400 itself and returns false.
func decodeJobInput[In any](w http.ResponseWriter, r *http.Request) (In, bool) {
	var body struct {
		Data In `json:"data"`
	}
	if r.Body != nil && r.ContentLength != 0 {
		defer r.Body.Close()
		r.Body = http.MaxBytesReader(w, r.Body, maxJobInputBytes)
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
			return body.Data, false
		}
	}
	return body.Data, true
}

// enqueueJob enqueues work and answers 202 {"jobId": ...}, or 503 when the
// queue is full.
func enqueueJob(w http.ResponseWriter, jobs *JobStore, kind string, work JobWork) {
	j, err := jobs.Enqueue(kind, work)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]string{"jobId": j.ID})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
