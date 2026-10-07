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
	"sync"
	"time"

	"github.com/firebase/genkit/go/core"
)

// ErrJobQueueFull is returned by Enqueue when the queue's 64-job buffer is
// still full after enqueueTimeout — the single worker is far enough behind
// that submitting more work would otherwise mean blocking the calling HTTP
// handler indefinitely.
var ErrJobQueueFull = errors.New("job queue is full, try again later")

// enqueueTimeout bounds how long Enqueue will wait for queue space before
// giving up. Generous relative to how fast the worker actually drains jobs
// (each Enqueue call itself is sub-millisecond), so it only ever triggers
// under genuine sustained overload, not an ordinary momentary burst.
const enqueueTimeout = 5 * time.Second

type JobStatus string

const (
	JobPending JobStatus = "pending"
	JobRunning JobStatus = "running"
	JobDone    JobStatus = "done"
	JobFailed  JobStatus = "failed"
)

// Job is a single background task tracked by JobStore — the async
// counterpart to a blocking index/crawl/reindex call.
type Job struct {
	ID        string     `json:"id"`
	Kind      string     `json:"kind"`
	Status    JobStatus  `json:"status"`
	Progress  string     `json:"progress,omitempty"`
	Result    any        `json:"result,omitempty"`
	Error     string     `json:"error,omitempty"`
	CreatedAt time.Time  `json:"createdAt"`
	StartedAt *time.Time `json:"startedAt,omitempty"`
	EndedAt   *time.Time `json:"endedAt,omitempty"`
}

// JobWork is the unit of work a job runs: given a progress reporter, it does
// the actual work and returns the flow's normal result.
type JobWork func(progress func(string)) (any, error)

// maxJobHistory caps how many jobs JobStore keeps, so a long-running server's
// job list doesn't grow without bound.
const maxJobHistory = 200

// JobStore tracks jobs in memory and runs them one at a time on a single
// background worker rather than a pool. Ollama's embed backend is already
// sized conservatively (see embedBatchSize in store.go); running an index and
// a crawl concurrently would mean two embed streams competing for the same
// constrained backend. A queued job finishing a few minutes later than it
// could in theory is a much smaller cost than reintroducing that risk.
type JobStore struct {
	mu    sync.Mutex
	jobs  map[string]*Job
	order []string // insertion order, oldest first; trimmed to maxJobHistory
	queue chan queuedJob
}

type queuedJob struct {
	id   string
	work JobWork
}

func NewJobStore() *JobStore {
	s := &JobStore{
		jobs:  make(map[string]*Job),
		queue: make(chan queuedJob, 64),
	}
	go s.worker()
	return s
}

func (s *JobStore) worker() {
	for qj := range s.queue {
		s.run(qj)
	}
}

func (s *JobStore) run(qj queuedJob) {
	s.mu.Lock()
	j, ok := s.jobs[qj.id]
	if !ok {
		s.mu.Unlock()
		return // evicted from history before its turn came up
	}
	now := time.Now()
	j.Status = JobRunning
	j.StartedAt = &now
	s.mu.Unlock()

	progress := func(msg string) {
		s.mu.Lock()
		if jj, ok := s.jobs[qj.id]; ok {
			jj.Progress = msg
		}
		s.mu.Unlock()
	}

	result, err := qj.work(progress)

	s.mu.Lock()
	endedAt := time.Now()
	j.EndedAt = &endedAt
	if err != nil {
		j.Status = JobFailed
		j.Error = err.Error()
	} else {
		j.Status = JobDone
		j.Result = result
	}
	s.mu.Unlock()
}

// Enqueue creates a pending job of the given kind and schedules work on the
// background worker. It returns immediately once the job is queued.
//
// If the queue is still full after enqueueTimeout, it returns
// ErrJobQueueFull instead of blocking the caller (an HTTP handler goroutine)
// indefinitely.
func (s *JobStore) Enqueue(kind string, work JobWork) (*Job, error) {
	j := &Job{
		ID:        newJobID(),
		Kind:      kind,
		Status:    JobPending,
		CreatedAt: time.Now(),
	}

	// Registered before the send, not after: the worker goroutine looks the
	// job up by ID as soon as it receives it, so registering afterwards would
	// race a worker that dequeues before this function gets back to its lock.
	s.mu.Lock()
	s.jobs[j.ID] = j
	s.order = append(s.order, j.ID)
	if len(s.order) > maxJobHistory {
		oldest := s.order[0]
		s.order = s.order[1:]
		delete(s.jobs, oldest)
	}
	s.mu.Unlock()

	timer := time.NewTimer(enqueueTimeout)
	defer timer.Stop()
	select {
	case s.queue <- queuedJob{id: j.ID, work: work}:
		return j, nil
	case <-timer.C:
		s.mu.Lock()
		delete(s.jobs, j.ID)
		for i, id := range s.order {
			if id == j.ID {
				s.order = append(s.order[:i], s.order[i+1:]...)
				break
			}
		}
		s.mu.Unlock()
		return nil, ErrJobQueueFull
	}
}

// Get returns a snapshot of the job with the given ID.
func (s *JobStore) Get(id string) (Job, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[id]
	if !ok {
		return Job{}, false
	}
	return *j, true
}

// List returns up to limit jobs, newest first. limit <= 0 means "all".
func (s *JobStore) List(limit int) []Job {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := len(s.order)
	if limit <= 0 || limit > n {
		limit = n
	}
	result := make([]Job, 0, limit)
	for i := n - 1; i >= 0 && len(result) < limit; i-- {
		result = append(result, *s.jobs[s.order[i]])
	}
	return result
}

func newJobID() string {
	return NewRandomHexID(8)
}

// RunStreamingFlowAsJob drains a streaming flow to completion, forwarding
// each intermediate value to progress and returning the final Output as the
// job's result — the same value genkit.Handler would have returned
// synchronously. ctx must be the app's long-lived context, not an HTTP
// request's: the request context is cancelled once the handler that enqueued
// this job returns, well before the flow finishes.
func RunStreamingFlowAsJob[In, Out any](ctx context.Context, flow *core.Flow[In, Out, string], input In, progress func(string)) (any, error) {
	for v, err := range flow.Stream(ctx, input) {
		if err != nil {
			return nil, err
		}
		if v.Done {
			return v.Output, nil
		}
		progress(v.Stream)
	}
	return nil, fmt.Errorf("stream ended without a final result")
}
