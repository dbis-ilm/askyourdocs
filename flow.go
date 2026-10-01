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

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

// DefineProgressFlow defines a Genkit streaming flow whose handler reports
// progress through a plain func(string) callback instead of genkit's
// core.StreamCallback[string], and returns Out when done.
//
// This is the common shape of a long-running ingest/crawl flow: an HTTP
// layer can run it as a background job and surface fn's progress messages
// incrementally (see docs/AsyncJobs.md-style patterns), while a synchronous
// caller — the genkit CLI, a test, Flow.Run — just discards the stream and
// gets Out at the end.
func DefineProgressFlow[In, Out any](g *genkit.Genkit, name string, fn func(ctx context.Context, input In, progress func(string)) (Out, error)) *core.Flow[In, Out, string] {
	return genkit.DefineStreamingFlow(g, name,
		func(ctx context.Context, input In, sendChunk core.StreamCallback[string]) (Out, error) {
			return fn(ctx, input, func(msg string) { sendChunk(ctx, msg) })
		},
	)
}
