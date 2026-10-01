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
	"strings"
	"testing"
)

// A block element that mixes inline text with a nested block must yield both.
// Before the walker fix, the nested <p> was dropped.
func TestHTMLMixedBlockKeepsNestedContent(t *testing.T) {
	const doc = `<html><body><main>
	  <h1>Studium</h1>
	  <div class="teaser">Einleitender Text zum Studiengang.
	    <p>Der eigentliche Inhalt des Absatzes.</p>
	  </div>
	</main></body></html>`

	paras, err := HTMLToStructuredText(strings.NewReader(doc))
	if err != nil {
		t.Fatalf("HTMLToStructuredText: %v", err)
	}

	var joined string
	for _, p := range paras {
		joined += p.Text + "\n"
	}
	for _, want := range []string{"Einleitender Text zum Studiengang.", "Der eigentliche Inhalt des Absatzes."} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in extracted paragraphs:\n%s", want, joined)
		}
	}
}

// Boilerplate inside a mixed block must still be skipped — walkChildren
// bypasses walk() for non-structural containers, so it repeats the check.
func TestHTMLMixedBlockSkipsBoilerplate(t *testing.T) {
	const doc = `<html><body><main>
	  <div class="wrapper">Sichtbarer Text.
	    <nav><p>Navigationslink</p></nav>
	    <div class="sidebar"><p>Seitenleiste</p></div>
	  </div>
	</main></body></html>`

	paras, err := HTMLToStructuredText(strings.NewReader(doc))
	if err != nil {
		t.Fatalf("HTMLToStructuredText: %v", err)
	}
	for _, p := range paras {
		if strings.Contains(p.Text, "Navigationslink") || strings.Contains(p.Text, "Seitenleiste") {
			t.Errorf("boilerplate leaked into paragraph: %q", p.Text)
		}
	}
}
