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

	"golang.org/x/net/html"
)

// parseFragmentNode parses s as a full document and returns the first element
// with the given tag name.
func parseFragmentNode(t *testing.T, s, tag string) *html.Node {
	t.Helper()
	doc, err := html.Parse(strings.NewReader(s))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var found *html.Node
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if found != nil {
			return
		}
		if n.Type == html.ElementNode && n.Data == tag {
			found = n
			return
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)
	if found == nil {
		t.Fatalf("no <%s> in %q", tag, s)
	}
	return found
}

func TestHeadingLevel(t *testing.T) {
	for tag, want := range map[string]int{"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6, "p": 0, "": 0, "h7": 0} {
		if got := headingLevel(tag); got != want {
			t.Errorf("headingLevel(%q) = %d, want %d", tag, got, want)
		}
	}
}

func TestIsSiteHeader(t *testing.T) {
	tests := []struct {
		name string
		doc  string
		want bool
	}{
		{"top-level", "<body><header>Site</header></body>", true},
		{"inside main", "<body><main><header>Title</header></main></body>", false},
		{"inside article", "<body><article><header>Title</header></article></body>", false},
		{"nested in div, no content ancestor", "<body><div><header>x</header></div></body>", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n := parseFragmentNode(t, tc.doc, "header")
			if got := isSiteHeader(n); got != tc.want {
				t.Errorf("isSiteHeader = %v, want %v", got, tc.want)
			}
		})
	}

	if isSiteHeader(parseFragmentNode(t, "<body><div>x</div></body>", "div")) {
		t.Error("non-header element must not be a site header")
	}
	if isSiteHeader(&html.Node{Type: html.TextNode, Data: "header"}) {
		t.Error("text node must not be a site header")
	}
}

func TestNodeTextLen(t *testing.T) {
	n := parseFragmentNode(t, "<body><div>  ab <b>cd</b>  </div></body>", "div")
	if got := nodeTextLen(n); got != 4 {
		t.Errorf("nodeTextLen = %d, want 4 (whitespace trimmed per text node)", got)
	}
}

func TestRenderTable(t *testing.T) {
	n := parseFragmentNode(t, `<table>
	  <tr><th>Name</th><th>Wert</th></tr>
	  <tr><td>a</td><td>1</td></tr>
	  <tr><td>b</td></tr>
	</table>`, "table")
	got := renderTable(n)
	want := "| Name | Wert |\n| --- | --- |\n| a | 1 |\n| b |  |"
	// header separator is written without a leading newline after the row
	// newline, so compare line by line ignoring blank-line differences.
	norm := func(s string) []string {
		var out []string
		for _, l := range strings.Split(s, "\n") {
			if strings.TrimSpace(l) != "" {
				out = append(out, l)
			}
		}
		return out
	}
	g, w := norm(got), norm(want)
	if len(g) != len(w) {
		t.Fatalf("renderTable = %q, want lines %q", got, w)
	}
	for i := range w {
		if g[i] != w[i] {
			t.Errorf("line %d = %q, want %q", i, g[i], w[i])
		}
	}
}

func TestRenderTableEmpty(t *testing.T) {
	n := parseFragmentNode(t, "<table></table>", "table")
	if got := renderTable(n); got != "" {
		t.Errorf("renderTable(empty) = %q, want empty", got)
	}
}

func TestSplitTableIfLarge(t *testing.T) {
	header := "| A | B |\n| --- | --- |"
	var rows []string
	for i := 0; i < 10; i++ {
		rows = append(rows, "| xxxxxx | yyyyyy |")
	}
	table := header + "\n" + strings.Join(rows, "\n")

	if parts := splitTableIfLarge(table, len(table)); len(parts) != 1 || parts[0] != table {
		t.Errorf("table within limit must stay whole, got %d parts", len(parts))
	}

	parts := splitTableIfLarge(table, 100)
	if len(parts) < 2 {
		t.Fatalf("got %d parts, want >= 2", len(parts))
	}
	total := 0
	for i, p := range parts {
		if !strings.HasPrefix(p, header) {
			t.Errorf("part %d lacks repeated header: %q", i, p)
		}
		total += strings.Count(p, "xxxxxx")
	}
	if total != 10 {
		t.Errorf("rows across parts = %d, want 10", total)
	}

	// Fewer than 3 lines cannot be split even if oversized.
	if parts := splitTableIfLarge("one line only that is long", 5); len(parts) != 1 {
		t.Errorf("short-line table split into %d parts, want 1", len(parts))
	}
}

func TestRenderList(t *testing.T) {
	ul := parseFragmentNode(t, "<ul><li>eins<ul><li>innen</li></ul></li><li>zwei</li></ul>", "ul")
	if got, want := renderList(ul, 0), "- eins\n  - innen\n- zwei\n"; got != want {
		t.Errorf("ul = %q, want %q", got, want)
	}

	ol := parseFragmentNode(t, "<ol><li>a</li><li>b</li></ol>", "ol")
	if got, want := renderList(ol, 0), "1. a\n2. b\n"; got != want {
		t.Errorf("ol = %q, want %q", got, want)
	}

	if got := renderList(parseFragmentNode(t, "<p>x</p>", "p"), 0); got != "" {
		t.Errorf("non-list = %q, want empty", got)
	}
}

func TestRenderDefinitionList(t *testing.T) {
	dl := parseFragmentNode(t, "<dl><dt>Term</dt><dd>Bedeutung</dd><dt>Zwei</dt><dd>Mehr</dd></dl>", "dl")
	got := renderDefinitionList(dl)
	for _, want := range []string{"Term: Bedeutung", "Zwei: Mehr"} {
		if !strings.Contains(got, want) {
			t.Errorf("renderDefinitionList = %q, missing %q", got, want)
		}
	}
}
