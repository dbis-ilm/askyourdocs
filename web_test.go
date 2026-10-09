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

func allText(paras []Paragraph) string {
	var sb strings.Builder
	for _, p := range paras {
		sb.WriteString(p.Text)
		sb.WriteString("\n")
	}
	return sb.String()
}

// Portal-style pages (JSF/HISinOne) wrap the whole content in one <form>
// inside "*_navi_*" wrapper classes. Neither may hide the content.
func TestHTMLContentInsideFormAndNaviClasses(t *testing.T) {
	long := strings.Repeat("Bewerbungen sind im Online-Portal möglich. ", 12)
	doc := `<html><body>
	<nav><a href="/x">Hauptnavigation</a></nav>
	<div class="content_navi_off content_max_navi_off"><div id="contentFrame">
	  <form id="startPage"><h1>Bewerbung</h1><p>` + long + `</p></form>
	</div></div>
	<footer>Impressum</footer></body></html>`

	paras, err := HTMLToStructuredText(strings.NewReader(doc))
	if err != nil {
		t.Fatalf("HTMLToStructuredText: %v", err)
	}
	got := allText(paras)
	if !strings.Contains(got, "Online-Portal") || !strings.Contains(got, "Bewerbung") {
		t.Errorf("content inside form lost: %q", got)
	}
	if strings.Contains(got, "Hauptnavigation") || strings.Contains(got, "Impressum") {
		t.Errorf("boilerplate leaked: %q", got)
	}
}

// Small forms (search, login) are still dropped, even next to content.
func TestHTMLSmallFormsStillSkipped(t *testing.T) {
	const doc = `<html><body><main>
	  <p>Inhalt der Seite.</p>
	  <form action="/search"><p>Suche im Portal</p><input name="q"></form>
	  <form><select><option>` + `Afghanistan Albanien Algerien Andorra Angola Argentinien Armenien Aserbaidschan Australien Bahamas Bahrain Bangladesch Barbados Belarus Belgien Belize Benin Bhutan Bolivien Bosnien Botswana Brasilien Brunei Bulgarien Burkina Burundi Chile China Costa Dänemark Deutschland Dominica Ecuador` + `</option></select></form>
	</main></body></html>`

	paras, err := HTMLToStructuredText(strings.NewReader(doc))
	if err != nil {
		t.Fatalf("HTMLToStructuredText: %v", err)
	}
	got := allText(paras)
	if !strings.Contains(got, "Inhalt der Seite") {
		t.Errorf("content lost: %q", got)
	}
	if strings.Contains(got, "Suche im Portal") || strings.Contains(got, "Afghanistan") {
		t.Errorf("small form leaked: %q", got)
	}
}

func TestIsBoilerplateWordBoundaries(t *testing.T) {
	tests := []struct {
		class string
		want  bool
	}{
		{"main-nav", true}, {"topNav", true}, {"nav", true}, {"navbar", true},
		{"ad_slot", true}, {"sidebar-left", true}, {"cookie-banner", true},
		{"content_navi_off", false}, {"road-map", false}, {"navigator-info", false},
		{"article", false},
	}
	for _, tc := range tests {
		n := &html.Node{Type: html.ElementNode, Data: "div", Attr: []html.Attribute{{Key: "class", Val: tc.class}}}
		if got := isBoilerplate(n); got != tc.want {
			t.Errorf("isBoilerplate(class=%q) = %v, want %v", tc.class, got, tc.want)
		}
	}
}
