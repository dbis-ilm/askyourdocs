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
	"strings"
	"testing"
)

func TestExtractTextParagraphsMarkdown(t *testing.T) {
	src := "# Wohnung\n\nEinleitung.\n\n## Kündigung\n\nSchriftlich mit Unterschrift.\nNicht per E-Mail.\n\n```\n# kein Heading\n\nnoch Code\n```\n"
	paras, err := ExtractTextParagraphs(strings.NewReader(src), true)
	if err != nil {
		t.Fatal(err)
	}
	if len(paras) != 5 {
		t.Fatalf("got %d paragraphs: %+v", len(paras), paras)
	}
	if !paras[2].Heading || paras[3].Breadcrumb != "Wohnung > Kündigung" || paras[3].Text != "Schriftlich mit Unterschrift.\nNicht per E-Mail." {
		t.Fatalf("heading/breadcrumb wrong: %+v", paras)
	}
	if paras[4].Text != "```\n# kein Heading\n\nnoch Code\n```" {
		t.Fatalf("fenced block mangled: %q", paras[4].Text)
	}
	if len(BuildChunks(paras, 1200, 150)) == 0 {
		t.Fatal("no chunks")
	}

	plain, _ := ExtractTextParagraphs(strings.NewReader("# not a heading\n\nsecond"), false)
	if len(plain) != 2 || plain[0].Heading || plain[0].Text != "# not a heading" {
		t.Fatalf("plain text: %+v", plain)
	}
}

func TestPlainText(t *testing.T) {
	paras := []Paragraph{
		{Text: "Protokoll", PageNum: 1, Heading: true},
		{Text: "Anwesend: A, B", PageNum: 1},
		{Text: "  ", PageNum: 1},
		{Text: "Beschluss", PageNum: 2},
	}
	if got, want := PlainText(paras), "[Page 1]\n\n## Protokoll\n\nAnwesend: A, B\n\n[Page 2]\n\nBeschluss"; got != want {
		t.Fatalf("PlainText = %q, want %q", got, want)
	}
	if PlainText([]Paragraph{{Text: "x"}, {Text: "y"}}) != "x\n\ny" {
		t.Error("non-paged text got page markers")
	}
}

func TestSplitText(t *testing.T) {
	text := "aaaa\n\nbbbb\n\ncccc\n\n" + strings.Repeat("d", 25) + "\n\nee"
	parts := SplitText(text, 12)
	want := []string{"aaaa\n\nbbbb", "cccc", "dddddddddddd", "dddddddddddd", "d\n\nee"}
	if strings.Join(parts, "|") != strings.Join(want, "|") {
		t.Fatalf("parts = %q, want %q", parts, want)
	}
	for _, p := range parts {
		if len([]rune(p)) > 12 {
			t.Errorf("part too long: %q", p)
		}
	}
	if got := SplitText("äöü", 2); len(got) != 2 || got[0] != "äö" {
		t.Errorf("runes: %q", got)
	}
}

func TestParsePages(t *testing.T) {
	sel, err := ParsePages("1, 3-4,7-")
	if err != nil {
		t.Fatal(err)
	}
	for p, want := range map[int]bool{1: true, 2: false, 3: true, 4: true, 5: false, 7: true, 99: true} {
		if sel(p) != want {
			t.Errorf("page %d: %v", p, !want)
		}
	}
	for _, bad := range []string{"0", "4-2", "a", "1-x", "-3", "1,,2"} {
		if _, err := ParsePages(bad); err == nil {
			t.Errorf("ParsePages(%q) accepted", bad)
		}
	}
	if sel, err := ParsePages(" "); sel != nil || err != nil {
		t.Error("empty selection should mean all pages")
	}
}
