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
	"fmt"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/ledongthuc/pdf"
)

// parentText must stay bounded on long sections but still carry a short
// section whole.
func TestParentTextIsCapped(t *testing.T) {
	long := strings.Repeat("Ein Satz mit etwas Inhalt. ", 40) // ~1080 chars
	paras := []Paragraph{{Text: "TOP 1 Langer Punkt", Heading: true}}
	for i := 0; i < 20; i++ {
		paras = append(paras, Paragraph{Text: long, PageNum: 1})
	}

	chunks := BuildChunks(paras, 1500, 200)
	if len(chunks) == 0 {
		t.Fatal("no chunks produced")
	}
	const limit = 3000
	for i, c := range chunks {
		// The window may overshoot by the chunk's own size when the chunk
		// alone is already large; what must not happen is parentText growing
		// to the full section (20 * ~1080 chars).
		if len(c.ParentText) > limit+len(c.Text) {
			t.Errorf("chunk %d: parentText = %d chars, want <= %d", i, len(c.ParentText), limit+len(c.Text))
		}
	}

	short := []Paragraph{
		{Text: "TOP 2 Kurzer Punkt", Heading: true},
		{Text: "Erster Absatz.", PageNum: 2},
		{Text: "Zweiter Absatz.", PageNum: 2},
	}
	sc := BuildChunks(short, 1500, 200)
	if len(sc) != 1 {
		t.Fatalf("short section: got %d chunks, want 1", len(sc))
	}
	for _, want := range []string{"TOP 2 Kurzer Punkt", "Erster Absatz.", "Zweiter Absatz."} {
		if !strings.Contains(sc[0].ParentText, want) {
			t.Errorf("short section parentText missing %q:\n%s", want, sc[0].ParentText)
		}
	}
}

func TestSplitTextAtSentences(t *testing.T) {
	sentence := "Der Senat beschliesst die Aenderung der Ordnung einstimmig. "
	text := strings.TrimSpace(strings.Repeat(sentence, 20))

	parts := splitTextAtSentences(text, 300)
	if len(parts) < 2 {
		t.Fatalf("got %d parts, want the text split up", len(parts))
	}
	for i, p := range parts {
		if len(p) > 300 {
			t.Errorf("part %d is %d bytes, want <= 300", i, len(p))
		}
		if !strings.HasSuffix(p, ".") {
			t.Errorf("part %d does not end at a sentence boundary: %q", i, p)
		}
	}
	if got := strings.Join(parts, " "); got != text {
		t.Error("splitting changed the text")
	}

	// Text without any boundary must still be cut, and never inside a rune.
	long := strings.Repeat("ä", 400)
	for _, p := range splitTextAtSentences(long, 100) {
		if !utf8.ValidString(p) {
			t.Fatalf("split produced invalid UTF-8: %q", p)
		}
	}
}

func TestMergePageBreaks(t *testing.T) {
	paras := []Paragraph{
		{Text: "Der Senat nimmt den Bericht der Fakultaet zur Kenntnis und beauftragt das", PageNum: 3},
		{Text: "Praesidium mit der weiteren Abstimmung.", PageNum: 4},
		{Text: "Ein neuer Absatz.", PageNum: 4},
		{Text: "Die Kommission wurde neu zusammenge-", PageNum: 5},
		{Text: "setzt.", PageNum: 6},
	}

	merged := mergePageBreaks(paras)
	if len(merged) != 3 {
		t.Fatalf("got %d paragraphs, want 3: %v", len(merged), merged)
	}
	if !strings.Contains(merged[0].Text, "beauftragt das Praesidium mit") {
		t.Errorf("page break not joined: %q", merged[0].Text)
	}
	if merged[0].PageNum != 3 {
		t.Errorf("merged paragraph page = %d, want 3", merged[0].PageNum)
	}
	if !strings.Contains(merged[2].Text, "zusammengesetzt.") {
		t.Errorf("hyphen across page break not closed: %q", merged[2].Text)
	}
}

func TestDehyphenate(t *testing.T) {
	in := "Stel-\nlungnahme des Senats zur Bund-\nLänder-Kommission"
	want := "Stellungnahme des Senats zur Bund-\nLänder-Kommission"
	if got := dehyphenate(in); got != want {
		t.Errorf("dehyphenate = %q, want %q", got, want)
	}
}

func TestStripRunningLines(t *testing.T) {
	pages := make([][]pdfLine, 5)
	for i := range pages {
		pages[i] = []pdfLine{
			{y: 800, Text: "Technische Universität Ilmenau — Protokoll"},
			{y: 700, Text: fmt.Sprintf("Inhalt von Seite %d, der erhalten bleiben muss.", i+1)},
			{y: 50, Text: fmt.Sprintf("Seite %d von 5", i+1)},
		}
	}

	stripRunningLines(pages)

	for i, lines := range pages {
		if len(lines) != 1 {
			t.Fatalf("page %d: got %d lines, want 1: %v", i, len(lines), lines)
		}
		if !strings.HasPrefix(lines[0].Text, "Inhalt von Seite") {
			t.Errorf("page %d kept the wrong line: %q", i, lines[0].Text)
		}
	}
}

func TestLinesToTextUsesVerticalGaps(t *testing.T) {
	lines := []pdfLine{
		{y: 700, Text: "Erste Zeile eines Absatzes"},
		{y: 686, Text: "zweite Zeile desselben Absatzes."},
		{y: 650, Text: "Ein neuer Absatz nach groesserem Abstand."},
	}
	got := linesToText(lines)
	want := "Erste Zeile eines Absatzes\nzweite Zeile desselben Absatzes.\n\nEin neuer Absatz nach groesserem Abstand."
	if got != want {
		t.Errorf("linesToText =\n%q\nwant\n%q", got, want)
	}
}

// A chunk must stay within maxSize including the section heading that flush()
// prepends and any overlap carried over from the previous chunk.
func TestChunkSizeIncludesHeadingAndOverlap(t *testing.T) {
	paras := []Paragraph{{Text: "TOP 7 Ein recht ausfuehrlich formulierter Tagesordnungspunkt", Heading: true}}
	for i := 0; i < 12; i++ {
		paras = append(paras, Paragraph{
			Text:    strings.TrimSpace(strings.Repeat("Ein Satz mit Inhalt. ", 30)), // ~600 chars
			PageNum: 1,
		})
	}

	const maxSize = 1500
	chunks := BuildChunks(paras, maxSize, 200)
	if len(chunks) == 0 {
		t.Fatal("no chunks produced")
	}
	for i, c := range chunks {
		if len(c.Text) > maxSize {
			t.Errorf("chunk %d is %d chars, want <= %d", i, len(c.Text), maxSize)
		}
		if c.Top != "TOP 7" {
			t.Errorf("chunk %d: top = %q, want \"TOP 7\"", i, c.Top)
		}
	}
}

// A long breadcrumb prefix must not push the first chunk of a section past
// maxSize — the size guard in buildChunks' main loop only fires once a chunk
// already has content, so without reserving room up front, a paragraph that
// fits maxSize on its own can still overflow once flush() prepends a long
// breadcrumb to it.
func TestChunkSizeReservesRoomForLongBreadcrumb(t *testing.T) {
	breadcrumb := strings.TrimSpace(strings.Repeat("Sehr lange Kategorie > ", 20)) // ~460 chars
	paras := []Paragraph{
		{Text: "Sehr lange Kategorie", Heading: true, Breadcrumb: breadcrumb},
		{Text: strings.TrimSpace(strings.Repeat("Ein Satz mit Inhalt. ", 68)), PageNum: 1}, // ~1360 chars, under maxSize on its own
	}

	const maxSize = 1500
	chunks := BuildChunks(paras, maxSize, 200)
	if len(chunks) == 0 {
		t.Fatal("no chunks produced")
	}
	for i, c := range chunks {
		if len(c.Text) > maxSize {
			t.Errorf("chunk %d is %d chars (breadcrumb %d chars), want <= %d", i, len(c.Text), len(breadcrumb), maxSize)
		}
	}
}

// A chunk's citation must point at the pages parentText actually carries, not
// just the chunk's own narrow paraIndices — the QA flow answers from
// parentText (see main.go's "Prefer parentText" comments), so a citation
// built from the chunk's own range alone can name pages the answer wasn't
// even drawn from once parentText grows past them.
func TestChunkPagesMatchParentTextWindow(t *testing.T) {
	paras := []Paragraph{{Text: "TOP 1 Lange Sitzung", Heading: true}}
	for i := 1; i <= 20; i++ {
		paras = append(paras, Paragraph{Text: fmt.Sprintf("Absatz %d mit Inhalt zur Verhandlung.", i), PageNum: i})
	}

	// Small enough to split into several chunks, but the whole section
	// (~900 chars) still fits inside parentSize (3000), so every chunk's
	// parentText should expand to the full page 1-20 span.
	chunks := BuildChunks(paras, 100, 20)
	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want several to exercise the window growth", len(chunks))
	}

	for i, c := range chunks {
		if c.Pages != "1-20" {
			t.Errorf("chunk %d: pages = %q, want %q (parentText spans the whole section)\nparentText:\n%s", i, c.Pages, "1-20", c.ParentText)
		}
	}
}

func TestExtractSectionLabel(t *testing.T) {
	cases := map[string]string{
		"TOP 1":                     "TOP 1",
		"TOP 10 Berufungsverfahren": "TOP 10",
		"TOP N1":                    "TOP N1",
		"TOP N2 Honorarprofessur":   "TOP N2",
		"N1. Antrag der Fakultaet":  "TOP N1",
		"§ 1 Geltungsbereich":       "§ 1",
		"§ 12 Modulabschluss":       "§ 12",
		"A. Allgemeine Regelungen":  "A",
		"II. Modulabschluss":        "II",
		"Bestaetigt:":               "",
		"1. Begruessung":            "",
		// Cross-references to another law's subsection, not headings — see
		// isParagraphSection.
		"§ 54 Absatz 6 ThürHG anwesend sein.":                               "",
		"§ 54 Absatz 11 ThürHG der Prüfungsausschuss ein Attest verlangen.": "",
	}
	for in, want := range cases {
		if got := extractSectionLabel(in); got != want {
			t.Errorf("extractSectionLabel(%q) = %q, want %q", in, got, want)
		}
		if want != "" && !isSectionBoundary(in) {
			t.Errorf("isSectionBoundary(%q) = false, want true", in)
		}
	}

	if isSectionBoundary("§ 54 Absatz 6 ThürHG anwesend sein.") {
		t.Error(`isSectionBoundary("§ 54 Absatz 6 ThürHG anwesend sein.") = true, want false (it's a cross-reference, not a heading)`)
	}
}

func TestIsNumberedSection(t *testing.T) {
	cases := map[string]bool{
		"1. Begruessung":           true,
		"1.2 Zwischenueberschrift": true,
		"1.2.3 Tiefe Gliederung":   true,
		// The bug fixed earlier this session: a bare digit run followed by a
		// space must not match without a '.' — see the doc comment on
		// isNumberedSection and its "sawDot" flag.
		"50 Mitglieder waren anwesend.": false,
		"12.5 kg Gewicht":               true, // still matches — a decimal number followed by a space is indistinguishable from "1.2 Titel" by this heuristic alone
		"":                              false,
		"Kein Zahlenanfang":             false,
		"1":                             false, // no trailing space at all
	}
	for in, want := range cases {
		if got := isNumberedSection(in); got != want {
			t.Errorf("isNumberedSection(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestIsHeading(t *testing.T) {
	cases := map[string]bool{
		"TOP 3":               true,  // isSectionBoundary path
		"NUR GROSSBUCHSTABEN": true,  // ALL CAPS path
		"ab":                  false, // ALL CAPS check needs len >= 3
		"1.2 Titel":           true,  // isNumberedSection path
		"Kurzer Titel:":       true,  // trailing-colon path
		"Ein ganz normaler Satz ohne Doppelpunkt am Ende, der als Fließtext gemeint ist und daher lang genug sein sollte, um jede Heading-Heuristik zu verfehlen.": false,
		"":                                   false,
		"Zeile eins\nZeile zwei\nZeile drei": false, // more than 2 lines
	}
	for in, want := range cases {
		if got := isHeading(in); got != want {
			t.Errorf("isHeading(%q) = %v, want %v", in, got, want)
		}
	}

	// A line over 120 chars never counts as a heading, regardless of shape.
	long := strings.Repeat("x", 121)
	if isHeading(long) {
		t.Errorf("isHeading(121-char line) = true, want false")
	}
}

func TestContainsLetter(t *testing.T) {
	cases := map[string]bool{
		"abc":   true,
		"123":   false,
		"":      false,
		"12a34": true,
		"§ 12":  false,
	}
	for in, want := range cases {
		if got := containsLetter(in); got != want {
			t.Errorf("containsLetter(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestSplitParagraphs(t *testing.T) {
	text := "Erster Absatz.\n\nZweiter Absatz.\n\n\n\nDritter Absatz nach mehreren Leerzeilen."
	got := splitParagraphs(text)
	want := []string{"Erster Absatz.", "Zweiter Absatz.", "Dritter Absatz nach mehreren Leerzeilen."}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("splitParagraphs = %#v, want %#v", got, want)
	}
}

func TestSplitParagraphsDropsEmptyBlocks(t *testing.T) {
	got := splitParagraphs("\n\n  \n\nEinziger Absatz.\n\n   \n\n")
	want := []string{"Einziger Absatz."}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("splitParagraphs = %#v, want %#v", got, want)
	}
}

// parseIntoParagraphs is the bridge between readPDFPages' output and
// buildChunks' input — pure once given pageContent, so testable without a
// real PDF.
func TestParseIntoParagraphs(t *testing.T) {
	pages := []pageContent{
		{PageNum: 1, Text: "TOP 1\n\nDer Senat begrüßt die Gäste."},
		{PageNum: 2, Text: "TOP 2\n\nDer Antrag wird angenommen."},
	}
	got := parseIntoParagraphs(pages)

	if len(got) != 4 {
		t.Fatalf("got %d paragraphs, want 4: %#v", len(got), got)
	}
	if !got[0].Heading || got[0].Text != "TOP 1" || got[0].PageNum != 1 {
		t.Errorf("paragraph 0 = %+v, want heading TOP 1 on page 1", got[0])
	}
	if got[1].Heading || got[1].Text != "Der Senat begrüßt die Gäste." {
		t.Errorf("paragraph 1 = %+v, want non-heading body text", got[1])
	}
	if !got[2].Heading || got[2].Text != "TOP 2" || got[2].PageNum != 2 {
		t.Errorf("paragraph 2 = %+v, want heading TOP 2 on page 2", got[2])
	}
}

func TestParseIntoParagraphsEmptyInput(t *testing.T) {
	if got := parseIntoParagraphs(nil); len(got) != 0 {
		t.Errorf("parseIntoParagraphs(nil) = %#v, want empty", got)
	}
}

// Real protocol PDFs don't put a blank line between a "TOP N" heading and its
// body — the heading and the first sentence of its body share one \n-joined
// text block (unlike TestParseIntoParagraphs above, which uses "\n\n" and so
// never exercises splitOnSectionBoundaries' actual job). Before the fix, the
// whole block (heading + however many body lines followed, right up to the
// next detected heading) was kept as one non-heading paragraph, because
// isHeading rejects anything over 2 lines — silently merging an entire
// agenda item's content into whichever heading preceded it. Reproduces the
// exact shape found in 362.Senat_Protokoll that caused wrong citations for
// "Was wurde zum Stand der Verhandlungen zur Rahmenvereinbarung berichtet?".
func TestParseIntoParagraphsSplitsHeadingFromUnseparatedBody(t *testing.T) {
	pages := []pageContent{
		{PageNum: 2, Text: "N6. Berichterstattung zum Stand der laufenden Berufungsverfahren"},
		{PageNum: 2, Text: "TOP 1\nDer Vorsitzende begrüßt die Mitglieder.\nEr eröffnet die Sitzung.\nDie Tagesordnung wird bestätigt."},
		{PageNum: 3, Text: "TOP 2\nDer Vorsitzende informiert über die Verlängerung der Rahmenvereinbarung V."},
	}
	got := parseIntoParagraphs(pages)

	var headings []string
	for _, p := range got {
		if p.Heading {
			headings = append(headings, p.Text)
		}
	}
	want := []string{"N6. Berichterstattung zum Stand der laufenden Berufungsverfahren", "TOP 1", "TOP 2"}
	if !reflect.DeepEqual(headings, want) {
		t.Fatalf("headings = %#v, want %#v (full paragraphs: %#v)", headings, want, got)
	}

	for _, p := range got {
		if !p.Heading && strings.Contains(p.Text, "Rahmenvereinbarung") {
			if p.PageNum != 3 {
				t.Errorf("Rahmenvereinbarung body landed on page %d, want 3 (TOP 2's own page): %+v", p.PageNum, p)
			}
			return
		}
	}
	t.Fatal("no body paragraph mentioning Rahmenvereinbarung found")
}

// isProtocolTOP must not treat "TOP N" mid-sentence as a new heading — both
// examples are real text from 362.Senat_Protokoll that newly risked becoming
// false headings once splitOnSectionBoundaries started isolating boundary
// lines instead of leaving them merged with their body (see the fix above).
func TestIsProtocolTOPRejectsMidSentenceMentions(t *testing.T) {
	cases := map[string]bool{
		"TOP 1":                      true,
		"TOP 3 Antrag der Fakultaet": true,
		"TOP 2:":                     true,
		"TOP N1":                     true,
		"TOP 8 bestätigt: Auf Seite 7, letzter Absatz werden Satz 2-4 gestrichen.": false,
		"TOP 10b 13 von 19 stimmberechtigten Senatorinnen und Senatoren anwesend.": false,
	}
	for in, want := range cases {
		if got := isProtocolTOP(in); got != want {
			t.Errorf("isProtocolTOP(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestRowTextConcatenatesWhenSpaceGlyphsPresent(t *testing.T) {
	items := []pdf.Text{{S: "Hallo"}, {S: " "}, {S: "Welt"}}
	if got := rowText(items); got != "Hallo Welt" {
		t.Errorf("rowText = %q, want %q", got, "Hallo Welt")
	}
}

func TestRowTextFallsBackToBuildRowTextWithoutSpaceGlyphs(t *testing.T) {
	// No item is pure whitespace, so rowText must delegate to buildRowText's
	// gap analysis instead of naive concatenation.
	items := []pdf.Text{
		{S: "Hallo", X: 0, FontSize: 12},
		{S: "Welt", X: 40, FontSize: 12}, // large gap relative to font size → word boundary
	}
	if got := rowText(items); got != "Hallo Welt" {
		t.Errorf("rowText = %q, want %q", got, "Hallo Welt")
	}
}

func TestBuildRowTextSingleItem(t *testing.T) {
	if got := buildRowText([]pdf.Text{{S: "Allein"}}); got != "Allein" {
		t.Errorf("buildRowText(single item) = %q, want %q", got, "Allein")
	}
}

func TestBuildRowTextUniformSmallGapsStayOneWord(t *testing.T) {
	// Uniform, small (intra-character) gaps relative to font size: no spaces
	// inserted — this is what a PDF that doesn't emit space glyphs but does
	// space out individual characters of one word looks like.
	items := []pdf.Text{
		{S: "H", X: 0, FontSize: 12},
		{S: "i", X: 1, FontSize: 12},
	}
	if got := buildRowText(items); got != "Hi" {
		t.Errorf("buildRowText = %q, want %q (uniform tiny gaps, one word)", got, "Hi")
	}
}

func TestBuildRowTextUniformLargeGapsAlwaysSpace(t *testing.T) {
	// Uniform gaps that are all large relative to font size: every gap is a
	// word boundary.
	items := []pdf.Text{
		{S: "Wort1", X: 0, FontSize: 12},
		{S: "Wort2", X: 20, FontSize: 12},
		{S: "Wort3", X: 40, FontSize: 12},
	}
	got := buildRowText(items)
	want := "Wort1 Wort2 Wort3"
	if got != want {
		t.Errorf("buildRowText = %q, want %q", got, want)
	}
}
