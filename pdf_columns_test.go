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

	"github.com/ledongthuc/pdf"
)

// line builds a pdfLine at y from segments; each segment is "x0:text" with
// a width of 6 points per character.
func line(y float64, segs ...string) pdfLine {
	var l pdfLine
	l.y = y
	var parts []string
	for _, s := range segs {
		var x float64
		text := s
		if i := strings.Index(s, ":"); i > 0 && s[0] >= '0' && s[0] <= '9' {
			for _, c := range s[:i] {
				x = x*10 + float64(c-'0')
			}
			text = s[i+1:]
		}
		l.segs = append(l.segs, pdfSegment{x0: x, x1: x + 6*float64(len([]rune(text))), text: text})
		parts = append(parts, text)
	}
	l.Text = strings.Join(parts, " ")
	return l
}

func texts(lines []pdfLine) string {
	var b strings.Builder
	for _, l := range lines {
		if l.brk {
			b.WriteString("¶")
		}
		b.WriteString(l.Text + "\n")
	}
	return b.String()
}

func TestLineSegments(t *testing.T) {
	items := []pdf.Text{
		{X: 98, W: 30, S: "Frau", FontSize: 10}, {X: 128, W: 3, S: " ", FontSize: 10}, {X: 131, W: 50, S: "Schneider", FontSize: 10},
		{X: 311, W: 30, S: "Herr", FontSize: 10}, {X: 341, W: 3, S: " ", FontSize: 10}, {X: 344, W: 40, S: "Sattler", FontSize: 10},
	}
	segs := lineSegments(items)
	if len(segs) != 2 || segs[0].text != "Frau Schneider" || segs[1].text != "Herr Sattler" || segs[1].x0 != 311 {
		t.Fatalf("segments: %+v", segs)
	}
	// Headings padded with space glyphs still split at the column.
	padded := []pdf.Text{{X: 76, W: 90, S: "Teilnehmer/-innen:", FontSize: 10}}
	for x := 166.0; x < 300; x += 3 {
		padded = append(padded, pdf.Text{X: x, W: 3, S: " ", FontSize: 10})
	}
	padded = append(padded, pdf.Text{X: 303, W: 30, S: "Gäste:", FontSize: 10})
	if segs := lineSegments(padded); len(segs) != 2 || segs[1].text != "Gäste:" {
		t.Fatalf("padded headings: %+v", segs)
	}
	prose := []pdf.Text{{X: 98, W: 40, S: "Normaler", FontSize: 10}, {X: 138, W: 3, S: " ", FontSize: 10}, {X: 141, W: 30, S: "Text", FontSize: 10}}
	if segs := lineSegments(prose); len(segs) != 1 {
		t.Fatalf("prose split: %+v", segs)
	}
}

// A member list next to guests and the excused: written column by column,
// each list in one piece.
func TestArrangeColumnsList(t *testing.T) {
	lines := []pdfLine{
		line(700, "98:Protokoll der Sitzung"),
		line(660, "98:Teilnehmer/-innen:", "311:Gäste:"),
		line(648, "98:Frau A", "311:Herr Gast"),
		line(636, "98:Herr B"),
		line(624, "98:Frau C", "311:Entschuldigt:"),
		line(612, "98:Herr D", "311:Frau E"),
		line(570, "98:Tagesordnung:"), // after a larger gap
		line(558, "98:TOP 1 Bericht des Präsidenten mit einer langen Zeile über die ganze Breite"),
	}
	got := texts(arrangeColumns(lines))
	want := "Protokoll der Sitzung\nTeilnehmer/-innen:\nFrau A\nHerr B\nFrau C\nHerr D\n¶Gäste:\nHerr Gast\nEntschuldigt:\nFrau E\n¶Tagesordnung:\nTOP 1 Bericht des Präsidenten mit einer langen Zeile über die ganze Breite\n"
	if got != want {
		t.Fatalf("got\n%s\nwant\n%s", got, want)
	}
}

// A table keeps its rows, so a label keeps its value.
func TestArrangeColumnsTable(t *testing.T) {
	lines := []pdfLine{
		line(500, "98:Ja-Stimmen", "300:12"),
		line(488, "98:Nein-Stimmen", "300:3"),
		line(476, "98:Enthaltungen", "300:1"),
	}
	if got := texts(arrangeColumns(lines)); got != "Ja-Stimmen | 12\nNein-Stimmen | 3\nEnthaltungen | 1\n" {
		t.Fatalf("table: %q", got)
	}
}

// A single split line is more likely a measuring error than a column.
func TestArrangeColumnsSingleLine(t *testing.T) {
	l := line(500, "98:Der Dekan der Fakultät für Wirtschaft", "330:swissenschaften und Medien")
	l.Text = "Der Dekan der Fakultät für Wirtschaftswissenschaften und Medien"
	got := arrangeColumns([]pdfLine{l, line(488, "98:berichtet über die Lage der Fakultät und ihre Pläne für das nächste Jahr")})
	if got[0].Text != l.Text || len(got) != 2 {
		t.Fatalf("single split line changed: %+v", got)
	}
}

// Running text in two columns comes column by column.
func TestArrangeColumnsProse(t *testing.T) {
	left := "Dies ist der Anfang eines Absatzes in der linken Spalte"
	right := "und hier steht Text der rechten Spalte im Layout"
	lines := []pdfLine{line(500, "50:"+left, "420:"+right), line(488, "50:"+left, "420:"+right)}
	got := texts(arrangeColumns(lines))
	if !strings.HasPrefix(got, left+"\n"+left+"\n¶"+right) {
		t.Fatalf("prose columns: %q", got)
	}
}

// Headings drawn as one text object are split onto the columns.
func TestArrangeColumnsJoinedHeadings(t *testing.T) {
	head := line(700, "76:Teilnehmer/-innen: Gäste:")
	lines := []pdfLine{
		head,
		line(688, "76:Frau A", "303:Herr Gast"),
		line(676, "76:Herr B", "303:Frau Gast"),
		line(664, "76:Frau C"),
	}
	want := "Teilnehmer/-innen:\nFrau A\nHerr B\nFrau C\n¶Gäste:\nHerr Gast\nFrau Gast\n"
	if got := texts(arrangeColumns(lines)); got != want {
		t.Fatalf("got\n%s\nwant\n%s", got, want)
	}
	if splitHeadings("Gemäß § 3: Die Sitzung beginnt um 10 Uhr: pünktlich", 2) != nil {
		t.Error("sentence taken for headings")
	}
}

// Joined headings above a table become its header row.
func TestArrangeColumnsTableHeadings(t *testing.T) {
	lines := []pdfLine{
		line(500, "98:Stimmen: Anzahl:"),
		line(488, "98:Ja", "300:12"),
		line(476, "98:Nein", "300:3"),
	}
	if got := texts(arrangeColumns(lines)); got != "Stimmen: | Anzahl:\nJa | 12\nNein | 3\n" {
		t.Fatalf("table headings: %q", got)
	}
}

func TestSplitHeadings(t *testing.T) {
	for _, tc := range []struct {
		text string
		k    int
		want int
	}{
		{"Teilnehmer/-innen: Gäste:", 2, 2},
		{"Teilnehmer/-innen: Gäste:", 3, 0},                              // wrong number of columns
		{"Teilnehmer/-innen:", 1, 0},                                     // a single column needs no split
		{"Gäste: Entschuldigt", 2, 0},                                    // does not end in a heading
		{"Die Sitzung beginnt um zehn Uhr und endet spät: Gäste:", 2, 0}, // a sentence
	} {
		if got := splitHeadings(tc.text, tc.k); len(got) != tc.want {
			t.Errorf("splitHeadings(%q, %d) = %q", tc.text, tc.k, got)
		}
	}
}

// Without font sizes the gap falls back to a 12 pt font; empty lines have
// no segments.
func TestLineSegmentsEdgeCases(t *testing.T) {
	if lineSegments(nil) != nil {
		t.Error("segments of an empty line")
	}
	items := []pdf.Text{{X: 50, W: 20, S: "Ja"}, {X: 300, W: 12, S: "12"}}
	if segs := lineSegments(items); len(segs) != 2 {
		t.Fatalf("no font size: %+v", segs)
	}
}

// A segment starting in a column but running into the next ends the run.
func TestArrangeColumnsSegmentCrossesColumn(t *testing.T) {
	lines := []pdfLine{
		line(500, "98:Frau A", "311:Herr B"),
		line(488, "98:Frau C", "311:Herr D"),
		line(476, "98:Ein Satz, der über die Spaltengrenze hinaus bis weit nach rechts läuft"),
	}
	got := arrangeColumns(lines)
	if last := got[len(got)-1]; !last.brk || !strings.HasPrefix(last.Text, "Ein Satz") {
		t.Fatalf("crossing line: %+v", got)
	}
}

// A segment at none of the columns ends the run.
func TestArrangeColumnsOffColumn(t *testing.T) {
	lines := []pdfLine{
		line(500, "98:Frau A", "311:Herr B"),
		line(488, "98:Frau C", "311:Herr D"),
		line(476, "200:Seite 2"),
	}
	got := arrangeColumns(lines)
	if last := got[len(got)-1]; !last.brk || last.Text != "Seite 2" {
		t.Fatalf("off-column line: %+v", got)
	}
}

// Long labels next to amounts stay a table, row by row.
func TestArrangeColumnsLongLabelsWithAmounts(t *testing.T) {
	lines := []pdfLine{
		line(500, "50:Kooperationsprojekte und Zuweisungen aus Landesmitteln", "380:527.000 € 527.000 € 400.000 €"),
		line(488, "50:Hochschulpakt 2020/Zukunftsvertrag SP 1 und SP 2", "380:3.352.154 € 3.352.154 € 3.153.545 €"),
		line(476, "50:zentral vereinnahmte Gebühren/Gemeinkosten", "380:2.800.000 € 2.951.575 € 2.800.000 €"),
	}
	got := texts(arrangeColumns(lines))
	if !strings.Contains(got, "Hochschulpakt 2020/Zukunftsvertrag SP 1 und SP 2 | 3.352.154 €") {
		t.Fatalf("amounts lost their labels:\n%s", got)
	}
}
