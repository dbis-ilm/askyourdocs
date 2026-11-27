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
