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
	"math"
	"sort"
	"strings"

	"github.com/ledongthuc/pdf"
)

// Two-column parts of a page (a list of members next to a list of guests,
// a two-column layout) used to come out line by line, both columns glued
// together: "Frau A. Schneider Herr Prof. B. Sattler" mixed a member and an
// excused guest in one line, and no reader could tell them apart. Lines are
// now split at column-wide gaps, and runs of lines sharing the same columns
// are written column by column, or row by row for tables.

const (
	// columnGapFactor times the font size separates columns; word spaces
	// are a fraction of the font size, column gaps several times it.
	columnGapFactor = 3.0
	minColumnGap    = 20.0 // points
	// columnTolerance is how far a segment's start may be from its column.
	columnTolerance = 15.0 // points
	// proseSegmentChars: segments this long on average are running text in
	// columns, not table cells.
	proseSegmentChars = 40
	// runGapFactor times the run's line spacing ends a column run.
	runGapFactor = 1.6
)

// lineSegments splits one line's items (in drawing order) at column-wide
// gaps. Items are grouped by position, but each segment keeps the drawing
// order of its items, which rowText relies on.
func lineSegments(items []pdf.Text) []pdfSegment {
	if len(items) == 0 {
		return nil
	}
	fontSize := items[0].FontSize
	if fontSize <= 0 {
		fontSize = 12
	}
	gap := math.Max(columnGapFactor*fontSize, minColumnGap)

	idx := make([]int, len(items))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return items[idx[a]].X < items[idx[b]].X })
	cluster := make([]int, len(items))
	n, end := 0, items[idx[0]].X+items[idx[0]].W
	for k, i := range idx {
		if k > 0 && items[i].X-end > gap && strings.TrimSpace(items[i].S) != "" {
			n++
		}
		cluster[i] = n
		end = math.Max(end, items[i].X+items[i].W)
	}

	segs := make([]pdfSegment, 0, n+1)
	for c := 0; c <= n; c++ {
		var run []pdf.Text
		x0, x1 := math.Inf(1), math.Inf(-1)
		for i, it := range items {
			if cluster[i] != c {
				continue
			}
			run = append(run, it)
			if strings.TrimSpace(it.S) != "" {
				x0 = math.Min(x0, it.X)
				x1 = math.Max(x1, it.X+it.W)
			}
		}
		if text := NormalizeSpaces(rowText(run)); strings.TrimSpace(text) != "" {
			segs = append(segs, pdfSegment{x0: x0, x1: x1, text: strings.TrimSpace(text)})
		}
	}
	return segs
}

// arrangeColumns rewrites runs of lines that share column positions. A run
// starts at a line with two or more segments and continues while every
// segment of a line starts at one of its columns and stays inside it; a
// line spanning several columns (running text below a list) ends it.
func arrangeColumns(lines []pdfLine) []pdfLine {
	out := make([]pdfLine, 0, len(lines))
	for i := 0; i < len(lines); {
		if len(lines[i].segs) < 2 {
			out = append(out, lines[i])
			i++
			continue
		}
		cols := make([]float64, len(lines[i].segs))
		for c, s := range lines[i].segs {
			cols[c] = s.x0
		}
		column := func(s pdfSegment) int {
			for c, x := range cols {
				if math.Abs(s.x0-x) <= columnTolerance {
					if c+1 < len(cols) && s.x1 > cols[c+1]-columnTolerance {
						return -1 // runs into the next column
					}
					return c
				}
			}
			return -1
		}
		// The run's line spacing; a clearly larger gap (the space after a
		// list, before the next heading) ends the run.
		spacing := 0.0
		if i+1 < len(lines) {
			spacing = lines[i].y - lines[i+1].y
		}
		j := i
		for j < len(lines) && len(lines[j].segs) > 0 {
			if j > i && spacing > 0 && lines[j-1].y-lines[j].y > runGapFactor*spacing {
				break
			}
			fits := true
			for _, s := range lines[j].segs {
				if column(s) < 0 {
					fits = false
					break
				}
			}
			if !fits {
				break
			}
			j++
		}
		// A real column shows in several lines at the same place. A single
		// split line is more likely a measuring error (some PDFs report text
		// widths too small, which looks like a gap inside a word): keep it.
		multi := 0
		for _, l := range lines[i:j] {
			if len(l.segs) >= 2 {
				multi++
			}
		}
		if multi < 2 {
			out = append(out, lines[i])
			i++
			continue
		}
		out = append(out, writeColumns(lines[i:j], cols, column)...)
		if j < len(lines) {
			lines[j].brk = true // what follows the run is a new paragraph
		}
		i = j
	}
	return out
}

// writeColumns renders a run of column lines. Tables (every row filled,
// short cells) stay row by row with " | " between cells, so a label keeps
// its value; lists of unequal length, columns with their own headings and
// running text in columns are written column by column.
func writeColumns(run []pdfLine, cols []float64, column func(pdfSegment) int) []pdfLine {
	full, chars, cells, subheading := true, 0, 0, false
	for r, l := range run {
		if len(l.segs) != len(cols) {
			full = false
		}
		for _, s := range l.segs {
			chars += len([]rune(s.text))
			cells++
			if r > 0 && strings.HasSuffix(s.text, ":") {
				subheading = true
			}
		}
	}
	prose := cells > 0 && chars/cells >= proseSegmentChars

	if full && !subheading && !prose {
		rows := make([]pdfLine, len(run))
		for r, l := range run {
			parts := make([]string, len(l.segs))
			for k, s := range l.segs {
				parts[k] = s.text
			}
			rows[r] = pdfLine{y: l.y, Text: strings.Join(parts, " | "), brk: r == 0 && l.brk}
		}
		return rows
	}

	byCol := make([][]pdfLine, len(cols))
	for _, l := range run {
		for _, s := range l.segs {
			c := column(s)
			byCol[c] = append(byCol[c], pdfLine{y: l.y, Text: s.text})
		}
	}
	var out []pdfLine
	for _, col := range byCol {
		if len(col) == 0 {
			continue
		}
		col[0].brk = len(out) > 0 || run[0].brk
		out = append(out, col...)
	}
	return out
}
