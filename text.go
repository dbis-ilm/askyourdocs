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
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// ExtractTextParagraphs splits plain text or Markdown into paragraphs at
// blank lines. With markdown set, ATX headings ("## Title") become heading
// paragraphs and set the breadcrumb of what follows ("Title > Subtitle"),
// and fenced code blocks stay whole. Paragraphs have no page number.
//
// It is not part of ExtractParagraphs's dispatch on purpose: apps use
// SupportedDocumentExt to decide which uploads they accept, and adding .md
// and .txt there would widen that silently.
func ExtractTextParagraphs(r io.Reader, markdown bool) ([]Paragraph, error) {
	var (
		paras    []Paragraph
		buf      []string
		headings []string // index = level-1
		inFence  bool
	)
	crumb := func() string {
		var parts []string
		for _, h := range headings {
			if h != "" {
				parts = append(parts, h)
			}
		}
		return strings.Join(parts, " > ")
	}
	flush := func() {
		text := strings.TrimSpace(strings.Join(buf, "\n"))
		buf = buf[:0]
		if text != "" {
			paras = append(paras, Paragraph{Text: text, Breadcrumb: crumb()})
		}
	}

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), " \t\r")
		trimmed := strings.TrimSpace(line)
		if markdown && strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
			buf = append(buf, line)
			continue
		}
		if markdown && !inFence {
			if level, title, ok := atxHeading(trimmed); ok {
				flush()
				for len(headings) < level {
					headings = append(headings, "")
				}
				headings = append(headings[:level-1], title)
				paras = append(paras, Paragraph{Text: title, Heading: true, Breadcrumb: crumb()})
				continue
			}
		}
		if trimmed == "" && !inFence {
			flush()
			continue
		}
		buf = append(buf, line)
	}
	flush()
	return paras, sc.Err()
}

func atxHeading(line string) (int, string, bool) {
	level := 0
	for level < len(line) && line[level] == '#' {
		level++
	}
	if level == 0 || level > 6 || level >= len(line) || line[level] != ' ' {
		return 0, "", false
	}
	return level, strings.TrimSpace(strings.TrimRight(line[level:], "# ")), true
}

// PlainText renders paragraphs as readable text, e.g. for an agent that
// reads a whole document: paragraphs separated by blank lines, headings as
// "## " lines, and a "[Page n]" marker wherever the page changes (paged
// formats only), so the reader can cite pages.
func PlainText(paras []Paragraph) string {
	var b strings.Builder
	page := 0
	for _, p := range paras {
		text := strings.TrimSpace(p.Text)
		if text == "" {
			continue
		}
		if p.PageNum > 0 && p.PageNum != page {
			page = p.PageNum
			if b.Len() > 0 {
				b.WriteString("\n\n")
			}
			fmt.Fprintf(&b, "[Page %d]", page)
		}
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		if p.Heading {
			b.WriteString("## ")
		}
		b.WriteString(text)
	}
	return b.String()
}

// SplitText cuts text into parts of at most max runes, at paragraph
// boundaries (blank lines) where possible, so each part fits a model's
// context on its own. A single paragraph longer than max is cut hard.
func SplitText(text string, max int) []string {
	if max <= 0 {
		return []string{text}
	}
	var parts []string
	var cur strings.Builder
	curLen := 0
	flush := func() {
		if s := strings.TrimSpace(cur.String()); s != "" {
			parts = append(parts, s)
		}
		cur.Reset()
		curLen = 0
	}
	for _, para := range strings.Split(text, "\n\n") {
		r := []rune(para)
		if curLen > 0 && curLen+len(r)+2 > max {
			flush()
		}
		for len(r) > max {
			flush()
			parts = append(parts, string(r[:max]))
			r = r[max:]
		}
		cur.WriteString(string(r))
		cur.WriteString("\n\n")
		curLen += len(r) + 2
	}
	flush()
	return parts
}

// ParsePages reads a page selection as a user or an agent writes it — "3",
// "1-3", "5-" (to the end) or a comma list of those ("1,4-6") — and returns
// a predicate for page numbers. An empty selection returns nil, meaning all
// pages.
func ParsePages(s string) (func(page int) bool, error) {
	s = strings.ReplaceAll(strings.TrimSpace(s), " ", "")
	if s == "" {
		return nil, nil
	}
	type span struct{ from, to int } // to 0 = open end
	var spans []span
	bad := fmt.Errorf("pages %q: expected e.g. 1, 1-3, 3- or 1,5-7", s)
	for _, f := range strings.Split(s, ",") {
		a, b, isRange := strings.Cut(f, "-")
		from, err := strconv.Atoi(a)
		if err != nil || from < 1 {
			return nil, bad
		}
		to := from
		if isRange {
			if b == "" {
				to = 0
			} else if to, err = strconv.Atoi(b); err != nil || to < from {
				return nil, bad
			}
		}
		spans = append(spans, span{from, to})
	}
	return func(p int) bool {
		for _, sp := range spans {
			if p >= sp.from && (sp.to == 0 || p <= sp.to) {
				return true
			}
		}
		return false
	}, nil
}
