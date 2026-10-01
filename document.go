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
	"math"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ledongthuc/pdf"
)

type pageContent struct {
	PageNum int
	Text    string
}

// Paragraph is one paragraph of extracted document text, annotated with its
// page number, heading status and (for web documents) heading breadcrumb.
type Paragraph struct {
	Text       string
	PageNum    int
	Heading    bool
	Breadcrumb string // hierarchical heading path, e.g. "Products > Enterprise > Pricing"
}

// Chunk is one unit of text ready for embedding, together with the expanded
// context window and citation metadata BuildChunks derived for it.
type Chunk struct {
	Text        string // small chunk for embedding
	ParentText  string // expanded context for LLM
	Pages       string // e.g. "3" or "3-5"
	Section     string
	Top         string // short section label, e.g. "TOP 3", "§ 12", "A", "II"
	Breadcrumb  string // hierarchical heading path for web documents
	URL         string // source URL (for web documents; empty for PDFs)
	paraIndices []int  // original paragraph indices (used to build parentText)
}

// pdfLine is one extracted text row together with its vertical position on
// the page. Y increases from bottom to top, so lines are ordered by
// descending Y.
type pdfLine struct {
	y    float64
	Text string
}

// SupportedDocumentExt reports whether ext (as returned by filepath.Ext,
// e.g. ".pdf") is a file type ExtractParagraphs can handle. Keep in sync with
// its switch below — this lets a caller (e.g. an upload handler) reject an
// unsupported file before it's written to disk, instead of only discovering
// the mismatch once ExtractParagraphs runs.
func SupportedDocumentExt(ext string) bool {
	switch strings.ToLower(ext) {
	case ".pdf", ".docx":
		return true
	default:
		return false
	}
}

// ExtractParagraphs reads path and returns its paragraphs, dispatching on
// file extension. ".pdf" and ".docx" are the only supported types — anything
// else is an explicit error rather than a silent misparse.
func ExtractParagraphs(path string) ([]Paragraph, error) {
	switch ext := strings.ToLower(filepath.Ext(path)); ext {
	case ".pdf":
		pages, err := readPDFPages(path)
		if err != nil {
			return nil, err
		}
		return parseIntoParagraphs(pages), nil
	case ".docx":
		paras, err := readDOCXParagraphs(path)
		if err != nil {
			return nil, err
		}
		return parseDOCXIntoParagraphs(paras), nil
	default:
		return nil, fmt.Errorf("unsupported file type %q (expected .pdf or .docx)", ext)
	}
}

// readPDFPages extracts text from each page of a PDF individually,
// preserving page boundaries for metadata. Lines are grouped into paragraphs
// using the vertical distance between them, running headers and footers are
// dropped, and words hyphenated across a line break are rejoined.
//
// Paragraph detection matters more than it looks: without it a whole page
// arrives as a single block, which BuildChunks cannot split, so one chunk
// ends up holding an entire page.
func readPDFPages(path string) ([]pageContent, error) {
	f, r, err := pdf.Open(path)
	if f != nil {
		defer f.Close()
	}
	if err != nil {
		return nil, err
	}

	// Pass 1: extract the lines of every page.
	pageNums := make([]int, 0, r.NumPage())
	pageLines := make([][]pdfLine, 0, r.NumPage())
	for i := 1; i <= r.NumPage(); i++ {
		p := r.Page(i)
		if p.V.IsNull() {
			continue
		}
		lines := extractPageLines(p)
		if len(lines) == 0 {
			continue
		}
		pageNums = append(pageNums, i)
		pageLines = append(pageLines, lines)
	}

	// Pass 2: drop headers and footers, which only become visible once all
	// pages are known.
	stripRunningLines(pageLines)

	var pages []pageContent
	for i, lines := range pageLines {
		text := strings.TrimSpace(dehyphenate(linesToText(lines)))
		if text != "" {
			pages = append(pages, pageContent{PageNum: pageNums[i], Text: text})
		}
	}
	return pages, nil
}

// lineTolerance is the vertical distance, in points, within which two text
// items are considered to sit on the same line.
const lineTolerance = 2.0

// extractPageLines groups a page's text items into lines by their Y position.
//
// This deliberately does not use pdf.Page.GetTextByRow: on the protocol PDFs
// in this project that method returns a single row holding a fraction of the
// page's text items, while the Y coordinates on Content().Text are intact
// (48 distinct lines on a page it reports as one row).
func extractPageLines(p pdf.Page) []pdfLine {
	texts := p.Content().Text
	if len(texts) == 0 {
		return nil
	}

	// Items keep their drawing order inside a line. Sorting a line by X looks
	// tempting but breaks whenever two text objects share a Y and the second
	// one's X is off — on the letterhead of these protocols that interleaves
	// "Technische Universität Ilmenau" with the date next to it, character by
	// character. Drawing order reproduces both runs intact.
	type placed struct {
		pdf.Text
		order int
	}
	items := make([]placed, 0, len(texts))
	for _, t := range texts {
		if t.S == "\n" || t.S == "\r" {
			continue // explicit line-break markers carry no glyph
		}
		items = append(items, placed{Text: t, order: len(items)})
	}
	sort.SliceStable(items, func(i, j int) bool { return items[i].Y > items[j].Y })

	var lines []pdfLine
	var current []placed
	var currentY float64

	flush := func() {
		if len(current) == 0 {
			return
		}
		sort.SliceStable(current, func(i, j int) bool { return current[i].order < current[j].order })
		texts := make([]pdf.Text, len(current))
		for i, c := range current {
			texts[i] = c.Text
		}
		if text := NormalizeSpaces(rowText(texts)); text != "" {
			lines = append(lines, pdfLine{y: currentY, Text: text})
		}
		current = nil
	}

	for _, t := range items {
		if len(current) > 0 && math.Abs(currentY-t.Y) > lineTolerance {
			flush()
		}
		if len(current) == 0 {
			currentY = t.Y
		}
		current = append(current, t)
	}
	flush()
	return lines
}

// rowText renders one line's items. PDFs that emit space glyphs only need the
// items concatenated; for those that position words without spaces,
// buildRowText derives the word boundaries from the gaps between items.
func rowText(items []pdf.Text) string {
	for _, t := range items {
		if t.S != "" && strings.TrimSpace(t.S) == "" {
			var sb strings.Builder
			for _, t := range items {
				sb.WriteString(t.S)
			}
			return sb.String()
		}
	}
	return buildRowText(items)
}

// linesToText joins lines into page text, separating them with a blank line
// wherever the vertical gap is noticeably larger than the page's usual line
// spacing. Those blank lines are what splitParagraphs later splits on.
func linesToText(lines []pdfLine) string {
	if len(lines) == 0 {
		return ""
	}

	// The median gap is the body line spacing: more than half the lines of a
	// page are consecutive lines within a paragraph.
	gaps := make([]float64, 0, len(lines)-1)
	for i := 1; i < len(lines); i++ {
		gap := lines[i-1].y - lines[i].y
		if gap > 0 {
			gaps = append(gaps, gap)
		}
	}
	var threshold float64
	if len(gaps) > 0 {
		sorted := make([]float64, len(gaps))
		copy(sorted, gaps)
		sort.Float64s(sorted)
		// Lower median: with only a couple of gaps on the page, taking the
		// upper one would raise the threshold above every gap there is.
		threshold = sorted[(len(sorted)-1)/2] * 1.4
	}

	var sb strings.Builder
	sb.WriteString(lines[0].Text)
	for i := 1; i < len(lines); i++ {
		gap := lines[i-1].y - lines[i].y
		if threshold > 0 && gap > threshold {
			sb.WriteString("\n\n")
		} else {
			sb.WriteString("\n")
		}
		sb.WriteString(lines[i].Text)
	}
	return sb.String()
}

// runningLineCandidates is how many lines at the top and at the bottom of a
// page are considered for header/footer detection.
const runningLineCandidates = 3

// stripRunningLines removes headers and footers from pageLines in place. A
// line near the top or bottom edge that repeats on most pages is boilerplate;
// so is a bare page number, which differs per page and therefore has to be
// recognised by shape rather than by repetition.
func stripRunningLines(pageLines [][]pdfLine) {
	if len(pageLines) < 3 {
		return // too few pages to tell a header from ordinary text
	}

	counts := make(map[string]int)
	for _, lines := range pageLines {
		seen := make(map[string]bool)
		for _, idx := range edgeIndices(len(lines)) {
			key := normalizeRunningLine(lines[idx].Text)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			counts[key]++
		}
	}

	// Require a clear majority so that a phrase which happens to appear at the
	// top of two pages is not mistaken for a header.
	minRepeats := len(pageLines) * 6 / 10
	if minRepeats < 2 {
		minRepeats = 2
	}

	for pi, lines := range pageLines {
		drop := make(map[int]bool)
		for _, idx := range edgeIndices(len(lines)) {
			text := lines[idx].Text
			if isPageNumberLine(text) || counts[normalizeRunningLine(text)] >= minRepeats {
				drop[idx] = true
			}
		}
		if len(drop) == 0 {
			continue
		}
		kept := make([]pdfLine, 0, len(lines))
		for i, l := range lines {
			if !drop[i] {
				kept = append(kept, l)
			}
		}
		pageLines[pi] = kept
	}
}

// edgeIndices returns the line indices at the top and bottom of a page that
// may hold a header or a footer. The count scales with the page so that a
// sparse page is not classified as nothing but edges — normalizeRunningLine
// drops digits, which makes body lines of a short page look alike.
func edgeIndices(n int) []int {
	count := n / 3
	if count > runningLineCandidates {
		count = runningLineCandidates
	}
	if count < 1 {
		return nil
	}

	var idx []int
	for i := 0; i < count; i++ {
		idx = append(idx, i)
	}
	for i := n - count; i < n; i++ {
		if i >= count {
			idx = append(idx, i)
		}
	}
	return idx
}

// normalizeRunningLine reduces a line to a form that compares equal across
// pages: lowercased, whitespace collapsed, digits dropped. Dropping digits is
// what lets "Seite 3 von 12" match "Seite 4 von 12".
func normalizeRunningLine(text string) string {
	var sb strings.Builder
	space := false
	for _, r := range strings.ToLower(text) {
		switch {
		case unicode.IsDigit(r):
			continue
		case unicode.IsSpace(r):
			space = true
		default:
			if space && sb.Len() > 0 {
				sb.WriteByte(' ')
			}
			space = false
			sb.WriteRune(r)
		}
	}
	return strings.TrimSpace(sb.String())
}

// isPageNumberLine detects bare page markers such as "3", "- 3 -" or
// "Seite 3 von 12", which repeat in position but never in text.
func isPageNumberLine(text string) bool {
	t := strings.TrimSpace(text)
	if t == "" || len(t) > 20 {
		return false
	}
	hasDigit := false
	for _, r := range t {
		switch {
		case unicode.IsDigit(r):
			hasDigit = true
		case unicode.IsLetter(r):
			// Only page-marker words are allowed alongside the digits.
			lower := strings.ToLower(t)
			if !strings.HasPrefix(lower, "seite") && !strings.HasPrefix(lower, "page") {
				return false
			}
		}
	}
	return hasDigit
}

// dehyphenate rejoins words split across a line break ("Ver-\nwaltung").
// Only a lowercase/lowercase pair is joined: an uppercase letter after the
// hyphen usually marks a real compound ("Bund-\nLänder-Kommission").
var hyphenBreak = regexp.MustCompile(`(\p{Ll})-\n(\p{Ll})`)

func dehyphenate(text string) string {
	return hyphenBreak.ReplaceAllString(text, "$1$2")
}

// NormalizeSpaces trims and collapses multiple spaces into one.
func NormalizeSpaces(s string) string {
	s = strings.TrimSpace(s)
	var sb strings.Builder
	inSpace := false
	for _, r := range s {
		if r == ' ' || r == '\t' {
			if !inSpace {
				sb.WriteByte(' ')
				inSpace = true
			}
		} else {
			sb.WriteRune(r)
			inSpace = false
		}
	}
	return sb.String()
}

// buildRowText assembles text fragments from a PDF row, inserting spaces
// only at actual word boundaries. It uses an adaptive approach: analyse
// all X-gaps in the row to distinguish intra-word character spacing from
// inter-word gaps.
func buildRowText(items []pdf.Text) string {
	if len(items) == 0 {
		return ""
	}
	if len(items) == 1 {
		return items[0].S
	}

	// Collect gaps between consecutive items.
	gaps := make([]float64, len(items)-1)
	for j := 1; j < len(items); j++ {
		gaps[j-1] = items[j].X - items[j-1].X
	}

	// Determine font size for fallback threshold.
	fontSize := items[0].FontSize
	if fontSize <= 0 {
		fontSize = 12
	}

	// Sort a copy to find min, median, max.
	sorted := make([]float64, len(gaps))
	copy(sorted, gaps)
	sort.Float64s(sorted)

	minGap := sorted[0]
	maxGap := sorted[len(sorted)-1]
	medianGap := sorted[len(sorted)/2]

	// Choose threshold based on gap distribution.
	var threshold float64
	if maxGap-minGap < fontSize*0.15 {
		// Gaps are uniform — either all intra-word or all inter-word.
		if medianGap > fontSize*0.5 {
			threshold = -1 // everything is a word gap → always insert space
		} else {
			threshold = maxGap + 1 // everything is intra-word → never insert space
		}
	} else {
		// Mixed gaps: adaptive threshold between the two clusters.
		threshold = medianGap * 1.3
	}

	var sb strings.Builder
	sb.WriteString(items[0].S)
	for j := 1; j < len(items); j++ {
		if gaps[j-1] > threshold {
			sb.WriteString(" ")
		}
		sb.WriteString(items[j].S)
	}
	return sb.String()
}

// parseIntoParagraphs splits page text into paragraphs annotated with
// page numbers and heading status.
func parseIntoParagraphs(pages []pageContent) []Paragraph {
	var paras []Paragraph
	for _, page := range pages {
		blocks := splitParagraphs(page.Text)
		for _, block := range blocks {
			trimmed := strings.TrimSpace(block)
			if trimmed == "" {
				continue
			}
			// Further split on section boundaries within a block. PDFs
			// often don't produce blank lines between a heading (TOP, §,
			// lettered/roman group) and its body, so both end up in the
			// same paragraph block. Splitting here ensures such headings
			// are always isolated.
			for _, sub := range splitOnSectionBoundaries(trimmed) {
				sub = strings.TrimSpace(sub)
				if sub == "" {
					continue
				}
				paras = append(paras, Paragraph{
					Text:    sub,
					PageNum: page.PageNum,
					Heading: isHeading(sub),
				})
			}
		}
	}
	return mergePageBreaks(paras)
}

// mergePageBreaks rejoins a paragraph that a page break cut in two. Each page
// is parsed on its own, so a sentence running across the break would otherwise
// end up as two paragraphs — and, once chunked, in two different chunks.
// The merged paragraph keeps the page number it started on.
func mergePageBreaks(paras []Paragraph) []Paragraph {
	merged := make([]Paragraph, 0, len(paras))
	for _, p := range paras {
		if len(merged) > 0 {
			prev := &merged[len(merged)-1]
			if prev.PageNum != p.PageNum && !prev.Heading && !p.Heading &&
				continuesParagraph(prev.Text, p.Text) {
				prev.Text = joinContinuation(prev.Text, p.Text)
				continue
			}
		}
		merged = append(merged, p)
	}
	return merged
}

// continuesParagraph reports whether next reads as the continuation of prev.
// The decisive signal is that prev stops without closing punctuation. Case is
// no help here: German capitalises every noun, so "… beauftragt das" /
// "Präsidium mit …" is an ordinary continuation. What does rule it out is next
// opening a new structure — a bullet, a numbered item or an agenda point.
func continuesParagraph(prev, next string) bool {
	prev = strings.TrimSpace(prev)
	next = strings.TrimSpace(next)
	if prev == "" || next == "" {
		return false
	}
	last, _ := utf8.DecodeLastRuneInString(prev)
	if strings.ContainsRune(".!?:;", last) {
		return false
	}
	if isProtocolTOP(next) || isNumberedSection(next) {
		return false
	}
	first, _ := utf8.DecodeRuneInString(next)
	return unicode.IsLetter(first) || first == '„' || first == '"' || first == '('
}

// joinContinuation appends next to prev, closing a word that the page break
// split across a hyphen.
func joinContinuation(prev, next string) string {
	prev = strings.TrimSpace(prev)
	next = strings.TrimSpace(next)
	if strings.HasSuffix(prev, "-") {
		return strings.TrimSuffix(prev, "-") + next
	}
	return prev + " " + next
}

// splitOnSectionBoundaries splits a text block at each line that starts a new
// addressable section — a protocol agenda item ("TOP N …", "N1. …"), a
// statute paragraph ("§ 12 …"), or a lettered/roman group heading ("A. …",
// "II. …"). This ensures such headings are isolated as separate paragraphs
// even when the PDF contains no blank line between the heading and its body
// text.
func splitOnSectionBoundaries(text string) []string {
	lines := strings.Split(text, "\n")
	var blocks []string
	var current []string

	flush := func() {
		if len(current) > 0 {
			blocks = append(blocks, strings.Join(current, "\n"))
			current = nil
		}
	}

	for _, line := range lines {
		if isSectionBoundary(strings.TrimSpace(line)) {
			// Flush whatever came before this heading, then isolate the
			// heading line itself as its own block — otherwise it stays
			// joined to every body line up to the next boundary, and
			// isHeading's own-line/own-block check (len(lines) > 2) rejects
			// the merged block outright. Without this, a heading with no
			// blank line before its body (the common case in these PDFs)
			// never gets flagged as a heading at all, silently merging an
			// entire agenda item's body into whatever section preceded it.
			flush()
			blocks = append(blocks, line)
			continue
		}
		current = append(current, line)
	}
	flush()
	return blocks
}

// splitParagraphs splits text on double newlines into blocks.
func splitParagraphs(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	raw := strings.Split(text, "\n\n")
	var result []string
	for _, block := range raw {
		trimmed := strings.TrimSpace(block)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

// isHeading uses heuristics to detect whether a text block is a heading.
func isHeading(text string) bool {
	lines := strings.Split(text, "\n")
	if len(lines) > 2 {
		return false
	}
	line := strings.TrimSpace(lines[0])
	if len(line) == 0 || len(line) > 120 {
		return false
	}
	// Protocol agenda item, statute paragraph, or lettered/roman group
	// heading: "TOP 1", "N1.", "§ 12 …", "A. …", "II. …"
	if isSectionBoundary(line) {
		return true
	}
	// ALL CAPS with at least 3 letter characters
	if len(line) >= 3 && line == strings.ToUpper(line) && containsLetter(line) {
		return true
	}
	// Numbered section pattern: "1. Title", "1.2 Title", "1.2.3 Title"
	if len(line) < 80 && isNumberedSection(line) {
		return true
	}
	// Short line ending with colon
	if len(line) < 60 && strings.HasSuffix(line, ":") {
		return true
	}
	return false
}

// isProtocolTOP detects agenda item headings in German meeting protocols:
// "TOP 1", "TOP 2:", "TOP 1 - Titel", "N1.", "N2." (nichtöffentlich).
func isProtocolTOP(line string) bool {
	trimmed := strings.TrimSpace(line)
	upper := strings.ToUpper(trimmed)

	// "TOP 1", "TOP 2:", "TOP 10 Titel", and the non-public "TOP N1".
	if strings.HasPrefix(upper, "TOP ") {
		rest := trimmed[4:] // original case kept — the title-shape check below needs it
		i := 0
		for i < len(rest) && rest[i] >= '0' && rest[i] <= '9' {
			i++
		}
		if i == 0 {
			if len(rest) > 1 && (rest[0] == 'N' || rest[0] == 'n') && unicode.IsDigit(rune(rest[1])) {
				i = 2
				for i < len(rest) && rest[i] >= '0' && rest[i] <= '9' {
					i++
				}
			} else {
				return false
			}
		}
		return topTitleLooksLikeHeading(rest[i:])
	}

	// "N1.", "N2." – nichtöffentliche Tagesordnungspunkte
	if len(upper) >= 3 && upper[0] == 'N' && unicode.IsDigit(rune(upper[1])) && upper[2] == '.' {
		return true
	}
	return false
}

// topTitleLooksLikeHeading reports whether rest — whatever follows the
// agenda number on a "TOP N" line — reads as a short heading title rather
// than prose that merely mentions "TOP N" mid-sentence. Two real examples
// from the corpus that must NOT count as a new heading: "TOP 8 bestätigt:
// Auf Seite 7, ..." (confirming a correction to a previous protocol) and
// "TOP 10b 13 von 19 ... anwesend" (a quorum count referencing a prior
// sub-item). A genuine title is either absent, a single-letter sub-item
// suffix ("TOP 10b" on its own), or starts with an uppercase word — German
// capitalizes nouns, so a title never starts lowercase, and a digit right
// after the number ("10b 13 ...") marks a number being read out, not a title.
func topTitleLooksLikeHeading(rest string) bool {
	i := 0
	if r, size := utf8.DecodeRuneInString(rest); unicode.IsLower(r) {
		// A lone trailing letter right after the digits ("TOP 10b") is a
		// sub-item suffix, not the start of a word — only treat it as one
		// when the rune after it isn't itself a letter.
		if next, _ := utf8.DecodeRuneInString(rest[size:]); !unicode.IsLetter(next) {
			i = size
		}
	}
	title := strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(rest[i:]), ":-–"))
	if title == "" {
		return true
	}
	first, _ := utf8.DecodeRuneInString(title)
	return unicode.IsUpper(first)
}

// isParagraphSection detects statute/regulation paragraph headings such as
// "§ 1 Geltungsbereich" or "§ 12 Modulabschluss".
//
// A cross-reference to a subsection of some law ("§ 54 Absatz 6 ThürHG
// anwesend sein.") reads exactly like a heading once a PDF wraps it onto its
// own line, but it's prose, not a heading — it names an "Absatz"/"Abs." right
// after the number and ends the sentence with a period, which no actual
// heading in the observed corpus does. Both are excluded.
func isParagraphSection(line string) bool {
	trimmed := strings.TrimSpace(line)
	rest, ok := strings.CutPrefix(trimmed, "§")
	if !ok {
		return false
	}
	rest = strings.TrimSpace(rest)
	i := 0
	for i < len(rest) && unicode.IsDigit(rune(rest[i])) {
		i++
	}
	if i == 0 {
		return false
	}
	afterNumber := strings.TrimSpace(rest[i:])
	if strings.HasPrefix(afterNumber, "Absatz") || strings.HasPrefix(afterNumber, "Abs.") {
		return false
	}
	return !strings.HasSuffix(trimmed, ".")
}

// isLetterSection detects top-level "A. Titel", "B. Titel" group headings
// that some statutes use to group several §§ together.
func isLetterSection(line string) bool {
	return len(line) >= 3 && line[0] >= 'A' && line[0] <= 'Z' && line[1] == '.' && line[2] == ' '
}

// romanSectionRe matches a roman-numeral group heading like "I. " or
// "III. " at the start of a line, e.g. "II. Modulabschluss – Art, Zulassung".
var romanSectionRe = regexp.MustCompile(`^[IVXLCDM]{1,6}\.\s`)

// isRomanSection detects roman-numeral group headings such as "I. Titel" or
// "II. Titel", used as a sub-level between lettered sections and §§.
func isRomanSection(line string) bool {
	return romanSectionRe.MatchString(line)
}

// isSectionBoundary reports whether line starts a new addressable section:
// a protocol agenda item, a statute paragraph, or a lettered/roman group
// heading. Lines matching this are split out mid-paragraph-block, because a
// PDF often doesn't leave a blank line between such a heading and the text
// that follows it — see splitOnSectionBoundaries.
func isSectionBoundary(line string) bool {
	return isProtocolTOP(line) || isParagraphSection(line) || isLetterSection(line) || isRomanSection(line)
}

// extractSectionLabel returns the short, self-contained citation label for a
// section heading, ready to print as-is: "TOP 3" and "TOP N1" for protocol
// agenda items, "§ 12" for statute paragraphs, "A" for a lettered group
// heading, "II" for a roman-numeral one. Returns "" for anything else (plain
// numbered sections and prose headings fall back to the full heading text).
func extractSectionLabel(section string) string {
	trimmed := strings.TrimSpace(section)

	upper := strings.ToUpper(trimmed)
	if strings.HasPrefix(upper, "TOP ") {
		rest := strings.TrimSpace(trimmed[4:])
		// "TOP N1" is a non-public agenda item and numbers independently.
		prefix := ""
		if len(rest) > 1 && (rest[0] == 'N' || rest[0] == 'n') && unicode.IsDigit(rune(rest[1])) {
			prefix, rest = "N", rest[1:]
		}
		i := 0
		for i < len(rest) && unicode.IsDigit(rune(rest[i])) {
			i++
		}
		if i > 0 {
			return "TOP " + prefix + rest[:i]
		}
	}
	// "N1.", "N2." → "TOP N1", "TOP N2"
	if len(trimmed) >= 2 && (trimmed[0] == 'N' || trimmed[0] == 'n') && unicode.IsDigit(rune(trimmed[1])) {
		i := 1
		for i < len(trimmed) && unicode.IsDigit(rune(trimmed[i])) {
			i++
		}
		return "TOP " + strings.ToUpper(trimmed[:i])
	}

	if rest, ok := strings.CutPrefix(trimmed, "§"); ok && isParagraphSection(trimmed) {
		rest = strings.TrimSpace(rest)
		i := 0
		for i < len(rest) && unicode.IsDigit(rune(rest[i])) {
			i++
		}
		if i > 0 {
			return "§ " + rest[:i]
		}
	}

	if isLetterSection(trimmed) {
		return string(trimmed[0])
	}

	if m := romanSectionRe.FindString(trimmed); m != "" {
		return strings.TrimRight(m, ". ")
	}

	return ""
}

func containsLetter(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

// isNumberedSection detects patterns like "1. Title" or "1.2.3 Title".
func isNumberedSection(line string) bool {
	if len(line) == 0 {
		return false
	}
	i := 0
	if i >= len(line) || !unicode.IsDigit(rune(line[i])) {
		return false
	}
	for i < len(line) && unicode.IsDigit(rune(line[i])) {
		i++
	}
	sawDot := false
	for i < len(line) && line[i] == '.' {
		sawDot = true
		i++
		for i < len(line) && unicode.IsDigit(rune(line[i])) {
			i++
		}
	}
	if !sawDot || i >= len(line) || line[i] != ' ' {
		return false
	}
	return true
}

// BuildChunks groups paragraphs into chunks that respect paragraph boundaries,
// section breaks, and a maximum chunk size. Each chunk carries page range,
// section metadata, and an expanded ParentText (~parentSize chars) built from
// surrounding paragraphs for richer LLM context.
// overlapSize controls how many characters from the end of one chunk are
// repeated at the start of the next (reset at section boundaries).
func BuildChunks(paras []Paragraph, maxSize, overlapSize int) []Chunk {
	// A paragraph is never split once chunking starts, so an oversized one
	// would become an oversized chunk — past the embedding model's context
	// window, where the tail is silently dropped. The size guard in the main
	// loop below only fires once a chunk already has content in it (currentLen
	// > 0), so it can't stop the first paragraph after a heading/section
	// break from being added on its own — by then flush() has already reset
	// currentLen to 0, and there's nothing to flush that paragraph out from
	// under. Reserve room for the longest heading/breadcrumb prefix this
	// document can produce up front instead, so that paragraph is never
	// itself larger than maxSize minus that prefix.
	maxHeadingPrefix := 0
	for _, p := range paras {
		if !p.Heading {
			continue
		}
		if l := len(p.Text) + 2; l > maxHeadingPrefix { // "\n\n"
			maxHeadingPrefix = l
		}
		if p.Breadcrumb != "" {
			if l := len(p.Breadcrumb) + 4; l > maxHeadingPrefix { // "[" + "]" + "\n\n"
				maxHeadingPrefix = l
			}
		}
	}
	splitTarget := maxSize - maxHeadingPrefix
	if splitTarget < maxSize/2 {
		// An implausibly long heading/breadcrumb shouldn't be allowed to chop
		// paragraphs down to a sliver — cap how much room gets reserved for it.
		splitTarget = maxSize / 2
	}
	paras = splitOversizedParagraphs(paras, splitTarget)

	// Filter out heading paragraphs and record section boundaries.
	// We keep headings separate so we can split on them.
	type indexedPara struct {
		Paragraph
		idx int // original index in paras
	}

	var chunks []Chunk
	var currentParas []indexedPara
	currentSection := ""
	currentTop := ""
	currentBreadcrumb := ""
	currentLen := 0
	// addedSinceFlush tracks whether a paragraph has joined currentParas since
	// the last flush, so that carried-over overlap is never flushed on its own.
	addedSinceFlush := false

	// headingPrefixLen is the room flush() needs for the section heading or
	// breadcrumb it prepends to the chunk text. Counting it here is what makes
	// maxSize the size of the finished chunk rather than of its body alone.
	headingPrefixLen := func() int {
		if currentBreadcrumb != "" {
			return len(currentBreadcrumb) + 4 // "[" + "]" + "\n\n"
		}
		if currentSection != "" {
			return len(currentSection) + 2 // "\n\n"
		}
		return 0
	}

	parasText := func(ps []indexedPara) string {
		var b strings.Builder
		for i, p := range ps {
			if i > 0 {
				b.WriteString("\n\n")
			}
			b.WriteString(p.Text)
		}
		return b.String()
	}

	pageRange := func(ps []indexedPara) string {
		if len(ps) == 0 || ps[0].PageNum <= 0 {
			// PageNum <= 0 means "no page info" (e.g. a .docx chunk, which
			// has no fixed pagination) rather than an actual page 0.
			return ""
		}
		start := ps[0].PageNum
		end := ps[len(ps)-1].PageNum
		if end > start {
			return fmt.Sprintf("%d-%d", start, end)
		}
		return fmt.Sprintf("%d", start)
	}

	// overlapParas returns the shortest suffix of ps whose combined text
	// length is >= overlapSize. Returns nil when the shortest qualifying
	// suffix would exceed overlapSize*2 (e.g. a single huge paragraph),
	// to avoid carrying oversized overlap that bloats subsequent chunks.
	overlapParas := func(ps []indexedPara) []indexedPara {
		total := 0
		for i := len(ps) - 1; i >= 0; i-- {
			total += len(ps[i].Text)
			if i < len(ps)-1 {
				total += 2 // "\n\n"
			}
			if total >= overlapSize {
				if total > overlapSize*2 {
					return nil // paragraph too large for useful overlap
				}
				return ps[i:]
			}
		}
		if total > overlapSize*2 {
			return nil
		}
		return ps
	}

	flush := func(sectionBreak bool) {
		if len(currentParas) == 0 {
			return
		}
		bodyText := strings.TrimSpace(parasText(currentParas))
		if bodyText == "" {
			currentParas = currentParas[:0]
			currentLen = 0
			return
		}

		// Prepend section heading/breadcrumb to the chunk text so the
		// embedding captures the topic context — not just the body text.
		var textBuf strings.Builder
		if currentBreadcrumb != "" {
			textBuf.WriteString("[")
			textBuf.WriteString(currentBreadcrumb)
			textBuf.WriteString("]\n\n")
		} else if currentSection != "" {
			textBuf.WriteString(currentSection)
			textBuf.WriteString("\n\n")
		}
		textBuf.WriteString(bodyText)
		text := textBuf.String()

		indices := make([]int, len(currentParas))
		for i, p := range currentParas {
			indices[i] = p.idx
		}
		chunks = append(chunks, Chunk{
			Text:        text,
			Pages:       pageRange(currentParas),
			Section:     currentSection,
			Top:         currentTop,
			Breadcrumb:  currentBreadcrumb,
			paraIndices: indices,
		})

		if sectionBreak {
			currentParas = currentParas[:0]
		} else {
			currentParas = append([]indexedPara{}, overlapParas(currentParas)...)
		}
		currentLen = len(parasText(currentParas))
		addedSinceFlush = false
	}

	for i, para := range paras {
		if para.Heading {
			flush(true) // section boundary: no overlap
			currentSection = para.Text
			currentTop = extractSectionLabel(para.Text)
			if para.Breadcrumb != "" {
				currentBreadcrumb = para.Breadcrumb
			}
			continue
		}

		addition := len(para.Text)
		if currentLen > 0 {
			addition += 2
		}
		if currentLen > 0 && currentLen+headingPrefixLen()+addition > maxSize {
			if addedSinceFlush {
				flush(false) // size boundary: carry overlap
			} else {
				// Nothing but carried-over overlap is pending, so flushing
				// again would emit a chunk that only repeats the tail of the
				// previous one.
				currentParas = currentParas[:0]
				currentLen = 0
			}
			// flush carries overlap into the next chunk, and this paragraph is
			// then appended to it unconditionally. Drop that overlap when the
			// two together would not fit, so a chunk stays within maxSize —
			// which is what keeps it inside the embedder's context window.
			if currentLen > 0 && currentLen+headingPrefixLen()+2+len(para.Text) > maxSize {
				currentParas = currentParas[:0]
				currentLen = 0
			}
		}

		if currentLen > 0 {
			currentLen += 2 // "\n\n" separator
		}
		currentParas = append(currentParas, indexedPara{Paragraph: para, idx: i})
		currentLen += len(para.Text)
		addedSinceFlush = true
	}
	flush(true)

	// Build ParentText for each chunk from the surrounding paragraphs of the
	// same section. Headings act as hard boundaries — ParentText never crosses
	// into a different section's content — so a section that fits within
	// parentSize is carried whole, which keeps short protocol TOPs intact.
	// Longer sections are windowed around the chunk instead of copied in full:
	// an uncapped ParentText grows to the size of the section (10k+ chars on
	// web pages), and ten of those overflow the LLM context in the QA flow.
	parentSize := maxSize * 2
	if parentSize < 3000 {
		parentSize = 3000
	}

	// Build sections: each section is a heading followed by its body paragraphs.
	type bodyPara struct {
		Text    string
		idx     int // original index in paras
		PageNum int
	}
	type section struct {
		Heading    string     // heading text (empty for paragraphs before first heading)
		Breadcrumb string     // breadcrumb path for web docs
		bodyParas  []bodyPara // non-heading paragraphs in this section
	}

	var sections []section
	cur := section{}
	for i, p := range paras {
		if p.Heading {
			if len(cur.bodyParas) > 0 || cur.Heading != "" {
				sections = append(sections, cur)
			}
			cur = section{Heading: p.Text, Breadcrumb: p.Breadcrumb}
		} else {
			cur.bodyParas = append(cur.bodyParas, bodyPara{Text: p.Text, idx: i, PageNum: p.PageNum})
		}
	}
	if len(cur.bodyParas) > 0 || cur.Heading != "" {
		sections = append(sections, cur)
	}

	// Build a lookup: original paragraph index → (section index, position within section).
	type bodyPos struct {
		secIdx  int
		bodyIdx int
	}
	idxToPos := make(map[int]bodyPos)
	for si, sec := range sections {
		for bi, bp := range sec.bodyParas {
			idxToPos[bp.idx] = bodyPos{secIdx: si, bodyIdx: bi}
		}
	}

	for ci := range chunks {
		c := &chunks[ci]
		if len(c.paraIndices) == 0 {
			c.ParentText = c.Text
			continue
		}

		firstPos, ok1 := idxToPos[c.paraIndices[0]]
		lastPos, ok2 := idxToPos[c.paraIndices[len(c.paraIndices)-1]]
		if !ok1 || !ok2 || firstPos.secIdx != lastPos.secIdx {
			c.ParentText = c.Text
			continue
		}

		sec := sections[firstPos.secIdx]
		body := sec.bodyParas

		// Start at the chunk's own paragraphs and grow outwards, alternating
		// sides, for as long as the next paragraph still fits in parentSize.
		// A chunk that already exceeds parentSize keeps its own range.
		lo, hi := firstPos.bodyIdx, lastPos.bodyIdx
		size := 0
		for i := lo; i <= hi; i++ {
			if i > lo {
				size += 2 // "\n\n"
			}
			size += len(body[i].Text)
		}
		if sec.Heading != "" {
			size += len(sec.Heading) + 2
		}
		for {
			grew := false
			if lo > 0 && size+len(body[lo-1].Text)+2 <= parentSize {
				lo--
				size += len(body[lo].Text) + 2
				grew = true
			}
			if hi < len(body)-1 && size+len(body[hi+1].Text)+2 <= parentSize {
				hi++
				size += len(body[hi].Text) + 2
				grew = true
			}
			if !grew {
				break
			}
		}

		// c.Pages so far only covers the chunk's own narrow paraIndices, but
		// ParentText (built below from [lo, hi]) is what the LLM actually
		// reads and cites from — on a large document the window can grow well
		// beyond the chunk's own pages, so the citation must reflect the
		// window's page range instead, or it reports pages the answer wasn't
		// even drawn from.
		if body[lo].PageNum > 0 {
			start, end := body[lo].PageNum, body[hi].PageNum
			if end > start {
				c.Pages = fmt.Sprintf("%d-%d", start, end)
			} else {
				c.Pages = fmt.Sprintf("%d", start)
			}
		}

		var b strings.Builder
		// Include section heading in ParentText.
		if c.Breadcrumb != "" {
			b.WriteString("[")
			b.WriteString(c.Breadcrumb)
			b.WriteString("]\n\n")
		} else if sec.Heading != "" {
			b.WriteString(sec.Heading)
			b.WriteString("\n\n")
		}
		for i := lo; i <= hi; i++ {
			if i > lo {
				b.WriteString("\n\n")
			}
			b.WriteString(body[i].Text)
		}
		c.ParentText = b.String()
	}

	return chunks
}

// splitOversizedParagraphs breaks paragraphs longer than maxSize into several
// paragraphs, preferring sentence boundaries. Headings are left alone.
func splitOversizedParagraphs(paras []Paragraph, maxSize int) []Paragraph {
	out := make([]Paragraph, 0, len(paras))
	for _, p := range paras {
		if p.Heading || len(p.Text) <= maxSize {
			out = append(out, p)
			continue
		}
		for _, part := range splitTextAtSentences(p.Text, maxSize) {
			q := p
			q.Text = part
			out = append(out, q)
		}
	}
	return out
}

// isSentenceEnd reports whether b closes a sentence.
func isSentenceEnd(b byte) bool {
	return b == '.' || b == '!' || b == '?' || b == ':' || b == ';'
}

// splitTextAtSentences cuts text into parts of at most maxSize bytes. Cuts are
// placed at the last sentence end within reach, else at the last line break or
// space, else — for text without any of those — at the next rune boundary.
func splitTextAtSentences(text string, maxSize int) []string {
	if maxSize <= 0 || len(text) <= maxSize {
		return []string{text}
	}

	var parts []string
	start := 0
	boundary, space := -1, -1

	appendPart := func(end int) {
		if part := strings.TrimSpace(text[start:end]); part != "" {
			parts = append(parts, part)
		}
		start = end
		boundary, space = -1, -1
	}

	for i := 0; i < len(text); i++ {
		switch c := text[i]; {
		case c == '\n':
			boundary, space = i+1, i+1
		case c == ' ':
			space = i + 1
			if i > start && isSentenceEnd(text[i-1]) {
				boundary = i + 1
			}
		}

		if i-start+1 < maxSize {
			continue
		}
		cut := boundary
		if cut <= start {
			cut = space
		}
		if cut <= start {
			// No boundary in this stretch at all — cut where we stand, but
			// never inside a multi-byte rune.
			cut = i + 1
			for cut < len(text) && !utf8.RuneStart(text[cut]) {
				cut++
			}
		}
		appendPart(cut)
		i = cut - 1
	}

	if start < len(text) {
		appendPart(len(text))
	}
	return parts
}
