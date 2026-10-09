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
	"io"
	"log"
	"net/http"
	"strings"
	"unicode"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

// FetchWebPage downloads the HTML content of a URL. It refuses targets that
// resolve to private, loopback or link-local addresses (see
// SetAllowPrivateTargets), follows at most 10 redirects, and reads at most
// SetMaxPageBytes bytes (10 MiB by default).
func FetchWebPage(url string) (string, error) {
	client := newFetchClient()
	defer client.CloseIdleConnections()
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}
	req.Header.Set("User-Agent", "askyourdocs/1.0 (web indexer)")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}

	limit := maxPageBytes.Load()
	body, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return "", fmt.Errorf("reading response: %w", err)
	}
	if int64(len(body)) > limit {
		return "", fmt.Errorf("page %s exceeds %d bytes", url, limit)
	}
	log.Printf("URL '%v' processed.\n", url)
	return string(body), nil
}

// skipTags are HTML elements whose content should be completely ignored.
var skipTags = map[string]bool{
	"script":   true,
	"style":    true,
	"nav":      true,
	"footer":   true,
	"aside":    true,
	"noscript": true,
}

// minContentFormChars is how much visible text a <form> needs to count as
// content rather than a search/login widget. Some sites (JSF/HISinOne portals,
// ASP.NET WebForms) wrap the entire page in one <form>, so skipping every form
// would drop the page.
const minContentFormChars = 300

// boilerplatePatterns are substrings in class/id attributes that indicate boilerplate.
// "header" is intentionally omitted — isSiteHeader() handles <header> elements,
// and class names containing "header" (e.g. "article-header", "section-header")
// often belong to real content areas.
var boilerplatePatterns = []string{
	"sidebar", "menu", "cookie", "banner", "popup", "modal",
	"advertisement", "footer",
	"site-header", "page-header", "global-header", "top-header",
}

// boilerplateTokens are short words that only count as boilerplate when they
// are a whole word in the class/id ("main-nav", "topNav", "ad_slot") — as a
// substring they hit unrelated names such as "content_navi_off" or "road-map".
var boilerplateTokens = map[string]bool{
	"nav": true, "navbar": true, "navigation": true, "ad": true, "ads": true,
}

// classTokens splits a class/id value into lowercase words at any
// non-alphanumeric character and at camelCase boundaries.
func classTokens(val string) []string {
	var toks []string
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			toks = append(toks, strings.ToLower(string(cur)))
			cur = cur[:0]
		}
	}
	var prev rune
	for _, r := range val {
		switch {
		case !unicode.IsLetter(r) && !unicode.IsDigit(r):
			flush()
		case unicode.IsUpper(r) && unicode.IsLower(prev):
			flush()
			cur = append(cur, r)
		default:
			cur = append(cur, r)
		}
		prev = r
	}
	flush()
	return toks
}

// skipElement reports whether n and its subtree are dropped as boilerplate.
func skipElement(n *html.Node) bool {
	if n.Type != html.ElementNode {
		return false
	}
	if n.Data == "form" {
		return !isContentForm(n) || isBoilerplate(n)
	}
	return skipTags[n.Data] || isBoilerplate(n) || isSiteHeader(n)
}

// isContentForm reports whether a <form> holds enough visible text to be page
// content. Text of options, scripts and textareas does not count, so a form
// with a long <select> stays a widget.
func isContentForm(n *html.Node) bool {
	total := 0
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.Data {
			case "script", "style", "option", "textarea", "noscript":
				return
			}
		}
		if n.Type == html.TextNode {
			total += len(strings.TrimSpace(n.Data))
		}
		for c := n.FirstChild; c != nil && total < minContentFormChars; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return total >= minContentFormChars
}

// headingLevel returns the heading level (1-6) for h1-h6 tags, or 0 for non-headings.
func headingLevel(tagName string) int {
	switch tagName {
	case "h1":
		return 1
	case "h2":
		return 2
	case "h3":
		return 3
	case "h4":
		return 4
	case "h5":
		return 5
	case "h6":
		return 6
	}
	return 0
}

// isBoilerplate checks if a node's class or id contains boilerplate patterns.
func isBoilerplate(n *html.Node) bool {
	if n.Type != html.ElementNode {
		return false
	}
	for _, attr := range n.Attr {
		if attr.Key == "class" || attr.Key == "id" {
			lower := strings.ToLower(attr.Val)
			for _, pat := range boilerplatePatterns {
				if strings.Contains(lower, pat) {
					return true
				}
			}
			for _, tok := range classTokens(attr.Val) {
				if boilerplateTokens[tok] {
					return true
				}
			}
		}
	}
	return false
}

// isSiteHeader returns true for <header> elements that are top-level site
// headers (not inside <main> or <article>). Headers inside content areas
// often contain page titles and should be kept.
func isSiteHeader(n *html.Node) bool {
	if n.Type != html.ElementNode || n.Data != "header" {
		return false
	}
	for p := n.Parent; p != nil; p = p.Parent {
		if p.Type == html.ElementNode && (p.DataAtom == atom.Main || p.DataAtom == atom.Article) {
			return false // inside content area — keep it
		}
	}
	return true // top-level site header — skip it
}

// findMainContent locates the best content root: <main>, largest <article>, or <body>.
func findMainContent(doc *html.Node) *html.Node {
	var body *html.Node
	var mainNode *html.Node
	var bestArticle *html.Node
	var bestArticleLen int

	var find func(*html.Node)
	find = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.DataAtom {
			case atom.Body:
				body = n
			case atom.Main:
				mainNode = n
			case atom.Article:
				textLen := nodeTextLen(n)
				if textLen > bestArticleLen {
					bestArticle = n
					bestArticleLen = textLen
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			find(c)
		}
	}
	find(doc)

	if mainNode != nil {
		return mainNode
	}
	if bestArticle != nil {
		return bestArticle
	}
	if body != nil {
		return body
	}
	return doc
}

// nodeTextLen returns the approximate text length under a node.
func nodeTextLen(n *html.Node) int {
	total := 0
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			total += len(strings.TrimSpace(n.Data))
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return total
}

// headingStack tracks the current breadcrumb path through heading levels.
type headingStack struct {
	entries []headingEntry
}

type headingEntry struct {
	level int
	text  string
}

// push adds a heading. It pops any entries at the same or deeper level first.
func (hs *headingStack) push(level int, text string) {
	// Remove entries at same or deeper level
	for len(hs.entries) > 0 && hs.entries[len(hs.entries)-1].level >= level {
		hs.entries = hs.entries[:len(hs.entries)-1]
	}
	hs.entries = append(hs.entries, headingEntry{level: level, text: text})
}

// breadcrumb returns the current path as "A > B > C".
func (hs *headingStack) breadcrumb() string {
	if len(hs.entries) == 0 {
		return ""
	}
	parts := make([]string, len(hs.entries))
	for i, e := range hs.entries {
		parts[i] = e.text
	}
	return strings.Join(parts, " > ")
}

// HTMLToStructuredText parses HTML and returns structured paragraphs with
// headings detected from HTML tags, links preserved, tables formatted,
// and lists properly rendered. Replaces htmlToText + parseWebIntoParagraphs.
func HTMLToStructuredText(r io.Reader) ([]Paragraph, error) {
	doc, err := html.Parse(r)
	if err != nil {
		return nil, fmt.Errorf("parsing HTML: %w", err)
	}

	root := findMainContent(doc)
	hs := &headingStack{}
	var paras []Paragraph

	var walk, walkChildren func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			// Skip boilerplate elements
			if skipElement(n) {
				return
			}

			// Handle headings
			if level := headingLevel(n.Data); level > 0 {
				text := strings.TrimSpace(extractPlainText(n))
				if text != "" {
					hs.push(level, text)
					paras = append(paras, Paragraph{
						Text:       text,
						Heading:    true,
						Breadcrumb: hs.breadcrumb(),
					})
				}
				return
			}

			// Handle definition lists (dl > dt/dd)
			if n.Data == "dl" {
				dlText := renderDefinitionList(n)
				if strings.TrimSpace(dlText) != "" {
					paras = append(paras, Paragraph{
						Text:       strings.TrimSpace(dlText),
						Breadcrumb: hs.breadcrumb(),
					})
				}
				return
			}

			// Handle tables
			if n.Data == "table" {
				tableText := renderTable(n)
				if tableText != "" {
					tableParagraphs := splitTableIfLarge(tableText, 1200)
					bc := hs.breadcrumb()
					for _, tp := range tableParagraphs {
						paras = append(paras, Paragraph{
							Text:       tp,
							Breadcrumb: bc,
						})
					}
				}
				return
			}

			// Handle lists
			if n.Data == "ul" || n.Data == "ol" {
				listText := renderList(n, 0)
				if strings.TrimSpace(listText) != "" {
					paras = append(paras, Paragraph{
						Text:       strings.TrimSpace(listText),
						Breadcrumb: hs.breadcrumb(),
					})
				}
				return
			}

			// Handle block elements that produce paragraphs. A block may mix
			// inline text with nested blocks (<div>Lead<p>Body</p></div>);
			// rendering only the inline part and returning would drop the
			// nested blocks, so those get walked separately.
			if isBlockElement(n.Data) && !hasStructuralDescendant(n) {
				text := strings.TrimSpace(renderInlineContent(n))
				if text != "" {
					paras = append(paras, Paragraph{
						Text:       text,
						Breadcrumb: hs.breadcrumb(),
					})
				}
				return
			}
		}

		walkChildren(n)
	}

	// walkChildren walks n's children in document order, emitting runs of
	// inline siblings as paragraphs and descending into structural nodes.
	walkChildren = func(n *html.Node) {
		var pending []*html.Node

		flushInline := func() {
			if len(pending) == 0 {
				return
			}
			text := strings.TrimSpace(renderInlineNodes(pending))
			pending = nil
			if text != "" {
				paras = append(paras, Paragraph{
					Text:       text,
					Breadcrumb: hs.breadcrumb(),
				})
			}
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			// Boilerplate is dropped here too — walkChildren bypasses walk()
			// for non-structural containers, so its skip check would not run.
			if skipElement(c) {
				continue
			}
			switch {
			case isStructuralNode(c):
				flushInline()
				walk(c)
			case c.Type == html.ElementNode && hasStructuralDescendant(c):
				flushInline()
				walkChildren(c)
			default:
				pending = append(pending, c)
			}
		}
		flushInline()
	}

	walk(root)

	paras = filterNoiseParagraphs(paras)

	return paras, nil
}

// isBlockElement returns true for HTML elements that should produce separate paragraphs.
var blockElements = map[string]bool{
	"p": true, "div": true, "article": true, "section": true,
	"blockquote": true, "pre": true, "figcaption": true, "figure": true,
	"main": true, "details": true, "summary": true,
	"address": true, "header": true, "hgroup": true,
}

func isBlockElement(tag string) bool {
	return blockElements[tag]
}

// isStructuralNode reports whether a node is handled as its own unit by the
// walker in HTMLToStructuredText — a block, a heading, a table or a list.
// Inline content around such a node has to be flushed before walking into it.
func isStructuralNode(n *html.Node) bool {
	if n.Type != html.ElementNode {
		return false
	}
	return isBlockElement(n.Data) || headingLevel(n.Data) > 0 ||
		n.Data == "table" || n.Data == "ul" || n.Data == "ol" || n.Data == "dl"
}

// hasStructuralDescendant reports whether n contains a structural node at any
// depth. A block element without one can be rendered as a single paragraph;
// one with structural children has to be walked, or its nested content is lost.
func hasStructuralDescendant(n *html.Node) bool {
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if isStructuralNode(c) || hasStructuralDescendant(c) {
			return true
		}
	}
	return false
}

// extractPlainText extracts all text content from a node, ignoring structure.
func extractPlainText(n *html.Node) string {
	var sb strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			sb.WriteString(n.Data)
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return sb.String()
}

// extractLinkText extracts text from an <a> element, skipping copyright/credit
// spans that label images and should not be used as link labels.
func extractLinkText(n *html.Node) string {
	var sb strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			cls := strings.ToLower(getAttr(n, "class"))
			if n.Data == "span" && (strings.Contains(cls, "copyright") || strings.Contains(cls, "credit")) {
				return
			}
		}
		if n.Type == html.TextNode {
			sb.WriteString(n.Data)
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return sb.String()
}

// renderInlineContent renders the inline content of a block element,
// preserving link URLs and handling nested inline elements.
func renderInlineContent(n *html.Node) string {
	var kids []*html.Node
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		kids = append(kids, c)
	}
	return renderInlineNodes(kids)
}

// renderInlineNodes renders a run of sibling nodes as inline text. Nested
// block-level nodes are skipped — the caller is responsible for walking them.
func renderInlineNodes(nodes []*html.Node) string {
	var sb strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			text := collapseWhitespace(n.Data)
			sb.WriteString(text)
			return
		}
		if n.Type == html.ElementNode {
			if skipTags[n.Data] || (n.Data == "form" && !isContentForm(n)) {
				return
			}

			// Handle nested block elements — don't recurse into them here
			if isBlockElement(n.Data) || headingLevel(n.Data) > 0 ||
				n.Data == "table" || n.Data == "ul" || n.Data == "ol" {
				return
			}

			// Links: render link text only — URLs add noise to embeddings/keyword search
			if n.Data == "a" {
				linkText := strings.TrimSpace(extractLinkText(n))
				if linkText != "" {
					sb.WriteString(linkText)
				}
				return
			}

			// Line break
			if n.Data == "br" {
				sb.WriteString("\n")
				return
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}

	for _, n := range nodes {
		walk(n)
	}
	return NormalizeSpaces(sb.String())
}

// renderTable converts an HTML table into pipe-formatted text.
func renderTable(tableNode *html.Node) string {
	var rows [][]string

	var walkTable func(*html.Node)
	walkTable = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "tr" {
			var cells []string
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				if c.Type == html.ElementNode && (c.Data == "td" || c.Data == "th") {
					cellText := strings.TrimSpace(renderInlineContent(c))
					cells = append(cells, cellText)
				}
			}
			if len(cells) > 0 {
				rows = append(rows, cells)
			}
			return
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walkTable(c)
		}
	}
	walkTable(tableNode)

	if len(rows) == 0 {
		return ""
	}

	// Determine max columns
	maxCols := 0
	for _, row := range rows {
		if len(row) > maxCols {
			maxCols = len(row)
		}
	}

	// Normalize rows to same number of columns
	for i := range rows {
		for len(rows[i]) < maxCols {
			rows[i] = append(rows[i], "")
		}
	}

	// Build pipe-formatted table
	var sb strings.Builder
	for i, row := range rows {
		sb.WriteString("| ")
		sb.WriteString(strings.Join(row, " | "))
		sb.WriteString(" |")
		if i < len(rows)-1 {
			sb.WriteString("\n")
		}
		// Add separator after first row (header)
		if i == 0 {
			var sep []string
			for range row {
				sep = append(sep, "---")
			}
			sb.WriteString("| ")
			sb.WriteString(strings.Join(sep, " | "))
			sb.WriteString(" |\n")
		}
	}
	return sb.String()
}

// splitTableIfLarge splits a large table text into smaller parts,
// repeating the header line for each part.
func splitTableIfLarge(tableText string, maxSize int) []string {
	if len(tableText) <= maxSize {
		return []string{tableText}
	}

	lines := strings.Split(tableText, "\n")
	if len(lines) < 3 {
		return []string{tableText}
	}

	// First two lines are header + separator
	header := lines[0] + "\n" + lines[1]
	dataLines := lines[2:]

	var parts []string
	var current strings.Builder
	current.WriteString(header)

	for _, line := range dataLines {
		if current.Len()+len(line)+1 > maxSize && current.Len() > len(header) {
			parts = append(parts, current.String())
			current.Reset()
			current.WriteString(header)
		}
		current.WriteString("\n")
		current.WriteString(line)
	}
	if current.Len() > len(header) {
		parts = append(parts, current.String())
	}

	return parts
}

// renderList renders a ul or ol into formatted text with proper indentation.
func renderList(n *html.Node, depth int) string {
	if n.Type != html.ElementNode || (n.Data != "ul" && n.Data != "ol") {
		return ""
	}

	ordered := n.Data == "ol"
	indent := strings.Repeat("  ", depth)
	var sb strings.Builder
	itemNum := 0

	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if c.Type == html.ElementNode && c.Data == "li" {
			itemNum++

			// Collect inline text and nested lists separately
			var inlineText strings.Builder
			var nestedLists []*html.Node

			for gc := c.FirstChild; gc != nil; gc = gc.NextSibling {
				if gc.Type == html.ElementNode && (gc.Data == "ul" || gc.Data == "ol") {
					nestedLists = append(nestedLists, gc)
				} else {
					// Render inline content of this child
					if gc.Type == html.TextNode {
						inlineText.WriteString(collapseWhitespace(gc.Data))
					} else if gc.Type == html.ElementNode {
						inlineText.WriteString(renderInlineContent(gc))
					}
				}
			}

			text := strings.TrimSpace(inlineText.String())
			if text != "" {
				if ordered {
					sb.WriteString(fmt.Sprintf("%s%d. %s\n", indent, itemNum, text))
				} else {
					sb.WriteString(fmt.Sprintf("%s- %s\n", indent, text))
				}
			}

			for _, nested := range nestedLists {
				sb.WriteString(renderList(nested, depth+1))
			}
		}
	}
	return sb.String()
}

// renderDefinitionList renders a <dl> element into formatted text.
func renderDefinitionList(n *html.Node) string {
	var sb strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if c.Type != html.ElementNode {
			continue
		}
		switch c.Data {
		case "dt":
			text := strings.TrimSpace(renderInlineContent(c))
			if text != "" {
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(text)
				sb.WriteString(": ")
			}
		case "dd":
			text := strings.TrimSpace(renderInlineContent(c))
			if text != "" {
				sb.WriteString(text)
				sb.WriteString("\n")
			}
		}
	}
	return sb.String()
}

// getAttr returns the value of the named attribute, or empty string.
func getAttr(n *html.Node, name string) string {
	for _, a := range n.Attr {
		if a.Key == name {
			return a.Val
		}
	}
	return ""
}

// collapseWhitespace replaces runs of whitespace with single spaces.
func collapseWhitespace(s string) string {
	var sb strings.Builder
	inSpace := false
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
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

// noisePatterns are substrings that indicate a paragraph is image credit,
// stock photo attribution, or other non-content noise.
var noisePatterns = []string{
	"stock.adobe.com", "istock.com", "istockphoto", "shutterstock",
	"getty images", "unsplash", "pixabay",
}

// isNoiseParagraph returns true for paragraphs that are likely image credits,
// stock photo attributions, or other non-informative noise.
func isNoiseParagraph(p Paragraph) bool {
	if p.Heading {
		return false
	}
	text := strings.TrimSpace(p.Text)

	// Very short non-heading paragraphs are likely noise (image credits,
	// button labels, etc.) unless they contain meaningful punctuation.
	if len(text) < 15 && !strings.Contains(text, ".") && !strings.Contains(text, ":") {
		return true
	}

	lower := strings.ToLower(text)

	// Image credit / stock photo patterns
	for _, pat := range noisePatterns {
		if strings.Contains(lower, pat) {
			return true
		}
	}

	// Pattern: "Fotograf/Quelle" lines like "TU Ilmenau/ari" or "TU Ilmenau/Michael Reichel (ari)"
	if len(text) < 60 && strings.Contains(text, "/") && !strings.Contains(text, " ") ||
		(len(text) < 80 && strings.HasPrefix(text, "TU Ilmenau/")) {
		return true
	}

	return false
}

// filterNoiseParagraphs removes paragraphs that are likely noise.
func filterNoiseParagraphs(paras []Paragraph) []Paragraph {
	filtered := make([]Paragraph, 0, len(paras))
	for _, p := range paras {
		if !isNoiseParagraph(p) {
			filtered = append(filtered, p)
		}
	}
	return filtered
}
