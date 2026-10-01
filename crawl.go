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
	"context"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/net/html"
)

// CrawlInput configures RunCrawl. It carries none of an app's own
// per-document metadata (e.g. cmd/ask-pdf's tag/datum) — that's attached by
// the IndexPageFunc closure the app passes in, not by the crawl itself.
type CrawlInput struct {
	SeedURLs []string
	MaxPages int // default 50
	MaxDepth int // default 3
	DelayMs  int // default 500 — politeness delay between page fetches
}

// CrawlResult summarises one crawl run.
type CrawlResult struct {
	PagesVisited int      `json:"pagesVisited"`
	PagesSaved   int      `json:"pagesSaved"`
	PagesIndexed int      `json:"pagesIndexed"`
	Errors       []string `json:"errors,omitempty"`
}

// ReindexResult summarises one run of ReindexCrawlData.
type ReindexResult struct {
	FilesProcessed int      `json:"filesProcessed"`
	PagesIndexed   int      `json:"pagesIndexed"`
	Errors         []string `json:"errors,omitempty"`
}

// IndexPageFunc is the callback RunCrawl and ReindexCrawlData use to index
// one page's paragraphs into the vector store — supplied by the app, which
// is what attaches its own metadata (tag/datum/rank, or just url/breadcrumb)
// before chunking and calling IndexInBatches. Returns how many documents
// were indexed.
type IndexPageFunc func(ctx context.Context, pageURL string, paras []Paragraph) (int, error)

// crawlPage represents a URL to visit with its current depth in the BFS.
type crawlPage struct {
	URL   string
	Depth int
}

// crawledPage stores the extracted data for one crawled page before indexing.
type crawledPage struct {
	url      string
	paras    []Paragraph
	filename string
}

// extractLinks parses raw HTML and returns all same-scheme links resolved
// against baseURL. Fragment-only, javascript:, mailto:, and non-HTML
// resource links are filtered out.
func extractLinks(htmlContent string, baseURL string) ([]string, error) {
	base, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}

	doc, err := html.Parse(strings.NewReader(htmlContent))
	if err != nil {
		return nil, fmt.Errorf("parsing HTML: %w", err)
	}

	// Extensions that are clearly not HTML pages.
	skipExts := map[string]bool{
		".pdf": true, ".jpg": true, ".jpeg": true, ".png": true,
		".gif": true, ".svg": true, ".webp": true, ".ico": true,
		".css": true, ".js": true, ".json": true, ".xml": true,
		".zip": true, ".gz": true, ".tar": true, ".mp3": true,
		".mp4": true, ".avi": true, ".mov": true, ".woff": true,
		".woff2": true, ".ttf": true, ".eot": true,
	}

	seen := make(map[string]bool)
	var links []string

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "a" {
			href := getAttr(n, "href")
			href = strings.TrimSpace(href)
			if href == "" || href == "#" || strings.HasPrefix(href, "javascript:") || strings.HasPrefix(href, "mailto:") {
				goto children
			}

			{
				resolved, err := base.Parse(href)
				if err != nil {
					goto children
				}

				// Only http/https.
				if resolved.Scheme != "http" && resolved.Scheme != "https" {
					goto children
				}

				// Strip fragment and query for dedup/cleanliness.
				resolved.Fragment = ""
				resolved.RawQuery = ""
				canonical := resolved.String()

				// Skip non-HTML extensions.
				ext := strings.ToLower(filepath.Ext(resolved.Path))
				if skipExts[ext] {
					goto children
				}

				if !seen[canonical] {
					seen[canonical] = true
					links = append(links, canonical)
				}
			}
		}
	children:
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)

	return links, nil
}

// urlPrefix describes the host + path prefix derived from a seed URL.
type urlPrefix struct {
	Host string
	Path string // always ends with "/"
}

// buildAllowedPrefixes extracts host+path prefixes from seed URLs.
// The path is trimmed to its directory component so that sibling pages are
// included (e.g. seed "example.com/docs/intro" → prefix "/docs/intro").
// A trailing slash is ensured for clean prefix matching.
func buildAllowedPrefixes(seedURLs []string) []urlPrefix {
	var prefixes []urlPrefix
	for _, seed := range seedURLs {
		u, err := url.Parse(seed)
		if err != nil {
			continue
		}
		p := u.Path
		if p == "" {
			p = "/"
		}
		// Ensure trailing slash so "/docs" won't accidentally match "/documents".
		if !strings.HasSuffix(p, "/") {
			p += "/"
		}
		prefixes = append(prefixes, urlPrefix{Host: u.Hostname(), Path: p})
	}
	return prefixes
}

// allowedByPrefix returns true when rawURL's host and path fall within at
// least one of the allowed prefixes.
func allowedByPrefix(rawURL string, prefixes []urlPrefix) bool {
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	host := u.Hostname()
	path := u.Path
	if !strings.HasSuffix(path, "/") {
		path += "/"
	}
	for _, pfx := range prefixes {
		if host == pfx.Host && strings.HasPrefix(path, pfx.Path) {
			return true
		}
	}
	return false
}

// urlToFilename converts a URL into a safe filename like
// "example.com_docs_getting-started.md". Query and fragment are stripped,
// the result is capped at 200 characters.
func urlToFilename(rawURL string) string {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "unknown.md"
	}

	host := u.Hostname()
	path := strings.Trim(u.Path, "/")

	var name string
	if path == "" {
		name = host
	} else {
		name = host + "_" + strings.ReplaceAll(path, "/", "_")
	}

	// Replace any remaining unsafe characters.
	name = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			return r
		}
		return '_'
	}, name)

	if len(name) > 200 {
		name = name[:200]
	}
	return name + ".md"
}

// savePageAsMarkdown writes a Markdown file with YAML frontmatter for one
// crawled page.
func savePageAsMarkdown(outputDir, filename, pageURL string, paras []Paragraph) error {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("creating output dir: %w", err)
	}

	var sb strings.Builder
	sb.WriteString("---\n")
	sb.WriteString(fmt.Sprintf("url: %s\n", pageURL))
	sb.WriteString(fmt.Sprintf("crawled: %s\n", time.Now().Format(time.RFC3339)))
	sb.WriteString("---\n\n")

	for _, p := range paras {
		if p.Heading {
			sb.WriteString("## ")
		}
		sb.WriteString(p.Text)
		sb.WriteString("\n\n")
	}

	dest := filepath.Join(outputDir, filename)
	return os.WriteFile(dest, []byte(sb.String()), 0o644)
}

// parseCrawlMarkdown parses a Markdown file written by savePageAsMarkdown back
// into a page URL and a slice of paragraphs. It expects YAML frontmatter with
// a "url:" field, followed by paragraph blocks separated by blank lines.
// Blocks starting with "## " are treated as headings.
func parseCrawlMarkdown(content string) (pageURL string, paras []Paragraph, err error) {
	if !strings.HasPrefix(content, "---\n") {
		return "", nil, fmt.Errorf("missing frontmatter")
	}
	rest := content[4:]
	end := strings.Index(rest, "\n---\n")
	if end < 0 {
		return "", nil, fmt.Errorf("frontmatter not closed")
	}
	for _, line := range strings.Split(rest[:end], "\n") {
		if strings.HasPrefix(line, "url: ") {
			pageURL = strings.TrimSpace(strings.TrimPrefix(line, "url: "))
		}
	}
	if pageURL == "" {
		return "", nil, fmt.Errorf("no url in frontmatter")
	}

	body := rest[end+5:] // skip "\n---\n"
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		if strings.HasPrefix(block, "## ") {
			paras = append(paras, Paragraph{Text: strings.TrimPrefix(block, "## "), Heading: true})
		} else {
			paras = append(paras, Paragraph{Text: block})
		}
	}
	return pageURL, paras, nil
}

// removeBoilerplateParas identifies paragraphs that appear on multiple pages
// and removes them. A non-heading paragraph appearing on more than 1 page is
// considered boilerplate (e.g. sidebar navigation, footer teasers, faculty
// cards that repeat across a site).
func removeBoilerplateParas(pages []crawledPage) {
	// Count on how many pages each paragraph text appears.
	paraPageCount := make(map[string]int)
	for _, p := range pages {
		seen := make(map[string]bool)
		for _, para := range p.paras {
			if para.Heading {
				continue
			}
			text := strings.TrimSpace(para.Text)
			if !seen[text] {
				seen[text] = true
				paraPageCount[text]++
			}
		}
	}

	// Remove paragraphs that appear on more than 1 page.
	for i := range pages {
		filtered := make([]Paragraph, 0, len(pages[i].paras))
		for _, para := range pages[i].paras {
			if para.Heading {
				filtered = append(filtered, para)
				continue
			}
			if paraPageCount[strings.TrimSpace(para.Text)] <= 1 {
				filtered = append(filtered, para)
			}
		}
		pages[i].paras = filtered
	}
}

// RunCrawl performs a breadth-first crawl starting from cfg.SeedURLs.
// It uses a two-phase approach: first crawl all pages and extract text,
// then remove cross-page boilerplate, then index the cleaned content via
// indexPage. If SAVE_CRAWL_DATA=1 is set, each cleaned page is also cached
// as Markdown under outputDir — ReindexCrawlData can later replay that cache
// without re-fetching the site.
//
// Errors on individual pages are collected in CrawlResult.Errors; only
// fatal errors (no seeds) cause an early return. progress, if non-nil, is
// called with a human-readable line at each page visit and at the phase
// transitions — used to report incremental status on a long crawl job; pass
// nil when nothing is listening.
func RunCrawl(ctx context.Context, cfg CrawlInput, outputDir string, indexPage IndexPageFunc, progress func(string)) (CrawlResult, error) {
	if progress == nil {
		progress = func(string) {}
	}
	if len(cfg.SeedURLs) == 0 {
		return CrawlResult{}, fmt.Errorf("no seed URLs provided")
	}

	maxPages := cfg.MaxPages
	if maxPages <= 0 {
		maxPages = 50
	}
	maxDepth := cfg.MaxDepth
	if maxDepth <= 0 {
		maxDepth = 3
	}
	delayMs := cfg.DelayMs
	if delayMs <= 0 {
		delayMs = 500
	}

	// Build allowed host+path prefixes from seed URLs.
	allowedPrefixes := buildAllowedPrefixes(cfg.SeedURLs)

	visited := make(map[string]bool)
	queue := make([]crawlPage, 0, len(cfg.SeedURLs))
	for _, seed := range cfg.SeedURLs {
		queue = append(queue, crawlPage{URL: seed, Depth: 0})
	}

	var result CrawlResult
	var crawledPages []crawledPage

	// ── Phase 1: Crawl all pages, extract text ──
	for len(queue) > 0 && result.PagesVisited < maxPages {
		select {
		case <-ctx.Done():
			result.Errors = append(result.Errors, "crawl cancelled: "+ctx.Err().Error())
			return result, nil
		default:
		}

		page := queue[0]
		queue = queue[1:]

		if visited[page.URL] {
			continue
		}
		visited[page.URL] = true
		result.PagesVisited++

		log.Printf("[crawl] visiting (%d/%d, depth=%d): %s", result.PagesVisited, maxPages, page.Depth, page.URL)
		progress(fmt.Sprintf("Seite %d/%d (Tiefe %d): %s", result.PagesVisited, maxPages, page.Depth, page.URL))

		htmlContent, err := FetchWebPage(page.URL)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("fetch %s: %v", page.URL, err))
			continue
		}

		paras, err := HTMLToStructuredText(strings.NewReader(htmlContent))
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("parse %s: %v", page.URL, err))
			continue
		}

		crawledPages = append(crawledPages, crawledPage{
			url:      page.URL,
			paras:    paras,
			filename: urlToFilename(page.URL),
		})

		// Extract links and enqueue if within depth.
		if page.Depth < maxDepth {
			links, err := extractLinks(htmlContent, page.URL)
			if err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("links %s: %v", page.URL, err))
			} else {
				for _, link := range links {
					if !visited[link] && allowedByPrefix(link, allowedPrefixes) {
						queue = append(queue, crawlPage{URL: link, Depth: page.Depth + 1})
					}
				}
			}
		}

		// Politeness delay
		if result.PagesVisited < maxPages && len(queue) > 0 {
			time.Sleep(time.Duration(delayMs) * time.Millisecond)
		}
	}

	// ── Phase 2: Remove cross-page boilerplate ──
	if len(crawledPages) > 1 {
		removeBoilerplateParas(crawledPages)
		log.Printf("[crawl] boilerplate removal done across %d pages", len(crawledPages))
	}
	progress(fmt.Sprintf("Indexiere %d gecrawlte Seiten...", len(crawledPages)))

	// ── Phase 3: Save and index cleaned content ──
	saveCrawlData := os.Getenv("SAVE_CRAWL_DATA") == "1"
	log.Printf("[crawl] SAVE_CRAWL_DATA=%q → saveCrawlData=%v (outputDir=%s)", os.Getenv("SAVE_CRAWL_DATA"), saveCrawlData, outputDir)
	for _, cp := range crawledPages {
		if saveCrawlData {
			if err := savePageAsMarkdown(outputDir, cp.filename, cp.url, cp.paras); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("save %s: %v", cp.url, err))
			} else {
				result.PagesSaved++
			}
		}

		indexed, err := indexPage(ctx, cp.url, cp.paras)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("index %s: %v", cp.url, err))
		} else {
			result.PagesIndexed += indexed
		}
		progress(fmt.Sprintf("%d/%d Seiten indexiert", result.PagesIndexed, len(crawledPages)))
	}

	return result, nil
}

// ReindexCrawlData replays every cached crawl-data Markdown file in dir
// (written by RunCrawl when SAVE_CRAWL_DATA=1) through indexPage, without
// re-fetching anything — lets a chunking fix or changed chunk size/overlap
// reach already-crawled sites. progress, if non-nil, is called after each
// file with a human-readable line; pass nil when nothing is listening.
func ReindexCrawlData(ctx context.Context, dir string, indexPage IndexPageFunc, progress func(string)) (ReindexResult, error) {
	if progress == nil {
		progress = func(string) {}
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return ReindexResult{}, fmt.Errorf("reading %s: %w", dir, err)
	}
	mdCount := 0
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".md") {
			mdCount++
		}
	}

	var result ReindexResult
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".md") {
			continue
		}
		result.FilesProcessed++
		path := filepath.Join(dir, entry.Name())

		data, err := os.ReadFile(path)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("read %s: %v", entry.Name(), err))
			continue
		}

		pageURL, paras, err := parseCrawlMarkdown(string(data))
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("parse %s: %v", entry.Name(), err))
			continue
		}

		if _, err := indexPage(ctx, pageURL, paras); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("index %s: %v", entry.Name(), err))
		} else {
			result.PagesIndexed++
		}
		progress(fmt.Sprintf("%d/%d Dateien verarbeitet (%s)", result.FilesProcessed, mdCount, entry.Name()))
	}

	log.Printf("[ReindexCrawlData] %d files processed, %d pages indexed, %d errors",
		result.FilesProcessed, result.PagesIndexed, len(result.Errors))
	return result, nil
}
