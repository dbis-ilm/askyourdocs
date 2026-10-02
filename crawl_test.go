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
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExtractLinks(t *testing.T) {
	htmlContent := `
	<html><body>
		<a href="/docs/intro">Intro</a>
		<a href="https://example.com/docs/setup?x=1#top">Setup</a>
		<a href="page.html">Relative</a>
		<a href="#section">Same page</a>
		<a href="">Empty</a>
		<a href="javascript:void(0)">JS</a>
		<a href="mailto:a@b.com">Mail</a>
		<a href="https://other.com/">Other host</a>
		<a href="/files/report.pdf">PDF</a>
		<a href="/docs/intro">Duplicate</a>
	</body></html>`

	links, err := extractLinks(htmlContent, "https://example.com/docs/")
	if err != nil {
		t.Fatalf("extractLinks: %v", err)
	}

	want := []string{
		"https://example.com/docs/intro",
		"https://example.com/docs/setup",
		"https://example.com/docs/page.html",
		"https://example.com/docs/", // "#section": only a bare "#" is special-cased, not "#anything"
		"https://other.com/",
	}
	if !equalStrings(links, want) {
		t.Errorf("links = %v, want %v", links, want)
	}
}

func TestExtractLinksInvalidBaseURL(t *testing.T) {
	if _, err := extractLinks("<html></html>", "://not a url"); err == nil {
		t.Error("expected an error for an invalid base URL")
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestBuildAllowedPrefixes(t *testing.T) {
	prefixes := buildAllowedPrefixes([]string{
		"https://example.com/docs/intro",
		"https://example.com/blog",
		"not a url with spaces and :://bad",
	})
	want := []urlPrefix{
		{Host: "example.com", Path: "/docs/intro/"},
		{Host: "example.com", Path: "/blog/"},
	}
	if len(prefixes) != len(want) {
		t.Fatalf("got %d prefixes, want %d: %+v", len(prefixes), len(want), prefixes)
	}
	for i := range want {
		if prefixes[i] != want[i] {
			t.Errorf("prefix %d = %+v, want %+v", i, prefixes[i], want[i])
		}
	}
}

func TestBuildAllowedPrefixesEmptyPathBecomesRoot(t *testing.T) {
	prefixes := buildAllowedPrefixes([]string{"https://example.com"})
	if len(prefixes) != 1 || prefixes[0].Path != "/" {
		t.Errorf("prefixes = %+v, want a single root prefix", prefixes)
	}
}

func TestAllowedByPrefix(t *testing.T) {
	prefixes := buildAllowedPrefixes([]string{"https://example.com/docs"})

	cases := map[string]bool{
		"https://example.com/docs/intro":      true,
		"https://example.com/docs/":           true,
		"https://example.com/documents/other": false, // must not match "/docs" as a string prefix of "/documents"
		"https://other.com/docs/intro":        false,
		"not a url":                           false,
	}
	for url, want := range cases {
		if got := allowedByPrefix(url, prefixes); got != want {
			t.Errorf("allowedByPrefix(%q) = %v, want %v", url, got, want)
		}
	}
}

func TestURLToFilename(t *testing.T) {
	cases := map[string]string{
		"https://example.com/docs/intro":   "example.com_docs_intro.md",
		"https://example.com":              "example.com.md",
		"https://example.com/":             "example.com.md",
		"https://example.com/a/b?x=1#frag": "example.com_a_b.md",
		"not a url at all with :// bad":    "unknown.md",
	}
	for url, want := range cases {
		if got := urlToFilename(url); got != want {
			t.Errorf("urlToFilename(%q) = %q, want %q", url, got, want)
		}
	}
}

func TestURLToFilenameReplacesUnsafeCharsAndCapsLength(t *testing.T) {
	longPath := strings.Repeat("a", 300)
	got := urlToFilename("https://example.com/" + longPath)
	if len(got) != 203 { // 200 chars + ".md"
		t.Errorf("len(got) = %d, want 203 (200 + .md)", len(got))
	}
	if !strings.HasSuffix(got, ".md") {
		t.Errorf("got %q, want it to end in .md", got)
	}

	got2 := urlToFilename("https://example.com/a b/c%d")
	for _, r := range strings.TrimSuffix(got2, ".md") {
		safe := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.'
		if !safe {
			t.Errorf("urlToFilename produced unsafe char %q in %q", r, got2)
		}
	}
}

func TestRemoveBoilerplateParas(t *testing.T) {
	pages := []crawledPage{
		{url: "a", paras: []Paragraph{
			{Text: "Nav Home About", Heading: false},
			{Text: "Unique content on page A", Heading: false},
			{Text: "Nav Home About", Heading: true}, // headings are never stripped
		}},
		{url: "b", paras: []Paragraph{
			{Text: "Nav Home About", Heading: false},
			{Text: "Unique content on page B", Heading: false},
		}},
	}

	removeBoilerplateParas(pages)

	for i, p := range pages {
		for _, para := range p.paras {
			if !para.Heading && strings.TrimSpace(para.Text) == "Nav Home About" {
				t.Errorf("page %d still has boilerplate paragraph %q", i, para.Text)
			}
		}
	}
	if len(pages[0].paras) != 2 { // unique content + the heading
		t.Errorf("page 0 has %d paras, want 2 (unique content + heading)", len(pages[0].paras))
	}
	if len(pages[1].paras) != 1 {
		t.Errorf("page 1 has %d paras, want 1 (unique content)", len(pages[1].paras))
	}
}

// On a large crawl, a short paragraph shared by only a handful of closely
// related pages (an address, a room number — the kind of thing a staff
// directory's several contact sub-pages legitimately repeat) must survive;
// only a paragraph repeating across a real fraction of the crawl (true
// site-wide chrome) should be stripped. Reproduces the bug found live
// against a university's executive board page, where a flat ">1 page"
// threshold stripped every contact detail that happened to also appear on
// any other staff page site-wide, leaving only 2 of 5 board members with any
// content at all.
func TestRemoveBoilerplateParasOnLargeCrawlKeepsNarrowlySharedContent(t *testing.T) {
	pages := make([]crawledPage, 20)
	for i := range pages {
		pages[i] = crawledPage{
			url: fmt.Sprintf("page-%d", i),
			paras: []Paragraph{
				{Text: "Site-wide nav footer, on every page"},
				{Text: fmt.Sprintf("Unique content on page %d", i)},
			},
		}
	}
	// A short contact detail shared by exactly 4 of the 20 pages — e.g. four
	// staff members in the same building — must not be treated as chrome.
	for i := 0; i < 4; i++ {
		pages[i].paras = append(pages[i].paras, Paragraph{Text: "Ehrenbergstraße 29 (Ernst-Abbe-Zentrum)"})
	}

	removeBoilerplateParas(pages)

	for i, p := range pages {
		for _, para := range p.paras {
			if strings.TrimSpace(para.Text) == "Site-wide nav footer, on every page" {
				t.Errorf("page %d still has true site-wide boilerplate", i)
			}
		}
	}
	for i := 0; i < 4; i++ {
		found := false
		for _, para := range pages[i].paras {
			if strings.TrimSpace(para.Text) == "Ehrenbergstraße 29 (Ernst-Abbe-Zentrum)" {
				found = true
			}
		}
		if !found {
			t.Errorf("page %d lost its narrowly-shared address paragraph — only shared by 4/20 pages, not boilerplate", i)
		}
	}
}

func TestRemoveBoilerplateParasDoesNotAffectUniqueContent(t *testing.T) {
	pages := []crawledPage{
		{url: "a", paras: []Paragraph{{Text: "Only on A"}}},
		{url: "b", paras: []Paragraph{{Text: "Only on B"}}},
	}
	removeBoilerplateParas(pages)
	if len(pages[0].paras) != 1 || len(pages[1].paras) != 1 {
		t.Errorf("unique paragraphs should survive: %+v", pages)
	}
}

func TestSavePageAndParseCrawlMarkdownRoundTrip(t *testing.T) {
	dir := t.TempDir()
	paras := []Paragraph{
		{Text: "Introduction", Heading: true},
		{Text: "Some body text about the topic."},
		{Text: "Another Section", Heading: true},
		{Text: "More body text here."},
	}

	if err := savePageAsMarkdown(dir, "test.md", "https://example.com/page", paras); err != nil {
		t.Fatalf("savePageAsMarkdown: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "test.md"))
	if err != nil {
		t.Fatalf("read saved file: %v", err)
	}

	gotURL, gotParas, err := parseCrawlMarkdown(string(data))
	if err != nil {
		t.Fatalf("parseCrawlMarkdown: %v", err)
	}
	if gotURL != "https://example.com/page" {
		t.Errorf("url = %q, want the original URL", gotURL)
	}
	if len(gotParas) != len(paras) {
		t.Fatalf("got %d paras, want %d: %+v", len(gotParas), len(paras), gotParas)
	}
	for i, want := range paras {
		if gotParas[i].Text != want.Text || gotParas[i].Heading != want.Heading {
			t.Errorf("para %d = %+v, want text=%q heading=%v", i, gotParas[i], want.Text, want.Heading)
		}
	}
}

func TestParseCrawlMarkdownErrors(t *testing.T) {
	cases := map[string]string{
		"no frontmatter at all":      "missing frontmatter",
		"---\nurl: x":                "frontmatter not closed",
		"---\nfoo: bar\n---\n\nbody": "no url",
		"---\nurl:  \n---\n\nbody":   "no url",
	}
	for content, wantErrSubstr := range cases {
		_, _, err := parseCrawlMarkdown(content)
		if err == nil {
			t.Errorf("parseCrawlMarkdown(%q) succeeded, want error containing %q", content, wantErrSubstr)
			continue
		}
		if !strings.Contains(err.Error(), wantErrSubstr) {
			t.Errorf("parseCrawlMarkdown(%q) error = %q, want it to contain %q", content, err.Error(), wantErrSubstr)
		}
	}
}

func TestRunCrawlRequiresSeedURLs(t *testing.T) {
	_, err := RunCrawl(context.Background(), CrawlInput{}, t.TempDir(), func(context.Context, string, []Paragraph) (int, error) {
		t.Fatal("indexPage should not be called without seed URLs")
		return 0, nil
	}, nil)
	if err == nil {
		t.Error("expected an error when no seed URLs are given")
	}
}

func TestRunCrawlFollowsLinksWithinPrefixAndIndexes(t *testing.T) {
	// allowedByPrefix does not trim a seed path to its parent directory (see
	// TestBuildAllowedPrefixes) — a leaf-page seed like "/base" only allows
	// paths nested under "/base/", not siblings. So the child page here must
	// live under the seed's own path to be followed.
	mux := http.NewServeMux()
	mux.HandleFunc("/base", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<html><body><p>Start page content here, long enough to be a real paragraph.</p><a href="/base/child">Child</a><a href="/outside">Outside</a></body></html>`)
	})
	mux.HandleFunc("/base/child", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<html><body><p>Child page content, also long enough to count as a paragraph.</p></body></html>`)
	})
	mux.HandleFunc("/outside", func(w http.ResponseWriter, r *http.Request) {
		t.Error("crawl should not have followed the /outside link — it's not under the seed's prefix")
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	var indexedURLs []string
	indexPage := func(ctx context.Context, pageURL string, paras []Paragraph) (int, error) {
		indexedURLs = append(indexedURLs, pageURL)
		return 1, nil
	}

	result, err := RunCrawl(context.Background(), CrawlInput{
		SeedURLs: []string{srv.URL + "/base"},
		MaxPages: 10,
		MaxDepth: 2,
		DelayMs:  1,
	}, t.TempDir(), indexPage, nil)
	if err != nil {
		t.Fatalf("RunCrawl: %v", err)
	}

	if result.PagesVisited != 2 {
		t.Errorf("PagesVisited = %d, want 2 (base + base/child, not outside)", result.PagesVisited)
	}
	if result.PagesIndexed != 2 {
		t.Errorf("PagesIndexed = %d, want 2", result.PagesIndexed)
	}
	if len(indexedURLs) != 2 {
		t.Errorf("indexed URLs = %v, want 2 entries", indexedURLs)
	}
}

func TestRunCrawlRespectsMaxPages(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/a", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<html><body><p>Page A with enough content to form a paragraph.</p><a href="/b">B</a></body></html>`)
	})
	mux.HandleFunc("/b", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<html><body><p>Page B with enough content to form a paragraph.</p><a href="/c">C</a></body></html>`)
	})
	mux.HandleFunc("/c", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<html><body><p>Page C with enough content to form a paragraph.</p></body></html>`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	result, err := RunCrawl(context.Background(), CrawlInput{
		SeedURLs: []string{srv.URL + "/a"},
		MaxPages: 1,
		MaxDepth: 5,
		DelayMs:  1,
	}, t.TempDir(), func(context.Context, string, []Paragraph) (int, error) { return 1, nil }, nil)
	if err != nil {
		t.Fatalf("RunCrawl: %v", err)
	}
	if result.PagesVisited != 1 {
		t.Errorf("PagesVisited = %d, want 1 (capped by MaxPages)", result.PagesVisited)
	}
}

func TestRunCrawlReportsProgress(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/start", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<html><body><p>Start page with enough content to form a paragraph.</p></body></html>`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	var progressed []string
	_, err := RunCrawl(context.Background(), CrawlInput{
		SeedURLs: []string{srv.URL + "/start"},
		MaxPages: 5,
		DelayMs:  1,
	}, t.TempDir(), func(context.Context, string, []Paragraph) (int, error) { return 1, nil },
		func(msg string) { progressed = append(progressed, msg) })
	if err != nil {
		t.Fatalf("RunCrawl: %v", err)
	}

	if len(progressed) == 0 {
		t.Fatal("expected at least one progress message")
	}
	if !strings.Contains(progressed[0], "/start") {
		t.Errorf("first progress message = %q, want it to mention the visited URL", progressed[0])
	}
}

func TestRunCrawlCollectsFetchErrorsWithoutAborting(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/ok", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<html><body><p>OK page with enough content to form a paragraph.</p><a href="/ok/missing">Missing</a></body></html>`)
	})
	mux.HandleFunc("/ok/missing", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	result, err := RunCrawl(context.Background(), CrawlInput{
		SeedURLs: []string{srv.URL + "/ok"},
		MaxPages: 10,
		MaxDepth: 2,
		DelayMs:  1,
	}, t.TempDir(), func(context.Context, string, []Paragraph) (int, error) { return 1, nil }, nil)
	if err != nil {
		t.Fatalf("RunCrawl: %v", err)
	}
	if len(result.Errors) != 1 {
		t.Fatalf("Errors = %v, want exactly 1 fetch error for /ok/missing", result.Errors)
	}
	if result.PagesIndexed != 1 {
		t.Errorf("PagesIndexed = %d, want 1 (the OK page still got indexed)", result.PagesIndexed)
	}
}

func TestRunCrawlSavesMarkdownWhenEnabled(t *testing.T) {
	t.Setenv("SAVE_CRAWL_DATA", "1")

	mux := http.NewServeMux()
	mux.HandleFunc("/start", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<html><body><p>Start page content, long enough to be a real paragraph.</p></body></html>`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dir := t.TempDir()
	_, err := RunCrawl(context.Background(), CrawlInput{
		SeedURLs: []string{srv.URL + "/start"},
		MaxPages: 5,
		DelayMs:  1,
	}, dir, func(context.Context, string, []Paragraph) (int, error) { return 1, nil }, nil)
	if err != nil {
		t.Fatalf("RunCrawl: %v", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read output dir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("got %d files in output dir, want 1", len(entries))
	}
}
