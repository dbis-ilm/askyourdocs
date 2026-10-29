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
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
)

// site is a small mutable test site: pages maps a path to its body
// paragraph (with links to every other page listed in links), status
// overrides the answer for a path.
type site struct {
	mu     sync.Mutex
	text   map[string]string
	links  map[string][]string
	status map[string]int
}

func newSite() *site {
	return &site{text: map[string]string{}, links: map[string][]string{}, status: map[string]int{}}
}

func (s *site) set(path, text string, links ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.text[path] = text
	s.links[path] = links
	delete(s.status, path)
}

func (s *site) setStatus(path string, code int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status[path] = code
}

func (s *site) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if code, ok := s.status[r.URL.Path]; ok {
		http.Error(w, http.StatusText(code), code)
		return
	}
	text, ok := s.text[r.URL.Path]
	if !ok {
		http.NotFound(w, r)
		return
	}
	fmt.Fprintf(w, "<html><body><p>%s</p>", text)
	for _, l := range s.links[r.URL.Path] {
		fmt.Fprintf(w, `<a href="%s">link</a>`, l)
	}
	fmt.Fprint(w, "</body></html>")
}

// recorder is an IndexPageFunc and a remove callback that log their calls.
type recorder struct {
	mu      sync.Mutex
	indexed []string
	removed []string
	failOn  string
}

func (r *recorder) index(_ context.Context, pageURL string, _ []Paragraph) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if pageURL == r.failOn {
		return 0, errors.New("embedding backend down")
	}
	r.indexed = append(r.indexed, pageURL)
	return 1, nil
}

func (r *recorder) remove(_ context.Context, pageURL string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.removed = append(r.removed, pageURL)
	return nil
}

func (r *recorder) reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.indexed, r.removed = nil, nil
}

// twoPageSite serves /base linking to /base/a and /base/b.
func twoPageSite(t *testing.T) (*site, string) {
	t.Helper()
	s := newSite()
	s.set("/base", "Start page with enough text to be a paragraph of its own.", "/base/a", "/base/b")
	s.set("/base/a", "Page A explains the admission rules in some detail here.")
	s.set("/base/b", "Page B lists the application deadlines for every semester.")
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	return s, srv.URL
}

func crawlOnce(t *testing.T, base, dir string, rec *recorder, opts ...CrawlOption) CrawlResult {
	t.Helper()
	res, err := RunCrawl(context.Background(), CrawlInput{
		SeedURLs: []string{base + "/base"}, MaxPages: 10, MaxDepth: 2, DelayMs: 1,
	}, dir, rec.index, nil, opts...)
	if err != nil {
		t.Fatalf("RunCrawl: %v", err)
	}
	return res
}

func TestRunCrawlWithManifestSkipsUnchangedPages(t *testing.T) {
	s, base := twoPageSite(t)
	dir := t.TempDir()
	rec := &recorder{}

	first := crawlOnce(t, base, dir, rec, WithManifest("chunk=1200"))
	if first.PagesIndexed != 3 || first.PagesUnchanged != 0 {
		t.Fatalf("first crawl: indexed %d, unchanged %d; want 3, 0", first.PagesIndexed, first.PagesUnchanged)
	}

	rec.reset()
	second := crawlOnce(t, base, dir, rec, WithManifest("chunk=1200"))
	if second.PagesIndexed != 0 || second.PagesUnchanged != 3 || len(rec.indexed) != 0 {
		t.Fatalf("unchanged recrawl: indexed %d (%v), unchanged %d; want 0, 3", second.PagesIndexed, rec.indexed, second.PagesUnchanged)
	}

	s.set("/base/a", "Page A now explains different admission rules than before.")
	rec.reset()
	third := crawlOnce(t, base, dir, rec, WithManifest("chunk=1200"))
	if want := []string{base + "/base/a"}; !slices.Equal(rec.indexed, want) {
		t.Errorf("after changing /base/a, indexed %v, want %v", rec.indexed, want)
	}
	if third.PagesUnchanged != 2 {
		t.Errorf("PagesUnchanged = %d, want 2", third.PagesUnchanged)
	}
}

func TestRunCrawlWithManifestReindexesOnNewFingerprintOrForce(t *testing.T) {
	_, base := twoPageSite(t)
	dir := t.TempDir()
	rec := &recorder{}
	crawlOnce(t, base, dir, rec, WithManifest("chunk=1200"))

	rec.reset()
	if res := crawlOnce(t, base, dir, rec, WithManifest("chunk=800")); res.PagesIndexed != 3 {
		t.Errorf("new fingerprint: PagesIndexed = %d, want 3", res.PagesIndexed)
	}
	rec.reset()
	if res := crawlOnce(t, base, dir, rec, WithManifest("chunk=800"), WithForceReindex()); res.PagesIndexed != 3 {
		t.Errorf("force: PagesIndexed = %d, want 3", res.PagesIndexed)
	}
	rec.reset()
	if res := crawlOnce(t, base, dir, rec, WithManifest("chunk=800")); res.PagesUnchanged != 3 {
		t.Errorf("after force the manifest must still be current: PagesUnchanged = %d, want 3", res.PagesUnchanged)
	}
}

func TestRunCrawlWithManifestRetriesPagesThatFailedToIndex(t *testing.T) {
	_, base := twoPageSite(t)
	dir := t.TempDir()
	rec := &recorder{failOn: base + "/base/b"}
	crawlOnce(t, base, dir, rec, WithManifest(""))

	rec.failOn = ""
	rec.reset()
	crawlOnce(t, base, dir, rec, WithManifest(""))
	if want := []string{base + "/base/b"}; !slices.Equal(rec.indexed, want) {
		t.Errorf("second crawl indexed %v, want only the page that failed before: %v", rec.indexed, want)
	}
}

func TestRunCrawlWithoutManifestWritesNoManifest(t *testing.T) {
	_, base := twoPageSite(t)
	dir := t.TempDir()
	crawlOnce(t, base, dir, &recorder{})
	if _, err := os.Stat(filepath.Join(dir, CrawlManifestFile)); !os.IsNotExist(err) {
		t.Errorf("manifest written without WithManifest (stat err %v)", err)
	}
}

func TestRunCrawlRemovesOnlyPagesThatAnswer404Or410(t *testing.T) {
	t.Setenv("SAVE_CRAWL_DATA", "1")
	s, base := twoPageSite(t)
	s.set("/base", "Start page with enough text to be a paragraph of its own.", "/base/a", "/base/b", "/base/c", "/base/d")
	s.set("/base/c", "Page C describes the semester fees and how to pay them.")
	s.set("/base/d", "Page D introduces the student advisory service and its hours.")
	dir := t.TempDir()
	rec := &recorder{}
	opts := []CrawlOption{WithManifest(""), WithRemovePage(rec.remove)}
	crawlOnce(t, base, dir, rec, opts...)

	// None of a–d is linked any more: a is 404, b is 410, c still exists,
	// d answers 500. Only a and b are gone.
	s.set("/base", "Start page with enough text to be a paragraph of its own.")
	s.setStatus("/base/a", http.StatusNotFound)
	s.setStatus("/base/b", http.StatusGone)
	s.setStatus("/base/d", http.StatusInternalServerError)
	rec.reset()
	res := crawlOnce(t, base, dir, rec, opts...)

	want := []string{base + "/base/a", base + "/base/b"}
	if !slices.Equal(rec.removed, want) {
		t.Fatalf("removed %v, want %v", rec.removed, want)
	}
	if res.PagesRemoved != 2 {
		t.Errorf("PagesRemoved = %d, want 2", res.PagesRemoved)
	}
	for _, u := range want {
		if _, err := os.Stat(filepath.Join(dir, urlToFilename(u))); !os.IsNotExist(err) {
			t.Errorf("cached Markdown for %s still there (stat err %v); ReindexCrawlData would bring it back", u, err)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, urlToFilename(base+"/base/c"))); err != nil {
		t.Errorf("cached Markdown for the kept page /base/c: %v", err)
	}

	m, err := loadCrawlManifest(filepath.Join(dir, CrawlManifestFile))
	if err != nil {
		t.Fatal(err)
	}
	for _, u := range want {
		if _, ok := m.Pages[u]; ok {
			t.Errorf("%s still in the manifest", u)
		}
	}
	for _, u := range []string{base + "/base/c", base + "/base/d"} {
		if _, ok := m.Pages[u]; !ok {
			t.Errorf("kept page %s dropped from the manifest", u)
		}
	}

	// The next run doesn't ask about the removed pages again.
	rec.reset()
	crawlOnce(t, base, dir, rec, opts...)
	if len(rec.removed) != 0 {
		t.Errorf("third crawl removed %v again", rec.removed)
	}
}

func TestRunCrawlCappedByMaxPagesRemovesNothingThatStillExists(t *testing.T) {
	_, base := twoPageSite(t)
	dir := t.TempDir()
	rec := &recorder{}
	opts := []CrawlOption{WithManifest(""), WithRemovePage(rec.remove)}
	crawlOnce(t, base, dir, rec, opts...)

	rec.reset()
	if _, err := RunCrawl(context.Background(), CrawlInput{
		SeedURLs: []string{base + "/base"}, MaxPages: 1, MaxDepth: 2, DelayMs: 1,
	}, dir, rec.index, nil, opts...); err != nil {
		t.Fatal(err)
	}
	if len(rec.removed) != 0 {
		t.Errorf("a crawl cut short by MaxPages removed %v", rec.removed)
	}
}

func TestRunCrawlLeavesPagesOutsideItsSeedsAlone(t *testing.T) {
	s, base := twoPageSite(t)
	s.set("/other", "A page under another seed, crawled by a different job.")
	dir := t.TempDir()
	rec := &recorder{}
	if _, err := RunCrawl(context.Background(), CrawlInput{SeedURLs: []string{base + "/other"}, DelayMs: 1},
		dir, rec.index, nil, WithManifest("")); err != nil {
		t.Fatal(err)
	}
	s.setStatus("/other", http.StatusNotFound)

	rec.reset()
	crawlOnce(t, base, dir, rec, WithManifest(""), WithRemovePage(rec.remove))
	if len(rec.removed) != 0 {
		t.Errorf("crawl of /base removed %v, which belongs to another seed", rec.removed)
	}
	m, _ := loadCrawlManifest(filepath.Join(dir, CrawlManifestFile))
	if _, ok := m.Pages[base+"/other"]; !ok {
		t.Error("the other seed's page was dropped from the shared manifest")
	}
}

func TestRunCrawlKeepsPageWhenRemoveFails(t *testing.T) {
	s, base := twoPageSite(t)
	dir := t.TempDir()
	rec := &recorder{}
	failing := func(context.Context, string) error { return errors.New("store unavailable") }
	crawlOnce(t, base, dir, rec, WithManifest(""))

	s.set("/base", "Start page with enough text to be a paragraph of its own.", "/base/a")
	s.setStatus("/base/b", http.StatusNotFound)
	res := crawlOnce(t, base, dir, rec, WithManifest(""), WithRemovePage(failing))
	if res.PagesRemoved != 0 || len(res.Errors) == 0 {
		t.Errorf("PagesRemoved %d, errors %v; want 0 and the remove error", res.PagesRemoved, res.Errors)
	}

	rec.reset()
	crawlOnce(t, base, dir, rec, WithManifest(""), WithRemovePage(rec.remove))
	if want := []string{base + "/base/b"}; !slices.Equal(rec.removed, want) {
		t.Errorf("retry removed %v, want %v", rec.removed, want)
	}
}

func TestSaveCrawlManifestMergesConcurrentRuns(t *testing.T) {
	path := filepath.Join(t.TempDir(), CrawlManifestFile)
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			u := fmt.Sprintf("https://a.example/%d", i)
			if err := saveCrawlManifest(path, manifestChanges{set: map[string]manifestEntry{u: {Hash: "h"}}}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	m, err := loadCrawlManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Pages) != 20 {
		t.Errorf("manifest has %d pages, want 20 — a save dropped another's entries", len(m.Pages))
	}
}

func TestRemoveCachedPageSparesAnotherURLsFile(t *testing.T) {
	dir := t.TempDir()
	// urlToFilename drops the query, so both URLs map to one file.
	file := urlToFilename("https://a.example/p?x=1")
	if err := savePageAsMarkdown(dir, file, "https://a.example/p?x=2", []Paragraph{{Text: "x"}}); err != nil {
		t.Fatal(err)
	}
	if err := removeCachedPage(dir, file, "https://a.example/p?x=1"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, file)); err != nil {
		t.Errorf("file of https://a.example/p?x=2 deleted while removing ?x=1: %v", err)
	}
}

func TestLoadCrawlManifestRejectsGarbage(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, CrawlManifestFile), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := RunCrawl(context.Background(), CrawlInput{SeedURLs: []string{"http://127.0.0.1:1/x"}, MaxPages: 1, DelayMs: 1},
		dir, (&recorder{}).index, nil, WithManifest(""))
	if err == nil {
		t.Error("RunCrawl with an unreadable manifest returned no error; it would reindex everything and lose removals")
	}
}

func TestRunCrawlForgetsHashOfPageThatFailedToReindex(t *testing.T) {
	_, base := twoPageSite(t)
	dir := t.TempDir()
	rec := &recorder{}
	crawlOnce(t, base, dir, rec, WithManifest(""))

	// A forced run fails on /base/b after indexPage dropped its old chunks.
	rec.failOn = base + "/base/b"
	crawlOnce(t, base, dir, rec, WithManifest(""), WithForceReindex())

	rec.failOn = ""
	rec.reset()
	crawlOnce(t, base, dir, rec, WithManifest(""))
	if want := []string{base + "/base/b"}; !slices.Equal(rec.indexed, want) {
		t.Errorf("indexed %v, want %v — the failed page must not count as unchanged", rec.indexed, want)
	}
}
