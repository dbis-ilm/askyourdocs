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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"time"
)

// CrawlManifestFile is the file RunCrawl keeps in its outputDir when
// WithManifest is given. An app that empties its index (a store Reset)
// should delete it too, or the next crawl skips every page as unchanged.
const CrawlManifestFile = "crawl-manifest.json"

// CrawlOption changes how RunCrawl treats pages it has seen before.
type CrawlOption func(*crawlOptions)

type crawlOptions struct {
	manifest    bool
	fingerprint string
	force       bool
	removePage  func(ctx context.Context, pageURL string) error
}

// WithManifest makes RunCrawl incremental. It remembers a hash of every
// indexed page's cleaned text in CrawlManifestFile under outputDir and skips
// indexPage (and so the embedding) for a page whose hash is unchanged.
//
// fingerprint is mixed into every hash. Put everything into it that changes
// what indexPage makes of the same text — chunk size and overlap, the
// metadata the closure attaches — so a change there reindexes the pages
// instead of leaving them as they were.
//
// The hash is taken after boilerplate removal, which depends on the whole
// crawl: a crawl with a different page set can see changes on pages whose
// content didn't change, and reindexes them. That costs time, never data.
func WithManifest(fingerprint string) CrawlOption {
	return func(o *crawlOptions) {
		o.manifest = true
		o.fingerprint = fingerprint
	}
}

// WithForceReindex hands every crawled page to indexPage even when its hash
// is unchanged, and still updates the manifest. For after the index was
// emptied or rebuilt without the manifest knowing.
func WithForceReindex() CrawlOption {
	return func(o *crawlOptions) { o.force = true }
}

// WithRemovePage lets RunCrawl remove pages that are gone. After the crawl,
// every page in the manifest that lies under this crawl's seed URLs but was
// not crawled this time is fetched once more. Only a 404 or 410 counts as
// gone: remove is called (the app deletes the page's chunks, e.g. with
// DeleteByURL), the page's cached Markdown file is deleted and the page
// leaves the manifest. Any other answer, or no answer, keeps the page — a
// crawl capped by MaxPages or a server having a bad minute must not empty
// the index.
//
// It needs WithManifest; without it RunCrawl doesn't know which pages it
// had before and remove is never called.
func WithRemovePage(remove func(ctx context.Context, pageURL string) error) CrawlOption {
	return func(o *crawlOptions) { o.removePage = remove }
}

// crawlManifest is the on-disk record of what a crawl indexed.
type crawlManifest struct {
	Version int                      `json:"version"`
	Pages   map[string]manifestEntry `json:"pages"`
}

type manifestEntry struct {
	Hash     string    `json:"hash"`
	File     string    `json:"file"`
	LastSeen time.Time `json:"lastSeen"`
}

// manifestLocks serializes manifest updates per file, so two crawls into
// the same outputDir in one process merge their changes instead of the
// later save dropping the earlier one's pages.
var manifestLocks sync.Map // abs path -> *sync.Mutex

func manifestLock(path string) *sync.Mutex {
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	mu, _ := manifestLocks.LoadOrStore(path, &sync.Mutex{})
	return mu.(*sync.Mutex)
}

// loadCrawlManifest reads path; a missing file is an empty manifest.
func loadCrawlManifest(path string) (crawlManifest, error) {
	m := crawlManifest{Version: 1, Pages: map[string]manifestEntry{}}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return m, nil
	}
	if err != nil {
		return m, fmt.Errorf("read crawl manifest: %w", err)
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return m, fmt.Errorf("parse crawl manifest %s: %w", path, err)
	}
	if m.Pages == nil {
		m.Pages = map[string]manifestEntry{}
	}
	return m, nil
}

// manifestChanges is what one crawl run did, applied to the manifest as it
// is on disk at the end of the run.
type manifestChanges struct {
	set     map[string]manifestEntry
	removed map[string]bool
}

// saveCrawlManifest re-reads path under its lock, applies ch and writes
// the result atomically (temp file + rename).
func saveCrawlManifest(path string, ch manifestChanges) error {
	mu := manifestLock(path)
	mu.Lock()
	defer mu.Unlock()

	m, err := loadCrawlManifest(path)
	if err != nil {
		return err
	}
	for u, e := range ch.set {
		m.Pages[u] = e
	}
	for u := range ch.removed {
		delete(m.Pages, u)
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create crawl manifest dir: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".crawl-manifest-*.json")
	if err != nil {
		return fmt.Errorf("write crawl manifest: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return fmt.Errorf("write crawl manifest: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return fmt.Errorf("write crawl manifest: %w", err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		os.Remove(tmp.Name())
		return fmt.Errorf("write crawl manifest: %w", err)
	}
	return nil
}

// pageHash hashes a page's cleaned paragraphs together with fingerprint.
func pageHash(fingerprint string, paras []Paragraph) string {
	h := sha256.New()
	fmt.Fprintf(h, "%d:%s\n", len(fingerprint), fingerprint)
	for _, p := range paras {
		kind := "p"
		if p.Heading {
			kind = "h"
		}
		fmt.Fprintf(h, "%s%d:%s\n", kind, len(p.Text), p.Text)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// pageGone reports whether pageURL answers 404 or 410 (after redirects).
// err is set only when there was no answer at all.
func pageGone(ctx context.Context, pageURL string) (bool, error) {
	client := newFetchClient()
	defer client.CloseIdleConnections()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, pageURL, nil)
	if err != nil {
		return false, fmt.Errorf("invalid URL: %w", err)
	}
	req.Header.Set("User-Agent", "askyourdocs/1.0 (web indexer)")
	resp, err := client.Do(req)
	if err != nil {
		return false, err
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusGone, nil
}

// removeCachedPage deletes the cached Markdown file for pageURL, but only
// if the file still belongs to it: urlToFilename drops the query, so two
// URLs can share a file name.
func removeCachedPage(outputDir, filename, pageURL string) error {
	if filename == "" {
		return nil
	}
	path := filepath.Join(outputDir, filename)
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if u, _, err := parseCrawlMarkdown(string(data)); err != nil || u != pageURL {
		return nil
	}
	return os.Remove(path)
}

// removeGonePages is RunCrawl's last phase under WithRemovePage: every
// manifest page under this crawl's seeds that wasn't crawled now is fetched
// once more and removed only on 404/410. Removals are recorded in ch.
func removeGonePages(ctx context.Context, result *CrawlResult, m crawlManifest, crawled []crawledPage,
	prefixes []urlPrefix, outputDir string, delayMs int,
	remove func(context.Context, string) error, ch manifestChanges, progress ProgressFunc) {

	seen := make(map[string]bool, len(crawled))
	for _, cp := range crawled {
		seen[cp.url] = true
	}
	var candidates []string
	for u := range m.Pages {
		if !seen[u] && allowedByPrefix(u, prefixes) {
			candidates = append(candidates, u)
		}
	}
	slices.Sort(candidates)

	for i, u := range candidates {
		if ctx.Err() != nil {
			result.Errors = append(result.Errors, "removal check cancelled: "+ctx.Err().Error())
			return
		}
		if i > 0 {
			time.Sleep(time.Duration(delayMs) * time.Millisecond)
		}
		progress.report(Progress{Stage: StageCrawlCheckGone, Done: i + 1, Total: len(candidates), URL: u})

		gone, err := pageGone(ctx, u)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("check %s: %v", u, err))
			continue
		}
		if !gone {
			continue
		}
		if err := remove(ctx, u); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("remove %s: %v", u, err))
			continue
		}
		if err := removeCachedPage(outputDir, m.Pages[u].File, u); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("remove cached %s: %v", u, err))
		}
		ch.removed[u] = true
		result.PagesRemoved++
		slog.InfoContext(ctx, "crawl: page gone, removed", "url", u)
	}
}
