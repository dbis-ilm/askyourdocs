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
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestGermanProgressWordsEveryStage(t *testing.T) {
	for _, tc := range []struct {
		p    Progress
		want string
	}{
		{Progress{Stage: StageChunked, Total: 12}, "12 Chunks erzeugt, indexiere..."},
		{Progress{Stage: StageBatchIndexed, Done: 2, Total: 5}, "Batch 2/5 indexiert"},
		{Progress{Stage: StageCrawlFetch, Done: 3, Total: 50, Depth: 1, URL: "https://a.example/x"}, "Seite 3/50 (Tiefe 1): https://a.example/x"},
		{Progress{Stage: StageCrawlIndexing, Total: 40}, "Indexiere 40 gecrawlte Seiten..."},
		{Progress{Stage: StageCrawlPageIndexed, Done: 7, Total: 40}, "7/40 Seiten indexiert"},
		{Progress{Stage: StageCrawlCheckGone, Done: 2, Total: 3, URL: "https://a.example/y"}, "Prüfe nicht mehr gefundene Seite 2/3: https://a.example/y"},
		{Progress{Stage: StageReindexFile, Done: 1, Total: 9, File: "a.md"}, "1/9 Dateien verarbeitet (a.md)"},
		{Progress{Stage: "unknown"}, ""},
	} {
		if got := GermanProgress(tc.p); got != tc.want {
			t.Errorf("GermanProgress(%+v) = %q, want %q", tc.p, got, tc.want)
		}
	}
}

func TestProgressMessages(t *testing.T) {
	if ProgressMessages(nil, GermanProgress) != nil {
		t.Error("nil send: want nil ProgressFunc")
	}
	var lines []string
	f := ProgressMessages(func(s string) { lines = append(lines, s) }, GermanProgress)
	f(Progress{Stage: StageBatchIndexed, Done: 1, Total: 2})
	f(Progress{Stage: "unknown"}) // formats to "": dropped
	if want := []string{"Batch 1/2 indexiert"}; !reflect.DeepEqual(lines, want) {
		t.Errorf("lines = %q, want %q", lines, want)
	}
	var nilFunc ProgressFunc
	nilFunc.report(Progress{Stage: StageChunked}) // must not panic
}

func TestRunCrawlProgressEvents(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/base", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `<html><body><p>Page A with enough content to form a paragraph.</p><a href="/base/child">B</a></body></html>`)
	})
	mux.HandleFunc("/base/child", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `<html><body><p>Page B with enough content to form a paragraph.</p></body></html>`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	// The crawl only follows links below the seed's path.
	var events []Progress
	_, err := RunCrawl(context.Background(), CrawlInput{SeedURLs: []string{srv.URL + "/base"}, MaxPages: 10, MaxDepth: 2, DelayMs: 1},
		t.TempDir(), func(context.Context, string, []Paragraph) (int, error) { return 3, nil },
		func(p Progress) { events = append(events, p) })
	if err != nil {
		t.Fatalf("RunCrawl: %v", err)
	}
	want := []Progress{
		{Stage: StageCrawlFetch, Done: 1, Total: 10, Depth: 0, URL: srv.URL + "/base"},
		{Stage: StageCrawlFetch, Done: 2, Total: 10, Depth: 1, URL: srv.URL + "/base/child"},
		{Stage: StageCrawlIndexing, Total: 2},
		{Stage: StageCrawlPageIndexed, Done: 1, Total: 2, URL: srv.URL + "/base"},
		{Stage: StageCrawlPageIndexed, Done: 2, Total: 2, URL: srv.URL + "/base/child"},
	}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("events =\n%+v\nwant\n%+v", events, want)
	}
}

func TestReindexCrawlDataProgressEvents(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a.md", "b.md"} {
		page := "---\nurl: https://a.example/" + name + "\n---\n\nEin Absatz mit genug Inhalt.\n"
		if err := os.WriteFile(filepath.Join(dir, name), []byte(page), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var events []Progress
	if _, err := ReindexCrawlData(context.Background(), dir, func(context.Context, string, []Paragraph) (int, error) { return 1, nil },
		func(p Progress) { events = append(events, p) }); err != nil {
		t.Fatalf("ReindexCrawlData: %v", err)
	}
	want := []Progress{
		{Stage: StageReindexFile, Done: 1, Total: 2, File: "a.md"},
		{Stage: StageReindexFile, Done: 2, Total: 2, File: "b.md"},
	}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("events = %+v, want %+v", events, want)
	}
}
