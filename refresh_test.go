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
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRefreshTarget(t *testing.T) {
	tests := []struct{ name, page, want string }{
		{"relative", `<meta http-equiv="refresh" content="0; url=/portal/start">`, "http://h.example/portal/start"},
		{"absolute, caps, no quotes on equiv", `<META HTTP-EQUIV=refresh CONTENT="1;URL=https://h.example/x">`, "https://h.example/x"},
		{"quoted url", `<meta http-equiv="refresh" content="0;url='/y'">`, "http://h.example/y"},
		{"long delay", `<meta http-equiv="refresh" content="30; url=/z">`, ""},
		{"reload only", `<meta http-equiv="refresh" content="5">`, ""},
		{"javascript scheme", `<meta http-equiv="refresh" content="0;url=javascript:alert(1)">`, ""},
		{"no meta", `<html><body>hi</body></html>`, ""},
	}
	for _, tc := range tests {
		if got := refreshTarget(tc.page, "http://h.example/a"); got != tc.want {
			t.Errorf("%s: refreshTarget = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestFetchFollowingRefresh(t *testing.T) {
	pages := map[string]string{
		"http://h.example/a":  `<meta http-equiv="refresh" content="0;url=/b">`,
		"http://h.example/b":  `<html><body>Inhalt</body></html>`,
		"http://h.example/c":  `<meta http-equiv="refresh" content="0;url=http://other.example/x">`,
		"http://h.example/l":  `<meta http-equiv="refresh" content="0;url=/l2">`,
		"http://h.example/l2": `<meta http-equiv="refresh" content="0;url=/l3">`,
		"http://h.example/l3": `<meta http-equiv="refresh" content="0;url=/l4">`,
	}
	fetch := func(u string) (string, error) { return pages[u], nil }

	page, from, err := fetchFollowingRefresh("http://h.example/a", fetch)
	if err != nil || from != "http://h.example/b" || !strings.Contains(page, "Inhalt") {
		t.Errorf("got (%q, %q, %v), want the page at /b", page, from, err)
	}
	if _, _, err := fetchFollowingRefresh("http://h.example/c", fetch); err == nil {
		t.Error("refresh to another host was followed")
	}
	if _, _, err := fetchFollowingRefresh("http://h.example/l", fetch); err == nil {
		t.Error("refresh chain longer than the hop limit was followed")
	}
}

func TestFetchWebPageFollowingRefreshEndToEnd(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/start", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><head><meta http-equiv="refresh" content="0; url=/real"></head></html>`))
	})
	mux.HandleFunc("/real", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body><p>Echter Inhalt</p></body></html>`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	page, from, err := FetchWebPageFollowingRefresh(srv.URL + "/start")
	if err != nil {
		t.Fatal(err)
	}
	if from != srv.URL+"/real" || !strings.Contains(page, "Echter Inhalt") {
		t.Errorf("got from=%q page=%q", from, page)
	}
}
