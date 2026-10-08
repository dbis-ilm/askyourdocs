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
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"os"
	"strings"
	"testing"
)

// The crawl tests serve pages from httptest servers on 127.0.0.1, which
// FetchWebPage refuses by default — so the suite opts in, and the tests below
// that exercise the guard switch it back on for their duration.
func TestMain(m *testing.M) {
	SetAllowPrivateTargets(true)
	os.Exit(m.Run())
}

func withGuard(t *testing.T) {
	t.Helper()
	SetAllowPrivateTargets(false)
	t.Cleanup(func() { SetAllowPrivateTargets(true) })
}

func TestIsPublicAddr(t *testing.T) {
	cases := []struct {
		ip   string
		want bool
	}{
		{"93.184.216.34", true},
		{"8.8.8.8", true},
		{"2606:4700:4700::1111", true},
		{"127.0.0.1", false},
		{"127.255.255.254", false},
		{"::1", false},
		{"10.0.0.1", false},
		{"172.16.0.1", false},
		{"172.31.255.255", false},
		{"172.32.0.1", true},
		{"192.168.1.1", false},
		{"169.254.169.254", false}, // cloud metadata
		{"100.64.0.1", false},      // carrier-grade NAT / Tailscale
		{"0.0.0.0", false},
		{"0.1.2.3", false},
		{"224.0.0.1", false},
		{"255.255.255.255", false},
		{"fe80::1", false},
		{"fc00::1", false},
		{"fd12:3456::1", false},
		{"::", false},
		{"::ffff:127.0.0.1", false}, // IPv4-mapped loopback
		{"::ffff:10.0.0.1", false},
		{"::ffff:8.8.8.8", true},
		{"64:ff9b::7f00:1", false}, // NAT64 embedding 127.0.0.1
		{"2002:7f00:1::1", false},  // 6to4 embedding 127.0.0.1
	}
	for _, c := range cases {
		if got := isPublicAddr(netip.MustParseAddr(c.ip)); got != c.want {
			t.Errorf("isPublicAddr(%s) = %v, want %v", c.ip, got, c.want)
		}
	}
}

func TestCheckDialAddress(t *testing.T) {
	withGuard(t)
	for addr, wantErr := range map[string]bool{
		"93.184.216.34:443":     false,
		"127.0.0.1:80":          true,
		"[::1]:80":              true,
		"10.1.2.3:8080":         true,
		"[::ffff:127.0.0.1]:80": true,
		"localhost:80":          true, // not an IP: never trusted
		"nonsense":              true,
	} {
		err := checkDialAddress("tcp", addr, nil)
		if (err != nil) != wantErr {
			t.Errorf("checkDialAddress(%q) err=%v, wantErr=%v", addr, err, wantErr)
		}
		if err != nil && !errors.Is(err, ErrPrivateTarget) {
			t.Errorf("checkDialAddress(%q): error should wrap ErrPrivateTarget, got %v", addr, err)
		}
	}

	SetAllowPrivateTargets(true)
	if err := checkDialAddress("tcp", "127.0.0.1:80", nil); err != nil {
		t.Errorf("with private targets allowed: %v", err)
	}
}

func TestFetchWebPageBlocksLoopbackByDefault(t *testing.T) {
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		fmt.Fprint(w, "<html>secret</html>")
	}))
	defer srv.Close()

	withGuard(t)
	_, err := FetchWebPage(srv.URL)
	if !errors.Is(err, ErrPrivateTarget) {
		t.Fatalf("want ErrPrivateTarget, got %v", err)
	}
	if hit {
		t.Error("the request reached the private server")
	}

	SetAllowPrivateTargets(true)
	body, err := FetchWebPage(srv.URL)
	if err != nil || !strings.Contains(body, "secret") {
		t.Errorf("allowed: body=%q err=%v", body, err)
	}
}

func TestFetchWebPageBlocksRedirectToPrivateTarget(t *testing.T) {
	// Both servers are on loopback, so the guard must refuse the very first
	// hop; this test checks the redirect itself is never followed either way.
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "internal")
	}))
	defer target.Close()
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusFound)
	}))
	defer redirector.Close()

	withGuard(t)
	if _, err := FetchWebPage(redirector.URL); !errors.Is(err, ErrPrivateTarget) {
		t.Fatalf("want ErrPrivateTarget, got %v", err)
	}
}

func TestFetchWebPageStopsRedirectLoops(t *testing.T) {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, srv.URL+"/again", http.StatusFound)
	}))
	defer srv.Close()
	_, err := FetchWebPage(srv.URL)
	if err == nil || !strings.Contains(err.Error(), "redirects") {
		t.Errorf("want redirect-limit error, got %v", err)
	}
}

func TestFetchWebPageLimitsBodySize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, strings.Repeat("a", 2000))
	}))
	defer srv.Close()

	SetMaxPageBytes(1000)
	t.Cleanup(func() { SetMaxPageBytes(0) })
	if _, err := FetchWebPage(srv.URL); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("2000 bytes with a 1000 byte limit: err=%v", err)
	}

	SetMaxPageBytes(2000)
	if body, err := FetchWebPage(srv.URL); err != nil || len(body) != 2000 {
		t.Errorf("exactly at the limit should pass: len=%d err=%v", len(body), err)
	}
}
