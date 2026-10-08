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
	"net"
	"net/http"
	"net/netip"
	neturl "net/url"
	"sync/atomic"
	"syscall"
	"time"
)

// A crawler fetches whatever URL it is given from the machine it runs on. Left
// unchecked, anyone who can start a crawl can make that machine read services
// only it can reach — a local model server, a database, an admin page, a
// cloud metadata endpoint — and the content ends up in the index. FetchWebPage
// therefore refuses private targets unless an app opts in.

// ErrPrivateTarget is returned (wrapped) when a fetch is refused because the
// target resolves to a private, loopback or link-local address.
var ErrPrivateTarget = errors.New("target address is not public")

var (
	allowPrivateTargets atomic.Bool
	maxPageBytes        atomic.Int64
)

// DefaultMaxPageBytes is the largest page FetchWebPage reads, unless changed
// with SetMaxPageBytes.
const DefaultMaxPageBytes = 10 << 20

func init() { maxPageBytes.Store(DefaultMaxPageBytes) }

// SetAllowPrivateTargets lets FetchWebPage reach loopback, private and
// link-local addresses (127.0.0.0/8, 10/8, 192.168/16, 169.254/16, ...). The
// default is false. Enable it only for crawling your own intranet or a local
// development site, and never when the crawl can be started by someone you
// don't trust. It also makes FetchWebPage honour HTTP(S)_PROXY again: a proxy
// resolves targets itself, so the address check can't apply to it. Safe to
// call at any time, but meant to be set once at startup.
func SetAllowPrivateTargets(allow bool) { allowPrivateTargets.Store(allow) }

// SetMaxPageBytes caps how much of a response FetchWebPage reads; larger pages
// fail instead of filling memory. n <= 0 restores DefaultMaxPageBytes.
func SetMaxPageBytes(n int64) {
	if n <= 0 {
		n = DefaultMaxPageBytes
	}
	maxPageBytes.Store(n)
}

// notPublic lists ranges that netip's own predicates (IsLoopback, IsPrivate,
// IsLinkLocalUnicast, ...) don't cover.
var notPublic = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),      // "this network"
	netip.MustParsePrefix("100.64.0.0/10"),  // carrier-grade NAT, e.g. Tailscale
	netip.MustParsePrefix("192.0.0.0/24"),   // IETF protocol assignments
	netip.MustParsePrefix("198.18.0.0/15"),  // benchmarking
	netip.MustParsePrefix("240.0.0.0/4"),    // reserved (incl. broadcast)
	netip.MustParsePrefix("64:ff9b::/96"),   // NAT64: embeds an IPv4 target
	netip.MustParsePrefix("64:ff9b:1::/48"), // local-use NAT64
	netip.MustParsePrefix("2002::/16"),      // 6to4: embeds an IPv4 target
	netip.MustParsePrefix("fec0::/10"),      // deprecated site-local
	netip.MustParsePrefix("100::/64"),       // discard-only
	netip.MustParsePrefix("2001:db8::/32"),  // documentation
	netip.MustParsePrefix("2001::/32"),      // Teredo
}

// isPublicAddr reports whether ip is an address a crawler may connect to.
func isPublicAddr(ip netip.Addr) bool {
	ip = ip.Unmap() // ::ffff:127.0.0.1 is 127.0.0.1
	if !ip.IsValid() || ip.IsUnspecified() || ip.IsLoopback() || ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsInterfaceLocalMulticast() ||
		ip.IsMulticast() {
		return false
	}
	for _, p := range notPublic {
		if p.Contains(ip) {
			return false
		}
	}
	return true
}

// checkDialAddress is a net.Dialer.Control function: it runs with the IP the
// connection is about to use — after DNS resolution, once per candidate
// address, also for every redirect hop — so a hostname that resolves to a
// private address (or flips to one between a check and the connect, "DNS
// rebinding") cannot slip through.
func checkDialAddress(network, address string, _ syscall.RawConn) error {
	if allowPrivateTargets.Load() {
		return nil
	}
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return fmt.Errorf("%w: %q", ErrPrivateTarget, address)
	}
	ip, err := netip.ParseAddr(host)
	if err != nil || !isPublicAddr(ip) {
		return fmt.Errorf("%w: %s", ErrPrivateTarget, host)
	}
	return nil
}

const maxRedirects = 10

// newFetchClient builds the HTTP client FetchWebPage uses.
func newFetchClient() *http.Client {
	dialer := &net.Dialer{Timeout: 10 * time.Second, Control: checkDialAddress}
	transport := &http.Transport{
		DialContext:           dialer.DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 20 * time.Second,
		MaxIdleConns:          10,
		IdleConnTimeout:       30 * time.Second,
		Proxy: func(r *http.Request) (*neturl.URL, error) {
			if allowPrivateTargets.Load() {
				return http.ProxyFromEnvironment(r)
			}
			return nil, nil
		},
	}
	return &http.Client{
		Timeout:   30 * time.Second,
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= maxRedirects {
				return fmt.Errorf("stopped after %d redirects", maxRedirects)
			}
			if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
				return fmt.Errorf("redirect to unsupported scheme %q", req.URL.Scheme)
			}
			return nil
		},
	}
}
