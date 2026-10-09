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
	"net/url"
	"regexp"
	"strconv"
	"strings"
)

// maxRefreshHops bounds how many <meta http-equiv="refresh"> redirects
// FetchWebPageFollowingRefresh follows.
const maxRefreshHops = 2

var (
	metaRefreshRE = regexp.MustCompile(`(?is)<meta\b[^>]*http-equiv\s*=\s*["']?refresh["']?[^>]*>`)
	refreshAttrRE = regexp.MustCompile(`(?is)content\s*=\s*(?:"([^"]*)"|'([^']*)')`)
	refreshContRE = regexp.MustCompile(`(?is)^\s*(\d+)\s*;?\s*(?:url\s*=\s*)?(.*)$`)
)

// refreshTarget returns the address a page redirects to by <meta refresh>
// (delay of five seconds or less), resolved against base, or "".
func refreshTarget(page, base string) string {
	tag := metaRefreshRE.FindString(page)
	if tag == "" {
		return ""
	}
	a := refreshAttrRE.FindStringSubmatch(tag)
	if a == nil {
		return ""
	}
	m := refreshContRE.FindStringSubmatch(a[1] + a[2])
	if m == nil {
		return ""
	}
	if delay, err := strconv.Atoi(m[1]); err != nil || delay > 5 {
		return ""
	}
	target := strings.TrimSpace(strings.Trim(m[2], `"' `))
	if target == "" {
		return ""
	}
	b, err := url.Parse(base)
	if err != nil {
		return ""
	}
	t, err := b.Parse(target)
	if err != nil || (t.Scheme != "http" && t.Scheme != "https") {
		return ""
	}
	return t.String()
}

// FetchWebPageFollowingRefresh is FetchWebPage for pages that are only a
// stub redirecting by <meta http-equiv="refresh"> (as some portals do). It
// follows such a redirect — delay of five seconds or less, same host only,
// at most two hops — and returns the final HTML together with the address it
// came from. A refresh it will not follow (other host, too many hops) is an
// error rather than a silently returned stub page. Every hop goes through
// FetchWebPage and so through its SSRF guard and size limit.
func FetchWebPageFollowingRefresh(start string) (page, from string, err error) {
	return fetchFollowingRefresh(start, FetchWebPage)
}

func fetchFollowingRefresh(start string, fetch func(string) (string, error)) (page, from string, err error) {
	cur := start
	for hop := 0; ; hop++ {
		page, err = fetch(cur)
		if err != nil {
			return "", "", err
		}
		next := refreshTarget(page, cur)
		if next == "" {
			return page, cur, nil
		}
		cu, _ := url.Parse(cur)
		nu, _ := url.Parse(next)
		if hop >= maxRefreshHops || cu == nil || nu == nil || !strings.EqualFold(cu.Host, nu.Host) {
			return "", "", fmt.Errorf("%s redirects by <meta refresh> to %s, which is not followed (same host only, at most %d hops)", cur, next, maxRefreshHops)
		}
		cur = next
	}
}
