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
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// Prompt-injection hardening helpers.
//
// None of this makes a model immune: an LLM cannot reliably tell
// instructions from data, so a hostile sentence in an indexed page or PDF
// can still sway an answer. What these helpers do is narrow the damage —
// keep untrusted text from forging the structure of the prompt, strip
// invisible characters used to smuggle instructions, and stop a model reply
// from phoning home through images or links. The real defense is giving the
// model no capability worth hijacking (read-only tools, no secrets in the
// prompt); treat this as a second layer, not a guarantee.

// Context delimiters used by WrapContext. An app's system prompt should name
// them and say that everything between them is quoted material, never
// instructions.
const (
	ContextOpenTag  = "<context>"
	ContextCloseTag = "</context>"
)

// SanitizeUntrusted removes characters that are invisible to a human reader
// but still seen by a model, a common way to hide instructions inside a
// document: Unicode tag characters (U+E0000–U+E007F, "ASCII smuggling"),
// zero-width and bidirectional-control characters, and other control
// characters except newline and tab. Visible text is left untouched.
func SanitizeUntrusted(text string) string {
	var sb strings.Builder
	sb.Grow(len(text))
	for _, r := range text {
		switch {
		case r == '\n' || r == '\t':
			sb.WriteRune(r)
		case r >= 0xE0000 && r <= 0xE007F: // tag characters
		case r >= 0x200B && r <= 0x200F: // zero-width space/joiners, LRM/RLM
		case r >= 0x202A && r <= 0x202E: // bidi embeddings and overrides
		case r >= 0x2060 && r <= 0x2064: // word joiner, invisible operators
		case r >= 0x2066 && r <= 0x2069: // bidi isolates
		case r == 0xFEFF: // zero-width no-break space / BOM
		case unicode.IsControl(r):
		default:
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

// WrapContext cleans the retrieved context with SanitizeUntrusted, defuses
// any copy of the delimiter tags inside it, and wraps the result in
// ContextOpenTag/ContextCloseTag. Pass the whole assembled context (source
// headers and chunk texts) as one string.
func WrapContext(context string) string {
	return WrapUntrusted("context", context)
}

var validTagName = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_-]*$`)

// WrapUntrusted is WrapContext for any delimiter: it cleans text with
// SanitizeUntrusted, replaces every copy of <tag> / </tag> inside it (any
// letter case, optional whitespace) with "[tag-tag removed]", trims it and
// wraps it in <tag> and </tag> on lines of their own. Use it for other
// untrusted blocks of a prompt — an incoming e-mail, a user-supplied
// document — and say in the system prompt that the block is quoted material,
// never instructions. tag must be a plain name ("mail"); anything else is a
// programming error and panics.
func WrapUntrusted(tag, text string) string {
	if !validTagName.MatchString(tag) {
		panic("askyourdocs: WrapUntrusted: invalid tag name " + strconv.Quote(tag))
	}
	re := regexp.MustCompile(`(?i)<\s*/?\s*` + regexp.QuoteMeta(tag) + `\s*>`)
	clean := re.ReplaceAllString(SanitizeUntrusted(text), "["+tag+"-tag removed]")
	return "<" + tag + ">\n" + strings.TrimSpace(clean) + "\n</" + tag + ">"
}

var (
	// answerImageRE matches a Markdown image: ![alt](url "title").
	answerImageRE = regexp.MustCompile(`!\[([^\]]*)\]\(\s*[^)]*\)`)
	// answerLinkRE matches a Markdown link: [text](url "title").
	answerLinkRE = regexp.MustCompile(`\[([^\]]*)\]\(\s*([^)\s]*)[^)]*\)`)
	// answerBareURLRE matches an http(s) URL in running text.
	answerBareURLRE = regexp.MustCompile(`https?://[^\s<>()\[\]"']+`)
	// answerHTMLRE matches an HTML tag or comment.
	answerHTMLRE = regexp.MustCompile(`<!--.*?-->|</?[a-zA-Z][^>]*>`)
)

// linkRemoved replaces a URL that is not on the allowlist.
const linkRemoved = "[Link entfernt]"

// SanitizeAnswer removes the parts of a model reply that could exfiltrate
// data or be rendered as active content, once the model has been talked
// into doing so: Markdown images (fetched automatically by most renderers,
// so a URL with data in it leaks without a click), raw HTML tags, and links
// or bare URLs that are not among allowedURLs. A link to an allowed URL
// keeps working; a link to anything else keeps its text and loses the URL.
//
// allowedURLs are the sources of the answer (the apps' citation URLs). A URL
// counts as allowed if it equals an entry or extends one at a path boundary
// ("https://a.org/x" allows "https://a.org/x/y" but not "https://a.org/xy").
// With an empty allowlist every link is removed. The removed marker is
// German ("[Link entfernt]"), like the rest of the apps' output.
//
// This works on a complete answer. A streamed reply can leak through a
// Markdown image before the closing parenthesis arrives, so streaming
// clients must not render images or follow links until the answer is done,
// and should replace what they displayed with the sanitized final text.
func SanitizeAnswer(answer string, allowedURLs []string) string {
	answer = SanitizeUntrusted(answer)
	answer = answerHTMLRE.ReplaceAllString(answer, "")
	answer = answerImageRE.ReplaceAllString(answer, "$1")
	answer = answerLinkRE.ReplaceAllStringFunc(answer, func(m string) string {
		sub := answerLinkRE.FindStringSubmatch(m)
		if urlAllowed(sub[2], allowedURLs) {
			return m
		}
		if sub[1] == "" {
			return linkRemoved
		}
		return sub[1]
	})
	return answerBareURLRE.ReplaceAllStringFunc(answer, func(u string) string {
		// Trailing sentence punctuation is not part of the URL.
		trimmed := strings.TrimRight(u, ".,;:!?")
		tail := u[len(trimmed):]
		if urlAllowed(trimmed, allowedURLs) {
			return u
		}
		return linkRemoved + tail
	})
}

// urlAllowed reports whether u equals an allowed URL or extends one at a
// path boundary. Comparison ignores a trailing slash and case in the scheme
// and host part.
func urlAllowed(u string, allowed []string) bool {
	u = strings.TrimRight(u, "/")
	for _, a := range allowed {
		a = strings.TrimRight(a, "/")
		if a == "" {
			continue
		}
		if len(u) < len(a) || !strings.EqualFold(u[:len(a)], a) {
			continue
		}
		if len(u) == len(a) || u[len(a)] == '/' || u[len(a)] == '?' || u[len(a)] == '#' {
			return true
		}
	}
	return false
}
