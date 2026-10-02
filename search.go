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
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"unicode"

	"github.com/firebase/genkit/go/ai"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ExtractQuotedPhrases returns substrings enclosed in single or double
// quotes from text (lowercased, trimmed) — used for exact-phrase keyword
// matching, bypassing whatever stemming/tokenization the rest of the search
// applies. Language-neutral: quoting works the same regardless of the
// documents' language.
func ExtractQuotedPhrases(text string) []string {
	var phrases []string
	for _, q := range []byte{'"', '\''} {
		s := text
		for {
			start := strings.IndexByte(s, q)
			if start < 0 {
				break
			}
			end := strings.IndexByte(s[start+1:], q)
			if end < 0 {
				break
			}
			phrase := strings.TrimSpace(s[start+1 : start+1+end])
			if phrase != "" {
				phrases = append(phrases, strings.ToLower(phrase))
			}
			s = s[start+1+end+1:]
		}
	}
	return phrases
}

// ExtractKeywords splits text into unique lowercase keywords, dropping
// tokens shorter than 3 characters and any in stopWords. stopWords is the
// caller's own, language-specific stop-word set (and any meta-query words
// like "find"/"show" worth excluding) — this function has no opinion on
// language. nil stopWords keeps every token that passes the length check.
func ExtractKeywords(text string, stopWords map[string]bool) []string {
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	seen := make(map[string]bool)
	var keywords []string
	for _, w := range words {
		if len(w) < 3 || stopWords[w] || seen[w] {
			continue
		}
		seen[w] = true
		keywords = append(keywords, w)
	}
	return keywords
}

// keywordDBEntry is the subset of a LocalvecStore DB file entry
// KeywordSearch needs: the chunk text plus its metadata.
type keywordDBEntry struct {
	Doc struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Metadata map[string]any `json:"metadata"`
	} `json:"doc"`
}

// KeywordSearch is LocalvecKeywordSearch against this store's own DB file —
// a convenience for a caller already holding a *LocalvecStore. An app with
// its own, hand-rolled localvec wrapper (a separate dbPath field, not this
// struct) calls LocalvecKeywordSearch directly instead.
func (s *LocalvecStore) KeywordSearch(ctx context.Context, query string, maxResults int, stopWords map[string]bool, filter func(meta map[string]any) bool) ([]*ai.Document, error) {
	return LocalvecKeywordSearch(s.dbPath, query, maxResults, stopWords, filter)
}

// LocalvecKeywordSearch returns up to maxResults documents whose content or
// metadata["parentText"] contains keywords or quoted phrases extracted from
// query (see ExtractKeywords/ExtractQuotedPhrases), scored by match count —
// a matched quoted phrase scores far above any number of matched keywords,
// so an exact name/phrase match always ranks first. filter, when non-nil,
// excludes any entry whose metadata it returns false for (nil keeps
// everything). Returns (nil, nil) when query yields no keywords or phrases
// to search for. dbPath is a localvec DB file as written by the localvec
// Genkit plugin (dir/__db_<name>.json).
func LocalvecKeywordSearch(dbPath string, query string, maxResults int, stopWords map[string]bool, filter func(meta map[string]any) bool) ([]*ai.Document, error) {
	keywords := ExtractKeywords(query, stopWords)
	phrases := ExtractQuotedPhrases(query)
	if len(keywords) == 0 && len(phrases) == 0 {
		return nil, nil
	}

	raw, err := os.ReadFile(dbPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read db: %w", err)
	}

	var db map[string]keywordDBEntry
	if err := json.Unmarshal(raw, &db); err != nil {
		return nil, fmt.Errorf("parse db: %w", err)
	}

	type scored struct {
		doc   *ai.Document
		score int
	}
	var results []scored

	// phraseBonus must outrank any pure keyword-count score.
	const phraseBonus = 1000

	for _, entry := range db {
		if filter != nil && !filter(entry.Doc.Metadata) {
			continue
		}
		var searchBuf, contentBuf strings.Builder
		for _, c := range entry.Doc.Content {
			contentBuf.WriteString(c.Text)
			searchBuf.WriteString(c.Text)
		}
		if pt, ok := entry.Doc.Metadata["parentText"].(string); ok {
			searchBuf.WriteString(" ")
			searchBuf.WriteString(pt)
		}
		fullText := strings.ToLower(searchBuf.String())

		score := 0
		for _, phrase := range phrases {
			if strings.Contains(fullText, phrase) {
				score += phraseBonus
			}
		}
		for _, kw := range keywords {
			if strings.Contains(fullText, kw) {
				score++
			}
		}
		if score == 0 {
			continue
		}
		results = append(results, scored{doc: ai.DocumentFromText(contentBuf.String(), entry.Doc.Metadata), score: score})
	}

	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })
	results = capPerDocument(results, maxPerDocument, func(r scored) string { return documentKey(r.doc.Metadata) })

	if len(results) > maxResults {
		results = results[:maxResults]
	}
	docs := make([]*ai.Document, len(results))
	for i, r := range results {
		docs[i] = r.doc
	}
	return docs, nil
}

// maxPerDocument caps how many results from the same document (identified by
// documentKey) a keyword search keeps before the final maxResults cutoff —
// without it, a document that happens to repeat a matched term often (or
// simply has many chunks) can fill the whole result budget on its own and
// starve out an equally or more relevant document that only matches once or
// twice. Found live: a generic portal page with 17 chunks mentioning
// "Fakultäten" crowded a specific faculty's own page (3 matching chunks)
// out of a 20-result keyword search entirely.
const maxPerDocument = 3

// documentKey identifies which indexed document a chunk's metadata belongs
// to — "url" for a crawled page, "source" for an uploaded file — so results
// can be diversified across documents instead of just ranked chunk by chunk.
func documentKey(meta map[string]any) string {
	if v, ok := meta["url"].(string); ok && v != "" {
		return v
	}
	if v, ok := meta["source"].(string); ok && v != "" {
		return v
	}
	return ""
}

// capPerDocument keeps only the first maxPer entries per key(item) from
// items, preserving order — items is expected to already be sorted by
// relevance, so "first" means "best" per document.
func capPerDocument[T any](items []T, maxPer int, key func(T) string) []T {
	seen := make(map[string]int, len(items))
	kept := items[:0]
	for _, item := range items {
		k := key(item)
		if seen[k] >= maxPer {
			continue
		}
		seen[k]++
		kept = append(kept, item)
	}
	return kept
}

// scanKeywordQuery runs q against pool and scans its rows with ScanDocs,
// closing the rows regardless of outcome.
func scanKeywordQuery(ctx context.Context, pool *pgxpool.Pool, q string, args []any) ([]*ai.Document, error) {
	rows, err := pool.Query(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return ScanDocs(rows)
}

// KeywordSearchSQL runs the same keyword/phrase search as
// LocalvecStore.KeywordSearch against a PostgreSQL table shaped like
// PgvecStore's documents table (content, metadata, parent_text columns):
//
//  1. Exact phrase ILIKE for quoted substrings — always wins, bypasses
//     stemming entirely, good for technical terms/compound nouns.
//  2. ILIKE on a truncated prefix of any long (>=8 char) keyword — bridges
//     inflections the stemmer misses (e.g. a word stemming differently than
//     its related forms).
//  3. Full-text search via websearch_to_tsquery, one query per keyword run
//     independently — so a high-frequency keyword can't dilute the ranking
//     of a more specific one sharing the same query.
//
// Each stage keeps at most maxPerDocument rows per document (see its doc
// comment) before its own LIMIT applies, so one document with many matching
// chunks can't single-handedly fill the result budget and starve out a
// document that only matches once or twice. Results from all three stages
// are then merged and deduplicated by document text, and capped at
// maxResults overall. language is a Postgres text-search configuration
// name ("german", "english", "simple", ...) — always a fixed string the
// caller controls, never user input, so it's safe to interpolate into the
// query text (bind parameters don't support configuration names). extraWhere
// is ANDed into every query's WHERE clause with extraArgs as its bind
// values, letting the caller add its own filter (e.g. a date/tag
// restriction); every query already binds the search term as $1 and the row
// limit as $2, so extraWhere's placeholders start at $3. Pass "", nil for no
// filter.
func KeywordSearchSQL(ctx context.Context, pool *pgxpool.Pool, query string, maxResults int, language string, stopWords map[string]bool, extraWhere string, extraArgs []any) ([]*ai.Document, error) {
	keywords := ExtractKeywords(query, stopWords)
	phrases := ExtractQuotedPhrases(query)
	if len(keywords) == 0 && len(phrases) == 0 {
		return nil, nil
	}

	filterClause := ""
	if extraWhere != "" {
		filterClause = " AND " + extraWhere
	}
	args := func(term any) []any {
		return append([]any{term, maxResults}, extraArgs...)
	}

	seen := make(map[string]bool)
	var docs []*ai.Document
	merge := func(got []*ai.Document) {
		for _, d := range got {
			key := DocText(d)
			if seen[key] {
				continue
			}
			seen[key] = true
			docs = append(docs, d)
		}
	}

	// docKey is the same "which document is this chunk part of" concept
	// documentKey uses for the localvec path, expressed in SQL — a crawled
	// page keys off its url, an uploaded file off its source name.
	const docKey = `COALESCE(metadata->>'url', metadata->>'source', '')`

	// Every stage below partitions its matches by docKey and keeps only the
	// best maxPerDocument rows per document (see maxPerDocument's doc
	// comment) before applying the overall LIMIT — otherwise a document
	// that simply has many matching chunks can fill the entire per-query
	// budget by itself and starve out an equally relevant document with
	// only one or two matches.

	// 1. Exact phrase ILIKE (quoted strings, priority).
	qPhrase := fmt.Sprintf(`
WITH matches AS (
    SELECT content, metadata, parent_text,
           row_number() OVER (PARTITION BY %[1]s ORDER BY id) AS rn
    FROM documents
    WHERE LOWER(content || ' ' || parent_text) LIKE $1%[2]s
)
SELECT content, metadata, parent_text FROM matches WHERE rn <= %[3]d
LIMIT $2`, docKey, filterClause, maxPerDocument)
	for _, phrase := range phrases {
		if len(docs) >= maxResults {
			break
		}
		got, err := scanKeywordQuery(ctx, pool, qPhrase, args("%"+phrase+"%"))
		if err != nil {
			log.Printf("[KeywordSearchSQL] phrase query error for %q: %v", phrase, err)
			continue
		}
		merge(got)
	}

	// 2. ILIKE on long keywords.
	qILIKE := fmt.Sprintf(`
WITH matches AS (
    SELECT content, metadata, parent_text,
           row_number() OVER (PARTITION BY %[1]s ORDER BY id) AS rn
    FROM documents
    WHERE LOWER(content || ' ' || parent_text) LIKE $1%[2]s
)
SELECT content, metadata, parent_text FROM matches WHERE rn <= %[3]d
LIMIT $2`, docKey, filterClause, maxPerDocument)
	for _, kw := range keywords {
		if len(kw) < 8 || len(docs) >= maxResults {
			continue
		}
		stem := kw
		if len(stem) > 10 {
			stem = stem[:len(stem)-2]
		}
		got, err := scanKeywordQuery(ctx, pool, qILIKE, args("%"+stem+"%"))
		if err != nil {
			log.Printf("[KeywordSearchSQL] ILIKE query error for %q: %v", stem, err)
			continue
		}
		merge(got)
	}

	// 3. Full-text search, one query per keyword.
	qFTS := fmt.Sprintf(`
WITH matches AS (
    SELECT content, metadata, parent_text,
           ts_rank(to_tsvector('%[1]s', content || ' ' || parent_text), websearch_to_tsquery('%[1]s', $1)) AS rank,
           row_number() OVER (
               PARTITION BY %[3]s
               ORDER BY ts_rank(to_tsvector('%[1]s', content || ' ' || parent_text), websearch_to_tsquery('%[1]s', $1)) DESC
           ) AS rn
    FROM documents
    WHERE to_tsvector('%[1]s', content || ' ' || parent_text) @@ websearch_to_tsquery('%[1]s', $1)%[2]s
)
SELECT content, metadata, parent_text FROM matches
WHERE rn <= %[4]d
ORDER BY rank DESC
LIMIT $2`, language, filterClause, docKey, maxPerDocument)
	for _, kw := range keywords {
		if len(docs) >= maxResults {
			break
		}
		got, err := scanKeywordQuery(ctx, pool, qFTS, args(kw))
		if err != nil {
			log.Printf("[KeywordSearchSQL] FTS query error for %q: %v", kw, err)
			continue
		}
		merge(got)
	}

	if len(docs) > maxResults {
		docs = docs[:maxResults]
	}
	return docs, nil
}
