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
	"log"
	"strings"
	"unicode"

	"github.com/firebase/genkit/go/ai"
)

// QATurn is one completed question/answer exchange from earlier in a
// conversation, as sent back by a multi-turn client.
type QATurn struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// MaxHistoryTurns caps how many prior turns feed the condense prompt, so a
// long-running conversation doesn't grow that prompt's context without
// bound. Older turns are dropped, not summarized — for pronoun/ellipsis
// resolution the most recent turns carry almost all the signal anyway.
const MaxHistoryTurns = 6

// TrimHistory keeps only the most recent MaxHistoryTurns entries.
func TrimHistory(history []QATurn) []QATurn {
	if len(history) <= MaxHistoryTurns {
		return history
	}
	return history[len(history)-MaxHistoryTurns:]
}

// FormatHistory renders history as plain turns for the condense prompt.
func FormatHistory(history []QATurn) string {
	var sb strings.Builder
	for _, t := range history {
		sb.WriteString("Nutzer: ")
		sb.WriteString(t.Question)
		sb.WriteString("\nAssistent: ")
		sb.WriteString(t.Answer)
		sb.WriteString("\n\n")
	}
	return strings.TrimSpace(sb.String())
}

// extractNumberTokens returns the set of maximal digit runs in text — years,
// session numbers, day-of-month, etc. Used to sanity-check a condensed
// question against the conversation it came from.
func extractNumberTokens(text string) map[string]bool {
	tokens := map[string]bool{}
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			tokens[cur.String()] = true
			cur.Reset()
		}
	}
	for _, r := range text {
		if r >= '0' && r <= '9' {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return tokens
}

func hasAny(a, b map[string]bool) bool {
	for k := range a {
		if b[k] {
			return true
		}
	}
	return false
}

// ConfidenceLabel derives a coarse "hoch"/"mittel"/"niedrig" label from
// topScore — the reranker's 0-10 relevance score (see the app's rerank prompt) for the
// highest-scoring source actually used to answer — and usedFallback, which is
// true when the reranker scored every candidate below minScore and the top-3
// fallback kicked in. A fallback answer is never "hoch" even if a stale
// neutral default score (a neutral default score) happens to be mid-range: it
// means the reranker found nothing it considered actually relevant, so the
// answer rests on the least-bad options rather than a confident match.
func ConfidenceLabel(topScore int, usedFallback bool) string {
	switch {
	case usedFallback:
		return "niedrig"
	case topScore >= 8:
		return "hoch"
	case topScore >= 5:
		return "mittel"
	default:
		return "niedrig"
	}
}

// splitIntoSentences splits text at every newline — both "\n\n" (paragraph
// breaks, how BuildChunks joins paragraphs; see parasText in document.go)
// and a lone "\n" (a soft line-wrap PDF extraction leaves *within* one
// paragraph when two lines sit close enough vertically to read as the same
// paragraph — see TestLinesToTextUsesVerticalGaps) — and, within each piece,
// at '.', '!' and '?' — including a following closing quote/paren in the
// same sentence. Deliberately not the same boundary rule as document.go's
// isSentenceEnd (which also treats ':'/';' as endings, fine for a
// size-bounded chunk cut but not for excerpts meant to be read as standalone
// sentences). The newline split matters on its own: a numbered resolution is
// routinely one long colon-introduced clause with no '.'/'!'/'?' anywhere
// nearby, and without it that whole paragraph — often several hundred chars
// — would come back as a single "sentence".
func splitIntoSentences(text string) []string {
	var sentences []string
	for _, para := range strings.Split(text, "\n") {
		start := 0
		for i := 0; i < len(para); i++ {
			b := para[i]
			if b != '.' && b != '!' && b != '?' {
				continue
			}
			end := i + 1
			for end < len(para) && (para[end] == '"' || para[end] == '\'' || para[end] == ')' || para[end] == '»') {
				end++
			}
			if s := strings.TrimSpace(para[start:end]); s != "" {
				sentences = append(sentences, s)
			}
			start = end
		}
		if s := strings.TrimSpace(para[start:]); s != "" {
			sentences = append(sentences, s)
		}
	}
	return sentences
}

// contentWords lowercases text and tokenizes it into words of at least 4
// characters — short enough to skip most German articles/prepositions
// (der/die/das/und/mit/...), which would otherwise dominate an overlap score
// without indicating real topical similarity.
func contentWords(text string) map[string]bool {
	words := map[string]bool{}
	var cur strings.Builder
	flush := func() {
		if w := strings.ToLower(cur.String()); len(w) >= 4 {
			words[w] = true
		}
		cur.Reset()
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return words
}

// BestExcerpt returns the sentence in chunkText sharing the most content
// words with answer — a short, verbatim quote a citation can point to,
// instead of leaving the reader to search the whole cited page for what
// grounds the claim. Falls back to the chunk's first sentence when nothing
// scores above zero (e.g. the answer paraphrased everything), so a citation
// still carries some concrete text; returns "" only when chunkText has no
// sentences at all.
func BestExcerpt(chunkText, answer string) string {
	sentences := splitIntoSentences(chunkText)
	if len(sentences) == 0 {
		return ""
	}
	answerWords := contentWords(answer)

	best, bestScore := sentences[0], -1
	for _, s := range sentences {
		score := 0
		for w := range contentWords(s) {
			if answerWords[w] {
				score++
			}
		}
		if score > bestScore {
			best, bestScore = s, score
		}
	}

	const maxExcerptLen = 220
	if len(best) > maxExcerptLen {
		best = strings.TrimSpace(best[:maxExcerptLen]) + "…"
	}
	return best
}

// IsReliableCondense sanity-checks a condensed question against the
// follow-up and history it was built from. The condense prompt has been
// observed doing two kinds of damage that a plain empty-reply check misses:
// it can invent a date/number nowhere in the conversation, or it can drop a
// concrete anchor ("die 372. Sitzung am 7. Juli 2026") in favor of a
// vague placeholder ("die Veranstaltung") — syntactically fine, but useless
// for retrieval.
//
// The check only looks at digit tokens (extractNumberTokens), so it catches
// this specifically for dates and session/paragraph numbers — a hallucinated
// or dropped non-numeric anchor (a person, committee, or document name) goes
// undetected. Widening it to non-numeric tokens isn't a safe drop-in: German
// capitalizes every noun, not just proper nouns, so there's no cheap way to
// tell "invented name" from "ordinary paraphrase" the way a bare digit
// mismatch reliably does, and a naive check would flood on real paraphrases.
func IsReliableCondense(original, condensed string, history []QATurn) bool {
	if condensed == "" {
		return false
	}

	var histText strings.Builder
	for _, t := range history {
		histText.WriteString(t.Question)
		histText.WriteString(" ")
		histText.WriteString(t.Answer)
		histText.WriteString(" ")
	}
	histNumbers := extractNumberTokens(histText.String())
	originalNumbers := extractNumberTokens(original)

	knownNumbers := histNumbers
	for n := range originalNumbers {
		knownNumbers[n] = true
	}

	condensedNumbers := extractNumberTokens(condensed)
	for n := range condensedNumbers {
		if !knownNumbers[n] {
			return false // a number appears that's nowhere in the conversation
		}
	}

	// If the history carries concrete numbers the follow-up itself lacks, the
	// rewrite must carry at least one of them forward, or it likely
	// genericized away the referent instead of resolving it.
	if len(histNumbers) > 0 && len(originalNumbers) == 0 && !hasAny(condensedNumbers, histNumbers) {
		return false
	}
	return true
}

// FallbackStandaloneQuestion is used when the condense prompt is unavailable
// or its result doesn't pass IsReliableCondense. Plain concatenation of the
// last question with the new one is crude, but it keeps whatever concrete
// anchors (dates, session numbers, names) the last turn mentioned in the
// text that retrieval actually searches on — which a bare, unresolved
// follow-up like "Wer war dabei anwesend?" does not.
func FallbackStandaloneQuestion(question string, history []QATurn) string {
	if len(history) == 0 {
		return question
	}
	return history[len(history)-1].Question + " " + question
}

// CondenseFollowUp rewrites question into a standalone question using the
// conversation history, so retrieval doesn't have to resolve "und die
// davor?" on its own. condensePrompt is expected to take {"history",
// "question"} and return the rewritten question. Falls back to
// FallbackStandaloneQuestion whenever the prompt call fails or the result
// fails IsReliableCondense, and returns question unchanged when there is no
// history or condensePrompt is nil. Callers that make multi-turn optional
// check their own switch before calling.
func CondenseFollowUp(ctx context.Context, condensePrompt ai.Prompt, model ai.Model, config any, question string, history []QATurn) string {
	hist := TrimHistory(history)
	if len(hist) == 0 || condensePrompt == nil {
		return question
	}
	fallback := FallbackStandaloneQuestion(question, hist)

	resp, err := condensePrompt.Execute(ctx,
		PromptOptions(model, config, map[string]any{
			"history":  FormatHistory(hist),
			"question": question,
		})...)
	if err != nil || strings.TrimSpace(resp.Text()) == "" {
		log.Printf("[condense] failed, using %q instead (err=%v)", fallback, err)
		return fallback
	}

	standalone := strings.TrimSpace(resp.Text())
	if !IsReliableCondense(question, standalone, hist) {
		log.Printf("[condense] result %q looks unreliable vs. history — using %q instead", standalone, fallback)
		return fallback
	}
	log.Printf("[condense] follow-up %q (history=%d turns) -> %q", question, len(hist), standalone)
	return standalone
}
