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
	"reflect"
	"strings"
	"testing"
)

func TestTrimHistoryKeepsRecentTurnsOnly(t *testing.T) {
	history := make([]QATurn, 10)
	for i := range history {
		history[i] = QATurn{Question: string(rune('a' + i))}
	}

	got := TrimHistory(history)
	if len(got) != MaxHistoryTurns {
		t.Fatalf("len = %d, want %d", len(got), MaxHistoryTurns)
	}
	// The kept turns must be the trailing ones, in order.
	want := history[len(history)-MaxHistoryTurns:]
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("turn %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestFallbackStandaloneQuestionNoHistory(t *testing.T) {
	got := FallbackStandaloneQuestion("Wer war dabei?", nil)
	if got != "Wer war dabei?" {
		t.Errorf("FallbackStandaloneQuestion = %q, want the question unchanged", got)
	}
}

func TestFallbackStandaloneQuestionPrependsLastQuestion(t *testing.T) {
	history := []QATurn{
		{Question: "Was wurde am 13.01.2026 beschlossen?", Answer: "..."},
		{Question: "Und davor?", Answer: "..."},
	}
	got := FallbackStandaloneQuestion("Wer war dabei?", history)
	want := "Und davor? Wer war dabei?"
	if got != want {
		t.Errorf("FallbackStandaloneQuestion = %q, want %q", got, want)
	}
}

func TestTrimHistoryShorterThanLimitIsUnchanged(t *testing.T) {
	history := []QATurn{{Question: "Q1", Answer: "A1"}, {Question: "Q2", Answer: "A2"}}
	got := TrimHistory(history)
	if len(got) != 2 || got[0] != history[0] || got[1] != history[1] {
		t.Errorf("TrimHistory changed a short history: %+v", got)
	}
}

func TestFormatHistory(t *testing.T) {
	history := []QATurn{
		{Question: "Wann war die letzte Sitzung?", Answer: "Am 7. Juli 2026."},
		{Question: "Wer war anwesend?", Answer: "12 Mitglieder."},
	}
	got := FormatHistory(history)
	want := "Nutzer: Wann war die letzte Sitzung?\nAssistent: Am 7. Juli 2026.\n\nNutzer: Wer war anwesend?\nAssistent: 12 Mitglieder."
	if got != want {
		t.Errorf("FormatHistory =\n%q\nwant\n%q", got, want)
	}
}

func TestFormatHistoryEmpty(t *testing.T) {
	if got := FormatHistory(nil); got != "" {
		t.Errorf("FormatHistory(nil) = %q, want empty", got)
	}
}

func TestCondenseFollowUpSkipsWithoutHistory(t *testing.T) {
	fake := &fakePrompt{reply: "should never be used"}

	got := CondenseFollowUp(context.Background(), fake, nil, nil, "Und wer war dabei?", nil)
	if got != "Und wer war dabei?" {
		t.Errorf("got %q, want the question unchanged", got)
	}
	if fake.calls != 0 {
		t.Errorf("condense prompt called %d times, want 0 without history", fake.calls)
	}
}

func TestCondenseFollowUpSkipsWithoutPrompt(t *testing.T) {
	history := []QATurn{{Question: "Wann war die letzte Sitzung?", Answer: "Am 7. Juli 2026."}}

	got := CondenseFollowUp(context.Background(), nil, nil, nil, "Und wer war dabei?", history)
	if got != "Und wer war dabei?" {
		t.Errorf("got %q, want the question unchanged when no condense prompt is configured", got)
	}
}

func TestCondenseFollowUpRewritesWithHistory(t *testing.T) {
	fake := &fakePrompt{reply: "Wer war bei der Senatssitzung am 7. Juli 2026 anwesend?"}
	history := []QATurn{{Question: "Wann war die letzte Sitzung?", Answer: "Am 7. Juli 2026."}}

	got := CondenseFollowUp(context.Background(), fake, nil, nil, "Und wer war dabei?", history)
	if got != "Wer war bei der Senatssitzung am 7. Juli 2026 anwesend?" {
		t.Errorf("got %q, want the rewritten standalone question", got)
	}
	if fake.calls != 1 {
		t.Errorf("condense prompt called %d times, want 1", fake.calls)
	}
}

func TestCondenseFollowUpFallsBackOnFailure(t *testing.T) {
	fake := &fakePrompt{failCall: 1}
	history := []QATurn{{Question: "Q1", Answer: "A1"}}

	got := CondenseFollowUp(context.Background(), fake, nil, nil, "Und dann?", history)
	if got != "Q1 Und dann?" {
		t.Errorf("got %q, want the last-question-concat fallback when the condense call fails", got)
	}
}

func TestCondenseFollowUpFallsBackOnEmptyReply(t *testing.T) {
	fake := &fakePrompt{reply: ""}
	history := []QATurn{{Question: "Q1", Answer: "A1"}}

	got := CondenseFollowUp(context.Background(), fake, nil, nil, "Und dann?", history)
	if got != "Q1 Und dann?" {
		t.Errorf("got %q, want the last-question-concat fallback when the condense reply is empty", got)
	}
}

func TestCondenseFollowUpRejectsHallucinatedNumber(t *testing.T) {
	// Nothing in the history or follow-up mentions 2023 — a rewrite that
	// invents it is worse than no rewrite at all.
	fake := &fakePrompt{reply: "Wer war anwesend bei der Veranstaltung am 15. April 2023?"}
	history := []QATurn{{Question: "Wann fand die 372. Senatssitzung statt?", Answer: "Am 7. Juli 2026."}}

	got := CondenseFollowUp(context.Background(), fake, nil, nil, "Wer war dabei anwesend?", history)
	want := "Wann fand die 372. Senatssitzung statt? Wer war dabei anwesend?"
	if got != want {
		t.Errorf("got %q, want fallback %q for a hallucinated number", got, want)
	}
}

func TestCondenseFollowUpRejectsAnchorLoss(t *testing.T) {
	// The history has concrete anchors (372, 2026-07-07) the follow-up itself
	// lacks; a rewrite that carries none of them forward genericized away the
	// referent instead of resolving it.
	fake := &fakePrompt{reply: "Wer war bei der Veranstaltung anwesend?"}
	history := []QATurn{{Question: "Wann fand die 372. Senatssitzung statt?", Answer: "Am 7. Juli 2026."}}

	got := CondenseFollowUp(context.Background(), fake, nil, nil, "Wer war dabei anwesend?", history)
	want := "Wann fand die 372. Senatssitzung statt? Wer war dabei anwesend?"
	if got != want {
		t.Errorf("got %q, want fallback %q when the rewrite drops every concrete anchor", got, want)
	}
}

func TestCondenseFollowUpAcceptsFollowUpWithItsOwnNumber(t *testing.T) {
	// The follow-up already names its own session number, so it needs no
	// anchor carried over from history — the carry-forward check must not
	// fire here even though the rewrite mentions no history numbers.
	fake := &fakePrompt{reply: "Wer war bei der 380. Senatssitzung anwesend?"}
	history := []QATurn{{Question: "Wann fand die 372. Senatssitzung statt?", Answer: "Am 7. Juli 2026."}}

	got := CondenseFollowUp(context.Background(), fake, nil, nil, "Wer war bei der 380. dabei?", history)
	if got != "Wer war bei der 380. Senatssitzung anwesend?" {
		t.Errorf("got %q, want the rewrite accepted since the follow-up already carries its own anchor", got)
	}
}

func TestIsReliableCondenseNoHistoryNumbersAlwaysPasses(t *testing.T) {
	// Nothing to anchor to and nothing hallucinated — a paraphrase is fine.
	if !IsReliableCondense("Und was noch?", "Was gibt es sonst noch zu berichten?", []QATurn{{Question: "Gibt es Neuigkeiten?", Answer: "Ja, einige."}}) {
		t.Error("expected a numberless rewrite of a numberless conversation to pass")
	}
}

func TestBestExcerptPicksSentenceOverlappingTheAnswer(t *testing.T) {
	chunk := "Die Sitzung begann um 14 Uhr. Der Senat beschließt einstimmig die Änderung der Prüfungsordnung. Anschließend wurde über sonstiges gesprochen."
	answer := "Der Senat hat die Änderung der Prüfungsordnung einstimmig beschlossen."

	got := BestExcerpt(chunk, answer)
	want := "Der Senat beschließt einstimmig die Änderung der Prüfungsordnung."
	if got != want {
		t.Errorf("BestExcerpt = %q, want %q", got, want)
	}
}

func TestBestExcerptFallsBackToFirstSentenceWithoutOverlap(t *testing.T) {
	chunk := "Die Katze saß auf der Fensterbank. Draußen regnete es."
	answer := "Etwas völlig anderes, das mit dem Chunk nichts zu tun hat."

	got := BestExcerpt(chunk, answer)
	want := "Die Katze saß auf der Fensterbank."
	if got != want {
		t.Errorf("BestExcerpt = %q, want %q (first sentence as fallback)", got, want)
	}
}

func TestBestExcerptEmptyChunkReturnsEmpty(t *testing.T) {
	if got := BestExcerpt("", "Eine Antwort."); got != "" {
		t.Errorf("BestExcerpt(\"\", ...) = %q, want \"\"", got)
	}
}

func TestBestExcerptTruncatesLongSentences(t *testing.T) {
	long := strings.Repeat("Modulprüfung ", 30) + "abgeschlossen."
	got := BestExcerpt(long, "Modulprüfung abgeschlossen.")
	if len(got) > 230 { // maxExcerptLen(220) + "…" + slack for the rune
		t.Errorf("BestExcerpt returned %d chars, want it truncated near 220", len(got))
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("BestExcerpt = %q, want it to end with the truncation marker", got)
	}
}

func TestSplitIntoSentences(t *testing.T) {
	got := splitIntoSentences(`Erster Satz. "Zweiter Satz!" Und ein dritter?`)
	want := []string{"Erster Satz.", `"Zweiter Satz!"`, "Und ein dritter?"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("splitIntoSentences = %#v, want %#v", got, want)
	}
}

// A colon-introduced resolution clause routinely has no '.'/'!'/'?' for a
// while — real example that surfaced this in a live protocol PDF: a TOP
// heading plus a whole "...übernimmt:" preamble came back as one 280+ char
// "sentence" and got truncated by BestExcerpt, at which point the excerpt no
// longer matched anything on the actual page. Splitting on paragraph breaks
// too (not just terminal punctuation) keeps each piece near actual sentence
// length even when punctuation alone wouldn't cut it in time.
func TestSplitIntoSentencesAlsoBreaksOnParagraphs(t *testing.T) {
	text := "TOP 4\n\nDer Tagesordnungspunkt wurde vorbereitet und es besteht kein Redebedarf bezüglich der Beschlussempfehlung, welche der Senat damit übernimmt:\n\nDer Senat beschließt einstimmig die Änderung."
	got := splitIntoSentences(text)
	want := []string{
		"TOP 4",
		"Der Tagesordnungspunkt wurde vorbereitet und es besteht kein Redebedarf bezüglich der Beschlussempfehlung, welche der Senat damit übernimmt:",
		"Der Senat beschließt einstimmig die Änderung.",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("splitIntoSentences = %#v, want %#v", got, want)
	}
}

// Real excerpt observed live against an actual indexed protocol: the
// separator there turned out to be a single "\n" (a soft line-wrap within
// one paragraph — see TestLinesToTextUsesVerticalGaps), not "\n\n" — the
// paragraph-only split above didn't catch it, and BestExcerpt returned a
// 200+ char run spanning "TOP 4" through the start of the actual resolution
// text, truncated mid-word by its 220-char cap.
func TestSplitIntoSentencesBreaksOnSingleNewlineToo(t *testing.T) {
	text := "TOP 4\nDer Tagesordnungspunkt wurde durch den Studienausschuss vorbereitet und es besteht kein\nRedebedarf bezüglich der Beschlussempfehlung, welche der Senat damit als Senatsbeschluss übernimmt:\nDer Senat beschließt einstimmig."
	got := splitIntoSentences(text)
	want := []string{
		"TOP 4",
		"Der Tagesordnungspunkt wurde durch den Studienausschuss vorbereitet und es besteht kein",
		"Redebedarf bezüglich der Beschlussempfehlung, welche der Senat damit als Senatsbeschluss übernimmt:",
		"Der Senat beschließt einstimmig.",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("splitIntoSentences = %#v, want %#v", got, want)
	}
}

func TestConfidenceLabel(t *testing.T) {
	cases := []struct {
		name         string
		topScore     int
		usedFallback bool
		want         string
	}{
		{"top score", 10, false, "hoch"},
		{"just above hoch threshold", 8, false, "hoch"},
		{"just below hoch threshold", 7, false, "mittel"},
		{"neutral default score", 5, false, "mittel"},
		{"just below mittel threshold", 4, false, "niedrig"},
		{"zero score", 0, false, "niedrig"},
		{"fallback overrides a high score", 10, true, "niedrig"},
	}
	for _, c := range cases {
		if got := ConfidenceLabel(c.topScore, c.usedFallback); got != c.want {
			t.Errorf("%s: ConfidenceLabel(%d, %v) = %q, want %q", c.name, c.topScore, c.usedFallback, got, c.want)
		}
	}
}
