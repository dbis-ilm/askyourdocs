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
	"os"
	"path/filepath"
	"testing"
)

func TestGlobDocumentsFindsPDFAndDOCXOnly(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a.pdf", "b.docx", "c.txt", "d.PDF"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatalf("write fixture %s: %v", name, err)
		}
	}

	got, err := globDocuments(dir)
	if err != nil {
		t.Fatalf("globDocuments: %v", err)
	}

	var names []string
	for _, p := range got {
		names = append(names, filepath.Base(p))
	}
	want := []string{"a.pdf", "b.docx"}
	if len(names) != len(want) {
		t.Fatalf("got %v, want %v", names, want)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Errorf("names[%d] = %q, want %q", i, names[i], want[i])
		}
	}
}

func TestGlobDocumentsEmptyDir(t *testing.T) {
	got, err := globDocuments(t.TempDir())
	if err != nil {
		t.Fatalf("globDocuments: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want empty", got)
	}
}

func TestGlobDocumentsMalformedDirReturnsError(t *testing.T) {
	// An unbalanced "[" makes filepath.Glob's pattern invalid
	// (ErrBadPattern) rather than just matching nothing.
	if _, err := globDocuments("foo[bar"); err == nil {
		t.Error("globDocuments with a malformed dir pattern returned no error")
	}
}

func TestParseGeneratedQA(t *testing.T) {
	q, a, err := parseGeneratedQA(`{"question": "Wer ist zuständig?", "answer": "Der Prüfungsausschuss."}`)
	if err != nil {
		t.Fatalf("parseGeneratedQA: %v", err)
	}
	if q != "Wer ist zuständig?" || a != "Der Prüfungsausschuss." {
		t.Errorf("got (%q, %q)", q, a)
	}
}

func TestParseGeneratedQATrimsSurroundingText(t *testing.T) {
	q, a, err := parseGeneratedQA("Hier ist das Ergebnis:\n{\"question\": \"Q?\", \"answer\": \"A.\"}\nEnde.")
	if err != nil {
		t.Fatalf("parseGeneratedQA: %v", err)
	}
	if q != "Q?" || a != "A." {
		t.Errorf("got (%q, %q)", q, a)
	}
}

func TestParseGeneratedQANoJSON(t *testing.T) {
	if _, _, err := parseGeneratedQA("kein json hier"); err == nil {
		t.Error("expected an error when the response has no JSON object")
	}
}

func TestParseGeneratedQAEmptyFields(t *testing.T) {
	if _, _, err := parseGeneratedQA(`{"question": "", "answer": ""}`); err == nil {
		t.Error("expected an error for empty question/answer")
	}
}

func TestParseGeneratedQAMalformedJSON(t *testing.T) {
	if _, _, err := parseGeneratedQA(`{"question": "Q?", "answer":`); err == nil {
		t.Error("expected an error for truncated JSON")
	}
}
