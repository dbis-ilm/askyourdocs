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

func TestGlobDocumentsFindsPDFDOCXAndMDOnly(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a.pdf", "b.docx", "c.md", "d.txt", "e.PDF"} {
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
	want := []string{"a.pdf", "b.docx", "c.md"}
	if len(names) != len(want) {
		t.Fatalf("got %v, want %v", names, want)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Errorf("names[%d] = %q, want %q", i, names[i], want[i])
		}
	}
}

func TestReadEvalSourceParsesCrawlMarkdown(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "page.md")
	content := "---\nurl: https://example.com/page\ncrawled: 2026-01-01T00:00:00Z\n---\n\n## Heading\n\nBody paragraph.\n\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	paras, err := readEvalSource(path)
	if err != nil {
		t.Fatalf("readEvalSource: %v", err)
	}
	if len(paras) != 2 {
		t.Fatalf("got %d paragraphs, want 2: %+v", len(paras), paras)
	}
	if !paras[0].Heading || paras[0].Text != "Heading" {
		t.Errorf("paras[0] = %+v, want heading %q", paras[0], "Heading")
	}
	if paras[1].Heading || paras[1].Text != "Body paragraph." {
		t.Errorf("paras[1] = %+v, want body %q", paras[1], "Body paragraph.")
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
