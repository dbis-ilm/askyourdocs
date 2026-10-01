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
	"archive/zip"
	"os"
	"path/filepath"
	"testing"
)

func TestParseDOCXParagraphsJoinsRunsWithinAParagraph(t *testing.T) {
	xml := `<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>World</w:t></w:r></w:p>
    <w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p>
  </w:body>
</w:document>`
	got, err := parseDOCXParagraphs([]byte(xml))
	if err != nil {
		t.Fatalf("parseDOCXParagraphs: %v", err)
	}
	want := []string{"Hello World", "Second paragraph."}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("paragraph %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestParseDOCXParagraphsSkipsEmptyParagraphs(t *testing.T) {
	xml := `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:jc w:val="center"/></w:pPr></w:p>
    <w:p><w:r><w:t>Actual text</w:t></w:r></w:p>
  </w:body>
</w:document>`
	got, err := parseDOCXParagraphs([]byte(xml))
	if err != nil {
		t.Fatalf("parseDOCXParagraphs: %v", err)
	}
	if len(got) != 1 || got[0] != "Actual text" {
		t.Fatalf("got %v, want [\"Actual text\"]", got)
	}
}

func TestParseDOCXParagraphsHandlesTabAndBreak(t *testing.T) {
	xml := `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Col1</w:t></w:r><w:r><w:tab/><w:t>Col2</w:t></w:r></w:p>
  </w:body>
</w:document>`
	got, err := parseDOCXParagraphs([]byte(xml))
	if err != nil {
		t.Fatalf("parseDOCXParagraphs: %v", err)
	}
	want := "Col1\tCol2"
	if len(got) != 1 || got[0] != want {
		t.Fatalf("got %v, want [%q]", got, want)
	}
}

func TestParseDOCXParagraphsRejectsMalformedXML(t *testing.T) {
	_, err := parseDOCXParagraphs([]byte("not xml at all <<<"))
	if err == nil {
		t.Fatal("expected an error for malformed XML")
	}
}

// writeTestDOCX builds a minimal but real .docx (a zip archive containing
// word/document.xml) so readDOCXParagraphs can be exercised end to end.
func writeTestDOCX(t *testing.T, path string, paragraphs ...string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	defer f.Close()

	zw := zip.NewWriter(f)
	w, err := zw.Create("word/document.xml")
	if err != nil {
		t.Fatalf("create zip entry: %v", err)
	}

	body := `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>`
	for _, p := range paragraphs {
		body += `<w:p><w:r><w:t>` + p + `</w:t></w:r></w:p>`
	}
	body += `</w:body></w:document>`

	if _, err := w.Write([]byte(body)); err != nil {
		t.Fatalf("write zip entry: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close zip writer: %v", err)
	}
}

func TestReadDOCXParagraphsRoundTrips(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.docx")
	writeTestDOCX(t, path, "First paragraph.", "Second paragraph.")

	got, err := readDOCXParagraphs(path)
	if err != nil {
		t.Fatalf("readDOCXParagraphs: %v", err)
	}
	want := []string{"First paragraph.", "Second paragraph."}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("paragraph %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestReadDOCXParagraphsRejectsNonZipFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "not-a-docx.docx")
	if err := os.WriteFile(path, []byte("plain text, not a zip"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	if _, err := readDOCXParagraphs(path); err == nil {
		t.Fatal("expected an error for a non-zip file")
	}
}

func TestReadDOCXParagraphsRejectsZipWithoutDocumentXML(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.docx")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	zw := zip.NewWriter(f)
	if _, err := zw.Create("word/other.xml"); err != nil {
		t.Fatalf("create zip entry: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close zip writer: %v", err)
	}
	f.Close()

	if _, err := readDOCXParagraphs(path); err == nil {
		t.Fatal("expected an error when word/document.xml is missing")
	}
}

func TestParseDOCXIntoParagraphsSetsNoPageNum(t *testing.T) {
	paras := parseDOCXIntoParagraphs([]string{"Ein normaler Absatz.", "TOP 3 Ein Tagesordnungspunkt"})
	if len(paras) != 2 {
		t.Fatalf("got %d paragraphs, want 2", len(paras))
	}
	for _, p := range paras {
		if p.PageNum != 0 {
			t.Errorf("paragraph %+v has PageNum != 0", p)
		}
	}
	if !paras[1].Heading {
		t.Errorf("paragraph %+v: expected a TOP heading to be detected", paras[1])
	}
}

func TestParseDOCXIntoParagraphsSkipsBlank(t *testing.T) {
	paras := parseDOCXIntoParagraphs([]string{"", "   ", "Real text"})
	if len(paras) != 1 || paras[0].Text != "Real text" {
		t.Fatalf("got %+v, want a single paragraph \"Real text\"", paras)
	}
}

func TestExtractParagraphsDispatchesOnExtension(t *testing.T) {
	dir := t.TempDir()

	docxPath := filepath.Join(dir, "doc.docx")
	writeTestDOCX(t, docxPath, "From docx.")
	got, err := ExtractParagraphs(docxPath)
	if err != nil {
		t.Fatalf("ExtractParagraphs(.docx): %v", err)
	}
	if len(got) != 1 || got[0].Text != "From docx." {
		t.Fatalf("got %+v, want a single paragraph \"From docx.\"", got)
	}

	badPath := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(badPath, []byte("hello"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	if _, err := ExtractParagraphs(badPath); err == nil {
		t.Fatal("expected an error for an unsupported extension")
	}

	pdfPath := filepath.Join(dir, "broken.pdf")
	if err := os.WriteFile(pdfPath, []byte("not a real pdf"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	if _, err := ExtractParagraphs(pdfPath); err == nil {
		t.Fatal("expected an error for a .pdf that isn't actually a valid PDF (confirms the .pdf branch ran)")
	}
}
