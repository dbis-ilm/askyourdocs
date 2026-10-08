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
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type pdfTestLine struct {
	y    int
	text string
}

// buildTestPDF assembles a minimal valid PDF (Helvetica, one content stream
// per page) with a correct xref table, so readPDFPages can be tested against
// a real file without a binary fixture.
func buildTestPDF(pages [][]pdfTestLine) []byte {
	var buf bytes.Buffer
	var offsets []int
	add := func(body string) {
		offsets = append(offsets, buf.Len())
		fmt.Fprintf(&buf, "%d 0 obj\n%s\nendobj\n", len(offsets), body)
	}

	buf.WriteString("%PDF-1.4\n")
	// Object layout: 1 catalog, 2 pages, 3 font, then (page, content) pairs.
	add("<< /Type /Catalog /Pages 2 0 R >>")
	var kids []string
	for i := range pages {
		kids = append(kids, fmt.Sprintf("%d 0 R", 4+2*i))
	}
	add(fmt.Sprintf("<< /Type /Pages /Kids [%s] /Count %d >>", strings.Join(kids, " "), len(pages)))
	add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
	for i, lines := range pages {
		add(fmt.Sprintf("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R /Resources << /Font << /F1 3 0 R >> >> >>", 5+2*i))
		var content strings.Builder
		for _, l := range lines {
			fmt.Fprintf(&content, "BT /F1 12 Tf 72 %d Td (%s) Tj ET\n", l.y, l.text)
		}
		add(fmt.Sprintf("<< /Length %d >>\nstream\n%sendstream", content.Len(), content.String()))
	}

	xref := buf.Len()
	fmt.Fprintf(&buf, "xref\n0 %d\n0000000000 65535 f \n", len(offsets)+1)
	for _, off := range offsets {
		fmt.Fprintf(&buf, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&buf, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(offsets)+1, xref)
	return buf.Bytes()
}

func writeTestPDF(t *testing.T, pages [][]pdfTestLine) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "doc.pdf")
	if err := os.WriteFile(path, buildTestPDF(pages), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestSupportedDocumentExt(t *testing.T) {
	for ext, want := range map[string]bool{
		".pdf": true, ".PDF": true, ".docx": true, ".DocX": true,
		".txt": false, ".doc": false, "": false, "pdf": false,
	} {
		if got := SupportedDocumentExt(ext); got != want {
			t.Errorf("SupportedDocumentExt(%q) = %v, want %v", ext, got, want)
		}
	}
}

func TestReadPDFPagesExtractsTextPerPage(t *testing.T) {
	path := writeTestPDF(t, [][]pdfTestLine{
		{{700, "Erste Seite Zeile eins"}, {686, "Erste Seite Zeile zwei"}},
		{{700, "Zweite Seite Inhalt"}},
	})

	pages, err := readPDFPages(path)
	if err != nil {
		t.Fatalf("readPDFPages: %v", err)
	}
	if len(pages) != 2 {
		t.Fatalf("got %d pages, want 2: %+v", len(pages), pages)
	}
	if pages[0].PageNum != 1 || pages[1].PageNum != 2 {
		t.Errorf("page numbers = %d, %d; want 1, 2", pages[0].PageNum, pages[1].PageNum)
	}
	if !strings.Contains(pages[0].Text, "Zeile eins") || !strings.Contains(pages[0].Text, "Zeile zwei") {
		t.Errorf("page 1 text = %q", pages[0].Text)
	}
	if !strings.Contains(pages[1].Text, "Zweite Seite") {
		t.Errorf("page 2 text = %q", pages[1].Text)
	}
}

func TestReadPDFPagesSplitsParagraphsOnLargeGap(t *testing.T) {
	// Body spacing 14pt, then a 40pt jump: should become a blank line.
	path := writeTestPDF(t, [][]pdfTestLine{{
		{700, "Absatz A Zeile 1"}, {686, "Absatz A Zeile 2"}, {672, "Absatz A Zeile 3"},
		{632, "Absatz B Zeile 1"}, {618, "Absatz B Zeile 2"},
	}})

	pages, err := readPDFPages(path)
	if err != nil || len(pages) != 1 {
		t.Fatalf("readPDFPages = %+v, %v", pages, err)
	}
	if !strings.Contains(pages[0].Text, "\n\n") {
		t.Errorf("expected a paragraph break in %q", pages[0].Text)
	}
}

func TestReadPDFPagesSkipsEmptyPages(t *testing.T) {
	path := writeTestPDF(t, [][]pdfTestLine{
		{{700, "Inhalt"}},
		{},
	})
	pages, err := readPDFPages(path)
	if err != nil {
		t.Fatalf("readPDFPages: %v", err)
	}
	if len(pages) != 1 || pages[0].PageNum != 1 {
		t.Errorf("pages = %+v, want only page 1", pages)
	}
}

func TestReadPDFPagesErrors(t *testing.T) {
	if _, err := readPDFPages(filepath.Join(t.TempDir(), "missing.pdf")); err == nil {
		t.Error("expected error for missing file")
	}
	bad := filepath.Join(t.TempDir(), "bad.pdf")
	if err := os.WriteFile(bad, []byte("this is not a pdf"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readPDFPages(bad); err == nil {
		t.Error("expected error for non-PDF content")
	}
}

func TestExtractParagraphsDispatch(t *testing.T) {
	path := writeTestPDF(t, [][]pdfTestLine{{{700, "Ein Absatz im PDF."}}})
	paras, err := ExtractParagraphs(path)
	if err != nil {
		t.Fatalf("ExtractParagraphs(pdf): %v", err)
	}
	if len(paras) == 0 || !strings.Contains(paras[0].Text, "Ein Absatz") || paras[0].PageNum != 1 {
		t.Errorf("paras = %+v", paras)
	}

	if _, err := ExtractParagraphs("notes.txt"); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Errorf("ExtractParagraphs(txt) error = %v, want unsupported", err)
	}
	if _, err := ExtractParagraphs(filepath.Join(t.TempDir(), "missing.pdf")); err == nil {
		t.Error("expected error for missing pdf")
	}
}
