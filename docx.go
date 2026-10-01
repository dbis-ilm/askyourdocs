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
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"strings"
)

// readDOCXParagraphs extracts paragraph text from a .docx file's
// word/document.xml. A .docx is a zip archive; the document body is plain
// OOXML, which stdlib's archive/zip and encoding/xml handle without a third
// party dependency.
func readDOCXParagraphs(path string) ([]string, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open docx: %w", err)
	}
	defer r.Close()

	var docXML *zip.File
	for _, f := range r.File {
		if f.Name == "word/document.xml" {
			docXML = f
			break
		}
	}
	if docXML == nil {
		return nil, fmt.Errorf("not a valid .docx: missing word/document.xml")
	}

	rc, err := docXML.Open()
	if err != nil {
		return nil, fmt.Errorf("open word/document.xml: %w", err)
	}
	defer rc.Close()

	data, err := io.ReadAll(rc)
	if err != nil {
		return nil, fmt.Errorf("read word/document.xml: %w", err)
	}

	return parseDOCXParagraphs(data)
}

// parseDOCXParagraphs walks document.xml's tokens, collecting the text of
// each <w:p> paragraph from its <w:t> runs. Table cells are themselves <w:p>
// paragraphs in OOXML, so this naturally yields one paragraph per cell too.
// Matching is on the local element name only (ignoring the "w:" namespace
// prefix), which holds for every .docx actually produced by Word or
// LibreOffice.
func parseDOCXParagraphs(data []byte) ([]string, error) {
	dec := xml.NewDecoder(bytes.NewReader(data))
	var paragraphs []string
	var current strings.Builder
	inParagraph := false
	inText := false

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("parse document.xml: %w", err)
		}

		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "p":
				inParagraph = true
				current.Reset()
			case "t":
				inText = true
			case "tab":
				if inParagraph {
					current.WriteString("\t")
				}
			case "br", "cr":
				if inParagraph {
					current.WriteString("\n")
				}
			}
		case xml.CharData:
			if inText {
				current.Write(t)
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				inText = false
			case "p":
				inParagraph = false
				text := strings.TrimSpace(current.String())
				if text != "" {
					paragraphs = append(paragraphs, text)
				}
			}
		}
	}
	return paragraphs, nil
}

// parseDOCXIntoParagraphs turns extracted .docx paragraphs into the same
// []Paragraph shape BuildChunks expects, reusing the section-boundary
// splitting and heading detection already tuned for PDF text — both are
// generic textual patterns ("TOP N", "§ 12", lettered/roman headings), not
// PDF-specific. PageNum is left at 0: Word has no fixed pagination in the
// XML (it paginates at render time), so pageRange() treats 0 as "no page
// info" and citations for these chunks omit "Seite" entirely.
func parseDOCXIntoParagraphs(paragraphs []string) []Paragraph {
	var paras []Paragraph
	for _, block := range paragraphs {
		trimmed := strings.TrimSpace(block)
		if trimmed == "" {
			continue
		}
		for _, sub := range splitOnSectionBoundaries(trimmed) {
			sub = strings.TrimSpace(sub)
			if sub == "" {
				continue
			}
			paras = append(paras, Paragraph{
				Text:    sub,
				PageNum: 0,
				Heading: isHeading(sub),
			})
		}
	}
	return paras
}
