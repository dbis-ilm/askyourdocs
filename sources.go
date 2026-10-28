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

package askyourdocs

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// MarshalJSON writes s as one flat object: url and source when set, chunks,
// and each non-empty Metadata entry as a field of its own, e.g.
// {"source":"a.pdf","tag":"Senat","datum":"2026-01-13","chunks":3}.
func (s Source) MarshalJSON() ([]byte, error) {
	m := make(map[string]any, 3+len(s.Metadata))
	for k, v := range s.Metadata {
		if v != "" && !reservedSourceKeys[k] {
			m[k] = v
		}
	}
	if s.URL != "" {
		m["url"] = s.URL
	}
	if s.Source != "" {
		m["source"] = s.Source
	}
	m["chunks"] = s.Chunks
	return json.Marshal(m)
}

// UnmarshalJSON reads what MarshalJSON writes, so an admin UI can send a
// source from a Sources listing back unchanged to UpdateMetadata or
// DeleteSource. Every field other than url, source and chunks goes into
// Metadata; it must be a string or null, and empty ones are left out.
func (s *Source) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*s = Source{}
	for k, v := range raw {
		var err error
		switch k {
		case "url":
			err = json.Unmarshal(v, &s.URL)
		case "source":
			err = json.Unmarshal(v, &s.Source)
		case "chunks":
			err = json.Unmarshal(v, &s.Chunks)
		default:
			var str *string
			if err = json.Unmarshal(v, &str); err == nil && str != nil && *str != "" {
				if s.Metadata == nil {
					s.Metadata = map[string]string{}
				}
				s.Metadata[k] = *str
			}
		}
		if err != nil {
			return fmt.Errorf("source.%s: %w", k, err)
		}
	}
	return nil
}

// reservedSourceKeys are Source's own JSON fields, which a metadata key
// would collide with.
var reservedSourceKeys = map[string]bool{"url": true, "source": true, "chunks": true}

// sourceMetaKeyName matches the metadata keys a store may group by. They are
// interpolated into SQL text by PgvecStore, so only plain names are allowed.
var sourceMetaKeyName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// checkSourceMetaKeys validates the keys given to WithSourceMetaKeys or
// WithLocalvecSourceMetaKeys.
func checkSourceMetaKeys(keys []string) error {
	seen := map[string]bool{}
	for _, k := range keys {
		switch {
		case !sourceMetaKeyName.MatchString(k):
			return fmt.Errorf("source metadata key %q: must be a plain name (letters, digits, underscore)", k)
		case reservedSourceKeys[k]:
			return fmt.Errorf("source metadata key %q: reserved", k)
		case seen[k]:
			return fmt.Errorf("source metadata key %q: given twice", k)
		}
		seen[k] = true
	}
	return nil
}

// metaString returns meta[key] if it is a string, "" otherwise.
func metaString(meta map[string]any, key string) string {
	s, _ := meta[key].(string)
	return s
}

// sourceOf returns the source group a chunk with metadata meta belongs to,
// and false for a chunk with no url, source or any of keys.
func sourceOf(meta map[string]any, keys []string) (Source, bool) {
	src := Source{URL: metaString(meta, "url"), Source: metaString(meta, "source")}
	found := src.URL != "" || src.Source != ""
	for _, k := range keys {
		if v := metaString(meta, k); v != "" {
			if src.Metadata == nil {
				src.Metadata = make(map[string]string, len(keys))
			}
			src.Metadata[k] = v
			found = true
		}
	}
	return src, found
}

// inSource reports whether a chunk with metadata meta belongs to exactly
// the group src, compared on url, source and every key in keys. A key
// missing from either side counts as "".
func inSource(meta map[string]any, src Source, keys []string) bool {
	if metaString(meta, "url") != src.URL || metaString(meta, "source") != src.Source {
		return false
	}
	for _, k := range keys {
		if metaString(meta, k) != src.Metadata[k] {
			return false
		}
	}
	return true
}

// sourceGroupKey encodes a group's identity as a comparable map key.
func sourceGroupKey(src Source, keys []string) string {
	parts := make([]string, 0, 2+len(keys))
	parts = append(parts, src.URL, src.Source)
	for _, k := range keys {
		parts = append(parts, src.Metadata[k])
	}
	return strings.Join(parts, "\x00")
}
