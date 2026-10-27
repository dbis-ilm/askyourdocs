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

// Package engine is a small, reusable RAG engine for "ask your documents"
// apps: it turns PDF, .docx and web pages into retrievable chunks, indexes
// them in a vector store (file-based LocalvecStore or PostgreSQL-backed
// PgvecStore), and offers the retrieval pieces around that core — keyword
// search, Reciprocal Rank Fusion, reranking, HyDE — plus domain-neutral app
// building blocks: an async job queue, a feedback log, multi-turn helpers,
// prompt-injection hardening and evaluation helpers.
//
// The module path is github.com/dbis-ilm/askyourdocs but the package name is
// engine, so import it with that name spelled out:
//
//	import engine "github.com/dbis-ilm/askyourdocs"
//
// The design boundary: generic RAG mechanics live here; anything
// app-specific — what metadata a chunk carries, prompt wording, auth, admin
// UI — stays in the consuming app. Stores satisfy VectorStore structurally,
// so an app embeds LocalvecStore or PgvecStore and adds or shadows methods
// (filtered Retrieve, KeywordSearch) on its own wrapper type.
//
// See the README for how the pieces fit together and why several of the
// trickier API decisions were made the way they were.
package engine
