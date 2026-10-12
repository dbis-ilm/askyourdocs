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
	"log"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/genkit"
)

// setupLoggingToFile runs SetupLogging into a temp file and restores the
// process-wide loggers afterwards. It returns a function reading the file.
func setupLoggingToFile(t *testing.T, cfg LogConfig) func() string {
	t.Helper()
	prev := slog.Default()
	t.Cleanup(func() {
		slog.SetDefault(prev)
		slog.SetLogLoggerLevel(slog.LevelInfo)
		log.SetOutput(os.Stderr)
		log.SetFlags(log.LstdFlags)
	})
	cfg.File = filepath.Join(t.TempDir(), "app.log")
	closer, err := SetupLogging(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { closer.Close() })
	return func() string {
		b, err := os.ReadFile(cfg.File)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
}

func TestSetupLoggingLevel(t *testing.T) {
	read := setupLoggingToFile(t, LogConfig{Level: "warn"})
	slog.Info("info-record")
	slog.Warn("warn-record")
	out := read()
	if strings.Contains(out, "info-record") {
		t.Error("info record logged at LOG_LEVEL=warn")
	}
	if !strings.Contains(out, "warn-record") {
		t.Errorf("warn record missing: %q", out)
	}
}

func TestSetupLoggingDebug(t *testing.T) {
	read := setupLoggingToFile(t, LogConfig{Level: "DEBUG"})
	slog.Debug("debug-record")
	if !strings.Contains(read(), "debug-record") {
		t.Error("debug record missing at LOG_LEVEL=debug")
	}
}

func TestSetupLoggingOff(t *testing.T) {
	read := setupLoggingToFile(t, LogConfig{Level: "off"})
	slog.Error("error-record")
	log.Print("log-package-record")
	if out := read(); out != "" {
		t.Errorf("LOG_LEVEL=off still logged %q", out)
	}
}

// What is left on the log package (log.Fatal at startup) must survive a
// level that drops info, so it is bridged at error level.
func TestSetupLoggingBridgesLogPackageAtError(t *testing.T) {
	read := setupLoggingToFile(t, LogConfig{Level: "error"})
	log.Print("startup failed")
	out := read()
	if !strings.Contains(out, "startup failed") || !strings.Contains(out, "level=ERROR") {
		t.Errorf("log package record not bridged at error level: %q", out)
	}
}

func TestSetupLoggingJSON(t *testing.T) {
	read := setupLoggingToFile(t, LogConfig{Format: "json"})
	slog.Info("json-record", "n", 3)
	var rec map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(read())), &rec); err != nil {
		t.Fatalf("not one JSON record: %v", err)
	}
	if rec["msg"] != "json-record" || rec["n"] != float64(3) {
		t.Errorf("record = %v", rec)
	}
}

func TestSetupLoggingAppends(t *testing.T) {
	prev := slog.Default()
	t.Cleanup(func() {
		slog.SetDefault(prev)
		slog.SetLogLoggerLevel(slog.LevelInfo)
		log.SetOutput(os.Stderr)
		log.SetFlags(log.LstdFlags)
	})
	file := filepath.Join(t.TempDir(), "app.log")
	if err := os.WriteFile(file, []byte("earlier run\n"), 0o640); err != nil {
		t.Fatal(err)
	}
	closer, err := SetupLogging(LogConfig{File: file})
	if err != nil {
		t.Fatal(err)
	}
	slog.Info("this run")
	closer.Close()
	b, _ := os.ReadFile(file)
	if !strings.HasPrefix(string(b), "earlier run\n") || !strings.Contains(string(b), "this run") {
		t.Errorf("log file was not appended to: %q", b)
	}
}

func TestSetupLoggingRejects(t *testing.T) {
	for _, cfg := range []LogConfig{
		{Level: "verbose"},
		{Format: "xml"},
		{File: filepath.Join(t.TempDir(), "missing-dir", "app.log")},
	} {
		if _, err := SetupLogging(cfg); err == nil {
			t.Errorf("SetupLogging(%+v) = nil error", cfg)
		}
	}
}

func TestLogConfigFromEnv(t *testing.T) {
	t.Setenv("LOG_LEVEL", "debug")
	t.Setenv("LOG_FILE", "/var/log/app.log")
	t.Setenv("LOG_FORMAT", "json")
	if got, want := LogConfigFromEnv(), (LogConfig{Level: "debug", File: "/var/log/app.log", Format: "json"}); got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

// Genkit's own records go through the same handler.
func TestSetupLoggingCoversGenkit(t *testing.T) {
	read := setupLoggingToFile(t, LogConfig{Level: "info"})
	genkit.Init(context.Background())
	if !strings.Contains(read(), "Genkit initialized") {
		t.Error("Genkit's startup record did not reach the log file")
	}
}

func TestSetupLoggingGenkitObeysLevel(t *testing.T) {
	read := setupLoggingToFile(t, LogConfig{Level: "warn"})
	genkit.Init(context.Background())
	if out := read(); strings.Contains(out, "Genkit initialized") {
		t.Errorf("Genkit's info record logged at LOG_LEVEL=warn: %q", out)
	}
}
