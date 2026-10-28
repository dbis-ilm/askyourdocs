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
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"

	"github.com/firebase/genkit/go/core/logger"
)

// LogConfig selects where log records go and which of them are kept.
type LogConfig struct {
	// Level is debug, info, warn, error or off; empty means info. Request
	// content (questions, mail text) is only ever logged at debug.
	Level string
	// File is appended to; empty means stderr. The file is not rotated —
	// under Docker log to stderr and let the logging driver rotate, otherwise
	// use logrotate with copytruncate.
	File string
	// Format is text or json; empty means text.
	Format string
}

// LogConfigFromEnv reads LOG_LEVEL, LOG_FILE and LOG_FORMAT.
func LogConfigFromEnv() LogConfig {
	return LogConfig{
		Level:  os.Getenv("LOG_LEVEL"),
		File:   os.Getenv("LOG_FILE"),
		Format: os.Getenv("LOG_FORMAT"),
	}
}

// levelOff is above every level anything logs at.
const levelOff = slog.Level(1 << 30)

func parseLogLevel(s string) (slog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "info":
		return slog.LevelInfo, nil
	case "debug":
		return slog.LevelDebug, nil
	case "warn", "warning":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	case "off", "none":
		return levelOff, nil
	}
	return 0, fmt.Errorf("LOG_LEVEL must be debug, info, warn, error or off, got %q", s)
}

type nopCloser struct{}

func (nopCloser) Close() error { return nil }

// SetupLogging installs the process-wide slog handler cfg describes. Call it
// first thing in main, before genkit.Init: the handler goes in through
// Genkit's logger package, so Genkit's own records obey the same level and
// destination, and dev-mode streaming to the Dev UI keeps working.
//
// What is still logged through the standard log package — in practice
// log.Fatal on a startup error — is routed through the same handler at
// error level, so it lands in the log file and survives LOG_LEVEL=warn or
// error; only off silences it.
//
// The returned Closer closes the log file; for stderr it does nothing.
func SetupLogging(cfg LogConfig) (io.Closer, error) {
	level, err := parseLogLevel(cfg.Level)
	if err != nil {
		return nil, err
	}

	var w io.Writer = os.Stderr
	var closer io.Closer = nopCloser{}
	if cfg.File != "" {
		f, err := os.OpenFile(cfg.File, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o640)
		if err != nil {
			return nil, fmt.Errorf("LOG_FILE: %w", err)
		}
		w, closer = f, f
	}

	opts := &slog.HandlerOptions{Level: level}
	var h slog.Handler
	switch strings.ToLower(cfg.Format) {
	case "", "text":
		h = slog.NewTextHandler(w, opts)
	case "json":
		h = slog.NewJSONHandler(w, opts)
	default:
		closer.Close()
		return nil, fmt.Errorf("LOG_FORMAT must be text or json, got %q", cfg.Format)
	}

	logger.SetDefaultHandler(h)
	slog.SetLogLoggerLevel(slog.LevelError)
	return closer, nil
}
