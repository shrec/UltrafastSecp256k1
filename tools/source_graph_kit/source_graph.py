#!/usr/bin/env python3
"""
Source Graph Universal — Multi-language SQLite-based codebase index.

Supported languages: C++, Python, TypeScript/JavaScript, Rust, Go, Java, C#

Usage:
    python source_graph.py build              # Full rebuild of the database
    python source_graph.py build -i           # Incremental build (changed files only)
    python source_graph.py find <term>        # Search everything (FTS5 full-text)
    python source_graph.py singleton <name>   # Find singleton by macro or class
    python source_graph.py file <pattern>     # Find files by name pattern
    python source_graph.py handler <headcode> # Find packet handler
    python source_graph.py event <name>       # Find event/dungeon system
    python source_graph.py method <name>      # Find method by name
    python source_graph.py player <domain>    # Find Player partial file by domain
    python source_graph.py ai <target>        # Find AI handler
    python source_graph.py const <name>       # Find constant/define
    python source_graph.py config <name>      # Find config XML
    python source_graph.py deps <class>       # Find what depends on a class
    python source_graph.py enum <name>        # Find enum by name or value
    python source_graph.py struct <name>      # Find struct/packet layout
    python source_graph.py class <name>       # Find class hierarchy (parent/children)
    python source_graph.py todo [filter]      # Find TODO/FIXME/HACK comments
    python source_graph.py dbtable <name>     # Find database table usage in queries
    python source_graph.py query <name>       # Find prepared statement by QUERY_ name
    python source_graph.py cfgkey <name>      # Find config key usage (sConfig->GetXxx)
    python source_graph.py define <name>      # Find #define macros by name or value
    python source_graph.py complexity         # Show largest files by line count
    python source_graph.py leaks              # Memory leak risk analysis
    python source_graph.py crashes [file]     # Crash risk patterns
    python source_graph.py nullrisks [file]   # NULL-check risk analysis
    python source_graph.py rawptrs [file]     # Raw pointer class members
    python source_graph.py casts [file]       # C-style casts
    python source_graph.py looprisks [file]   # Infinite loop risk patterns
    python source_graph.py deadmethods        # Unreferenced methods
    python source_graph.py duplicates         # Duplicate code blocks
    python source_graph.py context <file>     # Unified context: summary + functions + deps
    python source_graph.py func <name>        # Find function with line ranges
    python source_graph.py api <name>         # CPU public API signatures (compact, AI-oriented)
    python source_graph.py body <name>        # Print function source from DB (no file read)
    python source_graph.py bodygrep <text>    # Search string literals inside function bodies (replaces grep)
    python source_graph.py gaps               # Find documentation/coverage gaps
    python source_graph.py summary            # Project overview with statistics
    python source_graph.py stats              # Database statistics
    python source_graph.py tags <term>        # Find semantic tags
    python source_graph.py hotspots [term]    # Semantic hotspots with risk/gain scores
    python source_graph.py coverage [term]    # Audit/test coverage signals
    python source_graph.py churn [term]       # Git history/churn signals
    python source_graph.py reviewqueue [term] # Prioritized review queue
    python source_graph.py ownership [term]   # Git blame ownership signals
    python source_graph.py testmap [term]     # Test-to-function mappings
    python source_graph.py calls <term>       # Function call graph edges
    python source_graph.py symbols [term]     # Symbol-level semantic metadata
    python source_graph.py bottlenecks [term] # Scored optimization queue
    python source_graph.py auditmap [term]    # Symbol audit coverage
    python source_graph.py aitasks [term]     # Proactive AI task queue
    python source_graph.py trace <term>       # Minimal code/config/test trace
    python source_graph.py impact <term>      # Change impact analysis
    python source_graph.py research [term]    # Search research assets
    python source_graph.py validateai [path]  # Validate Monster AI XML
    python source_graph.py sql "<query>"      # Run raw SQL query

  Token-saving commands:
        python source_graph.py focus <term> [budget] [--core]   # Compact ranked snapshot for agents
        python source_graph.py slice <term> [budget] [--core]   # Minimal dependency/call slice
    python source_graph.py bundle bugfix <term>    # Assemble bugfix context bundle
    python source_graph.py bundle feature <term>   # Assemble feature context bundle
    python source_graph.py bundle refactor <term>  # Assemble refactor context bundle
        python source_graph.py bundle audit <term>     # Assemble audit/security context bundle
        python source_graph.py bundle optimize <term>  # Assemble optimization context bundle
        python source_graph.py bundle explore <term>   # Assemble general exploration bundle
    python source_graph.py summarize [pattern]     # Show/generate function summaries
    python source_graph.py decide "<text>" --why Y # Log architecture decision
    python source_graph.py decisions [term]        # Query decision log
    python source_graph.py claudemd                # Generate CLAUDE.md documentation

  Project setup:
    python source_graph.py init                    # Generate starter source_graph.toml
    python source_graph.py export-config           # Export seed data to TOML config
"""
import json
import sqlite3
import sys
import os
import re
import glob
import hashlib
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from xml.etree import ElementTree as ET

try:
    import tomllib
except ImportError:
    tomllib = None

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DB_PATH = SCRIPT_DIR / "source_graph.db"
SCHEMA_VERSION = 5
EXTRACTOR_VERSION = "2026.03.18.3"


def _apply_project_dir(project_dir: str):
    """Override SCRIPT_DIR / REPO_ROOT / DB_PATH to point at another project.

    This lets a single canonical source_graph.py serve multiple projects,
    each with its own source_graph.toml and source_graph.db.
    The project_dir should be the directory containing source_graph.toml.
    """
    global SCRIPT_DIR, REPO_ROOT, DB_PATH
    p = Path(project_dir).resolve()
    if not p.is_dir():
        print(f"[!] --project directory does not exist: {p}")
        sys.exit(1)
    SCRIPT_DIR = p
    REPO_ROOT = p.parent
    DB_PATH = p / "source_graph.db"

# ---------- Config-driven project settings ----------
# These are set by load_config() at startup, with fallback defaults.
SOURCE_DIRS = []          # list of (label, Path, [extensions])
SOURCE_DIR_ADAPTERS = {}  # label -> LanguageAdapter override
SOURCE_DIR_RECURSIVE = {} # label -> bool
CATEGORY_RULES = []       # list of dicts: {match, pattern, category, source_dir?}
EXTERNAL_PATHS = {}       # e.g. {"runtime_data": Path(...), "research_dir": Path(...)}
SEED_DATA = {}            # e.g. {"singletons": [...], "events": [...], ...}
CONFIG_PATH = None        # path to loaded config file, or None

# Legacy aliases — set by load_config() for backward compat with existing scanners
GAME_DIR = REPO_ROOT / "Game"
COMMON_DIR = REPO_ROOT / "Common"
RUNTIME_DATA_ROOT = Path(r"E:\Dev\DevEmuMuServer\Data")
RESEARCH_DIR = REPO_ROOT / "ReversingResearch"

# Active language adapter — set by load_config()
LANG_ADAPTER = None  # type: LanguageAdapter
_NON_RECURSIVE_SOURCE_DIRS = set()
# Subdirectory name components to skip during rglob (e.g. node_modules, CMakeFiles)
EXCLUDE_SUBDIR_NAMES: set = set()
_ORIGINAL_PATH_RGLOB = Path.rglob


def _path_rglob_with_source_dir_policy(self, pattern):
    if str(self.resolve()) in _NON_RECURSIVE_SOURCE_DIRS:
        return self.glob(pattern)
    if EXCLUDE_SUBDIR_NAMES:
        # Yield only files where no parent component matches an excluded name
        def _filtered():
            for p in _ORIGINAL_PATH_RGLOB(self, pattern):
                try:
                    parts = p.relative_to(self).parts[:-1]  # parent dir components
                except ValueError:
                    yield p
                    continue
                skip = False
                for part in parts:
                    if part in EXCLUDE_SUBDIR_NAMES:
                        skip = True
                        break
                    for excl in EXCLUDE_SUBDIR_NAMES:
                        if excl.endswith('*') and part.startswith(excl[:-1]):
                            skip = True
                            break
                    if skip:
                        break
                if not skip:
                    yield p
        return _filtered()
    return _ORIGINAL_PATH_RGLOB(self, pattern)


Path.rglob = _path_rglob_with_source_dir_policy

# ---------- AI output control ----------
# Set to True by --quiet / -q global flag to suppress decorative headers.
# Pre-scanned from sys.argv in main() before load_config().
_QUIET = False


# ============================================================
# LANGUAGE ADAPTER SYSTEM
# ============================================================

class LanguageAdapter:
    """Base class for language-specific parsing logic.

    Each adapter provides regex patterns, comment-stripping, scope detection,
    and risk-analysis patterns for a single language family.
    """

    name = "base"
    extensions = []            # e.g. ["*.py"]
    header_extensions = []     # e.g. ["*.h"] for C++, [] for Python
    uses_braces = True         # False for Python (indent-based)

    # --- Import / Dependency ---
    def import_pattern(self):
        """Return compiled regex that captures the imported module/file name."""
        return None

    # --- Enum ---
    def enum_pattern(self):
        """Return compiled regex matching enum declarations. Group(1) = enum name."""
        return None

    # --- Struct / Data class ---
    def struct_pattern(self):
        """Return compiled regex matching struct/dataclass. Group(1) = name."""
        return None

    # --- Class ---
    def class_pattern(self):
        """Return compiled regex matching class declaration.
        Group(1) = class name, Group(2) = parent class (optional)."""
        return None

    # --- Method declaration (in headers/interfaces) ---
    def method_pattern(self):
        """Return compiled regex for method declarations.
        Group(1) = return type or decorator, Group(2) = method name."""
        return None

    def method_skip_names(self):
        """Return frozenset of keywords that look like methods but aren't."""
        return frozenset()

    # --- Function definition ---
    def function_sig_pattern(self):
        """Return compiled regex for function/method definition.
        Group(1) = class_name (optional), Group(2) = function name."""
        return None

    def constructor_pattern(self):
        """Return compiled regex for constructor definitions, or None."""
        return None

    def function_skip_names(self):
        """Return frozenset of names to skip when scanning function defs."""
        return frozenset()

    # --- Comment / string stripping ---
    def strip_comments_and_strings(self, text):
        """Remove comments and string literals, preserving newlines."""
        return text

    def strip_block_comments_from_lines(self, lines):
        """Return list of cleaned lines with block/line comments removed."""
        return lines

    # --- Scope detection ---
    def find_opening_scope(self, lines, start):
        """Find the line containing the opening scope marker ('{' or ':' for Python).
        Returns line index or None."""
        if self.uses_braces:
            return _find_opening_brace(lines, start)
        return self._find_indent_scope_start(lines, start)

    def find_scope_end(self, lines, scope_start):
        """Find the line where the scope ends.
        Returns line index or None."""
        if self.uses_braces:
            return _find_function_end(lines, scope_start)
        return self._find_indent_scope_end(lines, scope_start)

    def _find_indent_scope_start(self, lines, start):
        """For indent-based languages: find the colon that opens the block."""
        for i in range(start, min(start + 10, len(lines))):
            if ':' in lines[i] and not lines[i].strip().startswith('#'):
                return i
        return None

    def _find_indent_scope_end(self, lines, scope_start):
        """For indent-based languages: find where indentation returns to base level."""
        if scope_start >= len(lines):
            return None
        # Find the indentation of the first line inside the block
        base_indent = len(lines[scope_start]) - len(lines[scope_start].lstrip())
        body_indent = None
        for i in range(scope_start + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue
            indent = len(lines[i]) - len(lines[i].lstrip())
            if body_indent is None:
                if indent > base_indent:
                    body_indent = indent
                else:
                    return scope_start  # empty body
            else:
                if indent < body_indent and stripped and not stripped.startswith('#'):
                    return i - 1
        # Reached end of file
        return len(lines) - 1

    def is_declaration_not_definition(self, lines, sig_line, scope_line):
        """Check if a matched signature is a forward declaration, not a definition."""
        if self.uses_braces:
            for check_idx in range(sig_line, scope_line + 1):
                check_line = lines[check_idx]
                paren_pos = check_line.rfind(')')
                brace_pos = check_line.find('{')
                semi_pos = check_line.find(';')
                if semi_pos >= 0 and (brace_pos < 0 or semi_pos < brace_pos):
                    if paren_pos >= 0 and semi_pos > paren_pos:
                        return True
        return False

    # --- Define / Constant patterns ---
    def define_pattern(self):
        """Return compiled regex for constant/define declarations, or None.
        Group(1) = name, Group(2) = value."""
        return None

    # --- Call detection ---
    def call_patterns(self):
        """Return dict of compiled regex patterns for call detection.
        Keys: 'plain', 'scoped', 'member'. Each captures callee name."""
        return {
            'plain': re.compile(r'\b([A-Za-z_~]\w*)\s*\('),
            'scoped': re.compile(r'\b([A-Za-z_]\w*)(?:::|\.)\s*([A-Za-z_~]\w*)\s*\('),
            'member': re.compile(r'(?:->|\.)\s*([A-Za-z_~]\w*)\s*\('),
        }

    def call_skip_names(self):
        """Return frozenset of names to skip in call graph analysis."""
        return frozenset()

    # --- Null / None check risks ---
    def null_risk_calls(self):
        """Return list of function names that return nullable values."""
        return []

    def null_check_pattern(self, var_name):
        """Return True if the given window text contains a null check for var_name."""
        return False

    # --- Leak / Resource risks ---
    def leak_patterns(self):
        """Return dict with 'alloc' and 'dealloc' compiled regex patterns, or None."""
        return None

    # --- Unsafe cast patterns ---
    def unsafe_cast_pattern(self):
        """Return compiled regex for unsafe casts, or None."""
        return None

    # --- TODO comment pattern ---
    def todo_pattern(self):
        """Return compiled regex for TODO/FIXME/HACK comments."""
        return re.compile(r'(?://|/\*|#)\s*(TODO|FIXME|HACK|BUG|XXX|NOTE)\s*:?\s*(.*)', re.IGNORECASE)

    # --- Raw pointer / ownership risks ---
    def raw_pointer_pattern(self):
        """Return compiled regex for raw pointer members, or None."""
        return None

    # --- Crash risk patterns ---
    def crash_risk_patterns(self):
        """Return list of (pattern_re, risk_type, severity) tuples, or empty list."""
        return []


# ============================================================
# CONCRETE LANGUAGE ADAPTERS
# ============================================================

class CppAdapter(LanguageAdapter):
    """C/C++ language adapter.

    Covers all C-family source and shader formats used in this repo:
      - Standard C/C++: .cpp .cc .cxx .c .h .hpp .hxx
      - CUDA:           .cu .cuh           (NVIDIA GPU kernels)
      - OpenCL:         .cl                (cross-vendor GPU shaders)
      - Metal/ObjC++:   .mm .metal         (Apple GPU + Objective-C++)
    """
    name = "cpp"
    extensions = [
        "*.cpp", "*.h", "*.cc", "*.cxx", "*.hpp", "*.hxx", "*.c",
        # GPU / shader extensions
        "*.cu", "*.cuh",    # CUDA
        "*.cl",             # OpenCL
        "*.mm", "*.metal",  # Metal / Objective-C++
    ]
    header_extensions = ["*.h", "*.hpp", "*.hxx", "*.cuh"]
    uses_braces = True

    def import_pattern(self):
        return re.compile(r'#include\s+[<"]([^>"]+)[>"]')

    def enum_pattern(self):
        return re.compile(r'^\s*enum\s+(?:class\s+)?(\w+)')

    def struct_pattern(self):
        return re.compile(r'^\s*(?:typedef\s+)?struct\s+(\w+)')

    def class_pattern(self):
        return re.compile(r'^\s*class\s+(\w+)\s*(?::\s*(?:public|protected|private)\s+(\w+))?')

    def method_pattern(self):
        return re.compile(
            r'^\s*(?:virtual\s+|static\s+|inline\s+)*'
            r'(?:const\s+)?(\w[\w:*&<> ]*?)\s+'
            r'(\w+)\s*\([^)]*\)\s*'
            r'(?:const\s*)?(?:override\s*)?(?:=\s*0\s*)?;'
        )

    def method_skip_names(self):
        return frozenset({'if', 'for', 'while', 'switch', 'return', 'delete', 'new',
                          'class', 'struct', 'enum', 'typedef', 'using', 'namespace'})

    def function_sig_pattern(self):
        return re.compile(
            r'^[\s]*'
            r'(?:[\w:*&<>,\s]+?)\s+'
            r'(?:(\w+)\s*::\s*)?'
            r'(~?\w+)\s*\('
        )

    def constructor_pattern(self):
        return re.compile(r'^\s*(\w+)\s*::\s*(\1)\s*\(')

    def function_skip_names(self):
        return frozenset({
            'if', 'else', 'for', 'while', 'switch', 'catch', 'do', 'return',
            'case', 'throw', 'new', 'delete', 'sizeof', 'typeof', 'alignof',
            'noexcept', 'static_assert', 'ASSERT', 'CHECK', 'TEST_CASE',
            'LOG', 'define', 'ifdef', 'ifndef', 'elif', 'endif',
            'SingletonInstance', 'DECLARE', 'IMPLEMENT', 'macro', 'emit',
            # CUDA/OpenCL/Metal GPU qualifiers (must not be captured as func names)
            '__device__', '__global__', '__host__', '__forceinline__',
            '__shared__', '__constant__', '__restrict__', '__launch_bounds__',
            '__kernel', '__attribute__',
        })

    def strip_comments_and_strings(self, text):
        return _strip_cpp_comments_and_strings(text)

    def strip_block_comments_from_lines(self, lines):
        in_block_comment = False
        cleaned = []
        for line in lines:
            result = []
            i = 0
            while i < len(line):
                if in_block_comment:
                    end = line.find('*/', i)
                    if end >= 0:
                        in_block_comment = False
                        i = end + 2
                    else:
                        break
                else:
                    start = line.find('/*', i)
                    start_line_comment = line.find('//', i)
                    if start_line_comment >= 0 and (start < 0 or start_line_comment < start):
                        result.append(line[i:start_line_comment])
                        break
                    if start >= 0:
                        result.append(line[i:start])
                        end = line.find('*/', start + 2)
                        if end >= 0:
                            i = end + 2
                        else:
                            in_block_comment = True
                            break
                    else:
                        result.append(line[i:])
                        break
            cleaned.append(''.join(result))
        return cleaned

    def define_pattern(self):
        return re.compile(r'^\s*#define\s+(\w+)\s+(.+?)\s*(?://.*)?$')

    def call_patterns(self):
        return {
            'plain': re.compile(r'\b([A-Za-z_~]\w*)\s*\('),
            'scoped': re.compile(r'\b([A-Za-z_]\w*)\s*::\s*([A-Za-z_~]\w*)\s*\('),
            'member': re.compile(r'(?:->|\.)\s*([A-Za-z_~]\w*)\s*\('),
        }

    def call_skip_names(self):
        return frozenset({
            "if", "else", "for", "while", "switch", "catch", "return", "sizeof", "alignof", "decltype",
            "static_cast", "dynamic_cast", "reinterpret_cast", "const_cast", "ASSERT", "CHECK", "TEST_CASE",
            "min", "max", "defined", "new", "delete", "noexcept", "throw"
        })

    def null_risk_calls(self):
        return [
            'FindPlayer', 'GetPlayer', 'GetCharacter', 'GetWorld', 'GetParty',
            'GetGuild', 'GetMonster', 'GetUnit', 'GetItem', 'GetNpc',
            'GetDamageData', 'GetMember', 'GetLeader', 'GetBoss', 'GetFreeConnection',
        ]

    def null_check_pattern(self, var_name):
        """Build a check string for C++ null checks."""
        return re.compile(
            rf'if\s*\(?\s*!?\s*{re.escape(var_name)}\b|'
            rf'{re.escape(var_name)}\s*==\s*nullptr|'
            rf'{re.escape(var_name)}\s*==\s*NULL|'
            rf'!\s*{re.escape(var_name)}\b'
        )

    def leak_patterns(self):
        return {
            'alloc': re.compile(r'\bnew\s+\w'),
            'dealloc': re.compile(r'\bdelete\s'),
            'smart_unique': re.compile(r'\bmake_unique\b|\bunique_ptr\b'),
            'smart_shared': re.compile(r'\bmake_shared\b|\bshared_ptr\b'),
        }

    def unsafe_cast_pattern(self):
        return re.compile(r'\(\s*(\w[\w:]+\s*\*+)\s*\)\s*(\w+)')

    def raw_pointer_pattern(self):
        return re.compile(r'^\s+(\w[\w:*<> ]+\*)\s+(m_\w+|_\w+)\s*[;=]')

    def todo_pattern(self):
        return re.compile(r'(?://|/\*)\s*(TODO|FIXME|HACK|BUG|XXX|NOTE)\s*:?\s*(.*)', re.IGNORECASE)

    def crash_risk_patterns(self):
        return [
            (re.compile(r'(?:->|\.)(?:GetTarget|GetSummoner|GetWorld|GetOwner|GetParty|GetGuild|GetSummoned|GetInterfaceState)\s*\(\)\s*->\s*(\w+)'),
             'null_chain', 'critical'),
        ]


class PythonAdapter(LanguageAdapter):
    """Python language adapter."""
    name = "python"
    extensions = ["*.py"]
    header_extensions = []     # Python has no headers
    uses_braces = False

    def import_pattern(self):
        return re.compile(r'^\s*(?:import\s+([\w.]+)|from\s+([\w.]+)\s+import)')

    def enum_pattern(self):
        return re.compile(r'^\s*class\s+(\w+)\s*\(\s*(?:enum\.)?(?:Enum|IntEnum|StrEnum|Flag|IntFlag)\s*\)')

    def struct_pattern(self):
        return re.compile(r'^\s*@dataclass[^)]*\)?\s*\n\s*class\s+(\w+)', re.MULTILINE)

    def class_pattern(self):
        return re.compile(r'^\s*class\s+(\w+)\s*(?:\(\s*(\w+))?')

    def method_pattern(self):
        return re.compile(r'^\s+def\s+(\w+)\s*\(self')

    def method_skip_names(self):
        return frozenset({'__init__', '__del__', '__repr__', '__str__'})

    def function_sig_pattern(self):
        return re.compile(
            r'^\s*(?:async\s+)?def\s+(?:(\w+)\.)?(\w+)\s*\('
        )

    def constructor_pattern(self):
        return re.compile(r'^\s+def\s+(__init__)\s*\(self')

    def function_skip_names(self):
        return frozenset()

    def strip_comments_and_strings(self, text):
        result = []
        i = 0
        in_triple_dq = False
        in_triple_sq = False
        in_string_dq = False
        in_string_sq = False

        while i < len(text):
            ch = text[i]
            # Triple-quoted strings
            if not in_triple_dq and not in_triple_sq and not in_string_dq and not in_string_sq:
                if text[i:i+3] == '"""':
                    in_triple_dq = True
                    i += 3
                    continue
                if text[i:i+3] == "'''":
                    in_triple_sq = True
                    i += 3
                    continue
                if ch == '#':
                    # Line comment — skip to newline
                    while i < len(text) and text[i] != '\n':
                        i += 1
                    if i < len(text):
                        result.append('\n')
                        i += 1
                    continue
                if ch == '"':
                    in_string_dq = True
                    i += 1
                    continue
                if ch == "'":
                    in_string_sq = True
                    i += 1
                    continue
                result.append(ch)
                i += 1
            elif in_triple_dq:
                if text[i:i+3] == '"""':
                    in_triple_dq = False
                    i += 3
                else:
                    if ch == '\n':
                        result.append('\n')
                    i += 1
            elif in_triple_sq:
                if text[i:i+3] == "'''":
                    in_triple_sq = False
                    i += 3
                else:
                    if ch == '\n':
                        result.append('\n')
                    i += 1
            elif in_string_dq:
                if ch == '\\' and i + 1 < len(text):
                    i += 2
                elif ch == '"':
                    in_string_dq = False
                    i += 1
                else:
                    i += 1
            elif in_string_sq:
                if ch == '\\' and i + 1 < len(text):
                    i += 2
                elif ch == "'":
                    in_string_sq = False
                    i += 1
                else:
                    i += 1

        return "".join(result)

    def strip_block_comments_from_lines(self, lines):
        cleaned = []
        in_triple = False
        quote_char = None
        for line in lines:
            if in_triple:
                idx = line.find(quote_char * 3)
                if idx >= 0:
                    in_triple = False
                    cleaned.append(line[idx + 3:])
                else:
                    cleaned.append('')
            else:
                # Check for triple quotes
                for qc in ('"""', "'''"):
                    idx = line.find(qc)
                    if idx >= 0:
                        end_idx = line.find(qc, idx + 3)
                        if end_idx >= 0:
                            # Single-line docstring, remove it
                            cleaned.append(line[:idx] + line[end_idx + 3:])
                            break
                        else:
                            in_triple = True
                            quote_char = qc[0]
                            cleaned.append(line[:idx])
                            break
                else:
                    # Strip inline comments
                    comment_idx = line.find('#')
                    if comment_idx >= 0:
                        cleaned.append(line[:comment_idx])
                    else:
                        cleaned.append(line)
        return cleaned

    def define_pattern(self):
        # Python constants: UPPER_CASE = value
        return re.compile(r'^\s*([A-Z][A-Z0-9_]{2,})\s*(?::\s*\w+\s*)?=\s*(.+?)$')

    def call_patterns(self):
        return {
            'plain': re.compile(r'\b([A-Za-z_]\w*)\s*\('),
            'scoped': re.compile(r'\b([A-Za-z_]\w*)\.\s*([A-Za-z_]\w*)\s*\('),
            'member': re.compile(r'\.\s*([A-Za-z_]\w*)\s*\('),
        }

    def call_skip_names(self):
        return frozenset({
            "if", "elif", "else", "for", "while", "with", "return", "yield",
            "print", "len", "range", "enumerate", "zip", "map", "filter",
            "isinstance", "issubclass", "type", "super", "property",
            "staticmethod", "classmethod", "abstractmethod",
        })

    def null_risk_calls(self):
        return ['get', 'find', 'search', 'match', 'fetchone']

    def null_check_pattern(self, var_name):
        """Python null-safety patterns: if x, if x is not None, x or default, etc."""
        return re.compile(
            rf'if\s+{re.escape(var_name)}\b|'
            rf'if\s+{re.escape(var_name)}\s+is\s+not\s+None|'
            rf'{re.escape(var_name)}\s+is\s+not\s+None|'
            rf'{re.escape(var_name)}\s+is\s+None|'
            rf'{re.escape(var_name)}\s+or\s+|'
            rf'if\s+not\s+{re.escape(var_name)}\b'
        )

    def leak_patterns(self):
        return {
            'alloc': re.compile(r'\bopen\s*\('),
            'dealloc': re.compile(r'\.close\s*\('),
            'smart_unique': re.compile(r'\bwith\s+open\b'),
            'smart_shared': re.compile(r'contextmanager|__enter__|__exit__'),
        }

    def todo_pattern(self):
        return re.compile(r'#\s*(TODO|FIXME|HACK|BUG|XXX|NOTE)\s*:?\s*(.*)', re.IGNORECASE)


class TypeScriptAdapter(LanguageAdapter):
    """TypeScript/JavaScript language adapter."""
    name = "typescript"
    extensions = ["*.ts", "*.tsx", "*.js", "*.jsx", "*.mts", "*.mjs"]
    header_extensions = ["*.d.ts"]
    uses_braces = True

    def import_pattern(self):
        return re.compile(r'''(?:import\s+.*?from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\))''')

    def enum_pattern(self):
        return re.compile(r'^\s*(?:export\s+)?(?:const\s+)?enum\s+(\w+)')

    def struct_pattern(self):
        return re.compile(r'^\s*(?:export\s+)?(?:interface|type)\s+(\w+)')

    def class_pattern(self):
        return re.compile(r'^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)\s*(?:extends\s+(\w+))?')

    def method_pattern(self):
        return re.compile(
            r'^\s*(?:public|private|protected|static|async|readonly|abstract|override)*\s*'
            r'(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*\w[^{;]*)?[;]'
        )

    def method_skip_names(self):
        return frozenset({'if', 'for', 'while', 'switch', 'return', 'new', 'delete',
                          'class', 'interface', 'type', 'enum', 'import', 'export'})

    def function_sig_pattern(self):
        return re.compile(
            r'^\s*(?:export\s+)?(?:async\s+)?'
            r'(?:function\s+(\w+)|(?:(\w+)\s*(?:=|:)\s*(?:async\s+)?(?:function|\([^)]*\)\s*(?:=>|:)))|'
            r'(?:(?:public|private|protected|static|async|override)\s+)*(\w+)\s*\()'
        )

    def constructor_pattern(self):
        return re.compile(r'^\s*constructor\s*\(')

    def function_skip_names(self):
        return frozenset({
            'if', 'else', 'for', 'while', 'switch', 'catch', 'return',
            'case', 'throw', 'new', 'delete', 'typeof', 'instanceof',
            'import', 'export', 'from', 'require', 'const', 'let', 'var',
        })

    def strip_comments_and_strings(self, text):
        # JS/TS uses same comment style as C++ plus template literals
        result = []
        i = 0
        in_block = False
        in_line = False
        in_dq = False
        in_sq = False
        in_template = False

        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""

            if in_block:
                if ch == "*" and nxt == "/":
                    in_block = False
                    i += 2
                else:
                    if ch == '\n':
                        result.append('\n')
                    i += 1
                continue

            if in_line:
                if ch == "\n":
                    in_line = False
                    result.append("\n")
                i += 1
                continue

            if in_dq:
                if ch == "\\" and nxt:
                    i += 2
                elif ch == '"':
                    in_dq = False
                    i += 1
                else:
                    i += 1
                continue

            if in_sq:
                if ch == "\\" and nxt:
                    i += 2
                elif ch == "'":
                    in_sq = False
                    i += 1
                else:
                    i += 1
                continue

            if in_template:
                if ch == "\\" and nxt:
                    i += 2
                elif ch == '`':
                    in_template = False
                    i += 1
                else:
                    if ch == '\n':
                        result.append('\n')
                    i += 1
                continue

            if ch == "/" and nxt == "*":
                in_block = True
                i += 2
                continue
            if ch == "/" and nxt == "/":
                in_line = True
                i += 2
                continue
            if ch == '"':
                in_dq = True
                i += 1
                continue
            if ch == "'":
                in_sq = True
                i += 1
                continue
            if ch == '`':
                in_template = True
                i += 1
                continue

            result.append(ch)
            i += 1

        return "".join(result)

    def strip_block_comments_from_lines(self, lines):
        # Same as C++ (/* */ and //)
        return CppAdapter().strip_block_comments_from_lines(lines)

    def define_pattern(self):
        return re.compile(r'^\s*(?:export\s+)?const\s+([A-Z][A-Z0-9_]{2,})\s*(?::\s*\w+\s*)?=\s*(.+?)$')

    def call_patterns(self):
        return {
            'plain': re.compile(r'\b([A-Za-z_$]\w*)\s*\('),
            'scoped': re.compile(r'\b([A-Za-z_$]\w*)\.\s*([A-Za-z_$]\w*)\s*\('),
            'member': re.compile(r'\.\s*([A-Za-z_$]\w*)\s*\('),
        }

    def call_skip_names(self):
        return frozenset({
            "if", "else", "for", "while", "switch", "catch", "return", "typeof",
            "instanceof", "new", "delete", "throw", "require", "import", "from",
            "console", "setTimeout", "setInterval", "clearTimeout", "clearInterval",
        })

    def null_risk_calls(self):
        return ['getElementById', 'querySelector', 'find', 'get', 'match', 'exec']

    def leak_patterns(self):
        return {
            'alloc': re.compile(r'\baddEventListener\b|\bnew\s+\w'),
            'dealloc': re.compile(r'\bremoveEventListener\b'),
            'smart_unique': re.compile(r'\busing\b|\bfinally\b'),
            'smart_shared': re.compile(r'\bdispose\b|\bcleanup\b'),
        }

    def todo_pattern(self):
        return re.compile(r'(?://|/\*)\s*(TODO|FIXME|HACK|BUG|XXX|NOTE)\s*:?\s*(.*)', re.IGNORECASE)


class RustAdapter(LanguageAdapter):
    """Rust language adapter."""
    name = "rust"
    extensions = ["*.rs"]
    header_extensions = []
    uses_braces = True

    def import_pattern(self):
        return re.compile(r'^\s*use\s+([\w:]+)')

    def enum_pattern(self):
        return re.compile(r'^\s*(?:pub\s+)?enum\s+(\w+)')

    def struct_pattern(self):
        return re.compile(r'^\s*(?:pub\s+)?struct\s+(\w+)')

    def class_pattern(self):
        # Rust uses impl blocks instead of classes
        return re.compile(r'^\s*impl\s+(?:<[^>]*>\s+)?(\w+)\s*(?:for\s+(\w+))?')

    def method_pattern(self):
        return re.compile(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(')

    def method_skip_names(self):
        return frozenset()

    def function_sig_pattern(self):
        return re.compile(
            r'^\s*(?:pub(?:\(crate\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?'
            r'fn\s+(\w+)\s*(?:<[^>]*>)?\s*\('
        )

    def constructor_pattern(self):
        return re.compile(r'^\s*(?:pub\s+)?fn\s+(new)\s*\(')

    def function_skip_names(self):
        return frozenset({'macro_rules'})

    def strip_comments_and_strings(self, text):
        # Rust uses // and /* */ like C++, plus raw strings r#""#
        return _strip_cpp_comments_and_strings(text)

    def strip_block_comments_from_lines(self, lines):
        return CppAdapter().strip_block_comments_from_lines(lines)

    def define_pattern(self):
        return re.compile(r'^\s*(?:pub\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:\s*\w+\s*=\s*(.+?)\s*;')

    def call_patterns(self):
        return {
            'plain': re.compile(r'\b([A-Za-z_]\w*)\s*[!(]\s*'),
            'scoped': re.compile(r'\b([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)\s*\('),
            'member': re.compile(r'\.\s*([A-Za-z_]\w*)\s*\('),
        }

    def call_skip_names(self):
        return frozenset({
            "if", "else", "for", "while", "match", "loop", "return",
            "let", "mut", "ref", "Box", "Vec", "Some", "None", "Ok", "Err",
            "println", "eprintln", "format", "write", "writeln",
            "panic", "unreachable", "unimplemented", "todo", "assert",
        })

    def null_risk_calls(self):
        return ['unwrap', 'expect', 'unwrap_or']

    def leak_patterns(self):
        # Rust ownership prevents leaks; only explicit escape-hatch patterns
        # (raw pointer extraction, deliberate forget) are real memory-management risks.
        return {
            'alloc': re.compile(r'\bBox::into_raw\b|\bmem::forget\b|\bManuallyDrop::new\b'),
            'dealloc': re.compile(r'\bdrop\s*\(|\bfrom_raw\b'),
            'smart_unique': re.compile(r'\bBox<|Arc<|Rc<'),
            'smart_shared': re.compile(r'\bRc::new\b|\bArc::new\b'),
        }

    def todo_pattern(self):
        return re.compile(r'(?://|/\*)\s*(TODO|FIXME|HACK|BUG|XXX|NOTE)\s*:?\s*(.*)', re.IGNORECASE)


class GoAdapter(LanguageAdapter):
    """Go language adapter."""
    name = "go"
    extensions = ["*.go"]
    header_extensions = []
    uses_braces = True

    def import_pattern(self):
        return re.compile(r'^\s*(?:import\s+)?"([\w./\-]+)"')

    def enum_pattern(self):
        # Go uses const blocks with iota as enums
        return re.compile(r'^\s*type\s+(\w+)\s+(?:int|uint|string)\s*$')

    def struct_pattern(self):
        return re.compile(r'^\s*type\s+(\w+)\s+struct\s*\{?')

    def class_pattern(self):
        # Go uses interfaces instead of classes
        return re.compile(r'^\s*type\s+(\w+)\s+interface\s*\{?')

    def method_pattern(self):
        return re.compile(r'^\s*func\s+\(\s*\w+\s+\*?(\w+)\s*\)\s+(\w+)\s*\(')

    def method_skip_names(self):
        return frozenset()

    def function_sig_pattern(self):
        return re.compile(
            r'^\s*func\s+(?:\(\s*\w+\s+\*?(\w+)\s*\)\s+)?(\w+)\s*\('
        )

    def constructor_pattern(self):
        return re.compile(r'^\s*func\s+New(\w+)\s*\(')

    def function_skip_names(self):
        return frozenset({'init'})

    def strip_comments_and_strings(self, text):
        # Go uses // and /* */ like C++, plus backtick raw strings
        result = []
        i = 0
        in_block = False
        in_line_comment = False
        in_dq = False
        in_raw = False

        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""

            if in_block:
                if ch == '*' and nxt == '/':
                    in_block = False
                    i += 2
                else:
                    if ch == '\n':
                        result.append('\n')
                    i += 1
                continue

            if in_line_comment:
                if ch == '\n':
                    in_line_comment = False
                    result.append('\n')
                i += 1
                continue

            if in_dq:
                if ch == '\\' and nxt:
                    i += 2
                elif ch == '"':
                    in_dq = False
                    i += 1
                else:
                    i += 1
                continue

            if in_raw:
                if ch == '`':
                    in_raw = False
                    i += 1
                else:
                    if ch == '\n':
                        result.append('\n')
                    i += 1
                continue

            if ch == '/' and nxt == '*':
                in_block = True
                i += 2
                continue
            if ch == '/' and nxt == '/':
                in_line_comment = True
                i += 2
                continue
            if ch == '"':
                in_dq = True
                i += 1
                continue
            if ch == '`':
                in_raw = True
                i += 1
                continue

            result.append(ch)
            i += 1

        return "".join(result)

    def strip_block_comments_from_lines(self, lines):
        return CppAdapter().strip_block_comments_from_lines(lines)

    def define_pattern(self):
        return re.compile(r'^\s*(?:const\s+)?([A-Z][A-Za-z0-9_]*)\s*(?:\w+\s*)?=\s*(.+?)$')

    def call_patterns(self):
        return {
            'plain': re.compile(r'\b([A-Za-z_]\w*)\s*\('),
            'scoped': re.compile(r'\b([A-Za-z_]\w*)\.\s*([A-Za-z_]\w*)\s*\('),
            'member': re.compile(r'\.\s*([A-Za-z_]\w*)\s*\('),
        }

    def call_skip_names(self):
        return frozenset({
            "if", "else", "for", "switch", "select", "case", "return", "go",
            "defer", "range", "make", "append", "len", "cap", "new", "close",
            "delete", "copy", "panic", "recover", "print", "println",
            "fmt", "log", "errors",
        })

    def null_risk_calls(self):
        return []  # Go uses error returns, not null pointers

    def leak_patterns(self):
        return {
            'alloc': re.compile(r'\bos\.Open\b|\bnet\.Listen\b|\bhttp\.Get\b'),
            'dealloc': re.compile(r'\.Close\s*\('),
            'smart_unique': re.compile(r'\bdefer\s+\w+\.Close\b'),
            'smart_shared': re.compile(r'\bdefer\b'),
        }

    def todo_pattern(self):
        return re.compile(r'(?://|/\*)\s*(TODO|FIXME|HACK|BUG|XXX|NOTE)\s*:?\s*(.*)', re.IGNORECASE)


class JavaAdapter(LanguageAdapter):
    """Java language adapter."""
    name = "java"
    extensions = ["*.java"]
    header_extensions = []
    uses_braces = True

    def import_pattern(self):
        return re.compile(r'^\s*import\s+(?:static\s+)?([\w.]+)')

    def enum_pattern(self):
        return re.compile(r'^\s*(?:public\s+|private\s+|protected\s+)?enum\s+(\w+)')

    def struct_pattern(self):
        # Java records (Java 16+)
        return re.compile(r'^\s*(?:public\s+)?record\s+(\w+)')

    def class_pattern(self):
        return re.compile(
            r'^\s*(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?'
            r'(?:class|interface)\s+(\w+)\s*(?:extends\s+(\w+))?'
        )

    def method_pattern(self):
        return re.compile(
            r'^\s*(?:public|private|protected|static|final|abstract|synchronized|native|default)*\s*'
            r'(?:<[^>]*>\s+)?(\w[\w<>\[\], ]*?)\s+(\w+)\s*\([^)]*\)\s*'
            r'(?:throws\s+\w[\w, ]*)?\s*[;{]'
        )

    def method_skip_names(self):
        return frozenset({'if', 'for', 'while', 'switch', 'return', 'new',
                          'class', 'interface', 'enum', 'import', 'package'})

    def function_sig_pattern(self):
        return re.compile(
            r'^\s*(?:@\w+\s*(?:\([^)]*\)\s*)?)*'
            r'(?:public|private|protected|static|final|abstract|synchronized|native|default|\s)*'
            r'(?:<[^>]*>\s+)?(?:[\w<>\[\], ]+?)\s+'
            r'(\w+)\s*\('
        )

    def constructor_pattern(self):
        return re.compile(
            r'^\s*(?:public|private|protected)\s+(\w+)\s*\('
        )

    def function_skip_names(self):
        return frozenset({
            'if', 'else', 'for', 'while', 'switch', 'catch', 'return',
            'case', 'throw', 'new', 'class', 'interface', 'enum',
            'import', 'package', 'assert',
        })

    def strip_comments_and_strings(self, text):
        return _strip_cpp_comments_and_strings(text)

    def strip_block_comments_from_lines(self, lines):
        return CppAdapter().strip_block_comments_from_lines(lines)

    def define_pattern(self):
        return re.compile(
            r'^\s*(?:public|private|protected)?\s*static\s+final\s+\w+\s+([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*;'
        )

    def call_patterns(self):
        return {
            'plain': re.compile(r'\b([A-Za-z_]\w*)\s*\('),
            'scoped': re.compile(r'\b([A-Za-z_]\w*)\.\s*([A-Za-z_]\w*)\s*\('),
            'member': re.compile(r'\.\s*([A-Za-z_]\w*)\s*\('),
        }

    def call_skip_names(self):
        return frozenset({
            "if", "else", "for", "while", "switch", "catch", "return",
            "throw", "new", "instanceof", "assert",
            "System", "String", "Integer", "Long", "Double", "Boolean",
        })

    def null_risk_calls(self):
        return ['get', 'find', 'findById', 'getParameter', 'getAttribute',
                'getElementById', 'querySelector']

    def leak_patterns(self):
        return {
            'alloc': re.compile(r'\bnew\s+(?:FileInputStream|BufferedReader|Connection|Socket)\b'),
            'dealloc': re.compile(r'\.close\s*\('),
            'smart_unique': re.compile(r'\btry\s*\('),  # try-with-resources
            'smart_shared': re.compile(r'\bAutoCloseable\b|\bCloseable\b'),
        }

    def todo_pattern(self):
        return re.compile(r'(?://|/\*)\s*(TODO|FIXME|HACK|BUG|XXX|NOTE)\s*:?\s*(.*)', re.IGNORECASE)


class CSharpAdapter(LanguageAdapter):
    """C# language adapter."""
    name = "csharp"
    extensions = ["*.cs"]
    header_extensions = []
    uses_braces = True

    def import_pattern(self):
        return re.compile(r'^\s*using\s+(?:static\s+)?([\w.]+)\s*;')

    def enum_pattern(self):
        return re.compile(r'^\s*(?:public\s+|private\s+|protected\s+|internal\s+)?enum\s+(\w+)')

    def struct_pattern(self):
        return re.compile(
            r'^\s*(?:public\s+|private\s+|protected\s+|internal\s+)?'
            r'(?:readonly\s+)?(?:ref\s+)?struct\s+(\w+)'
        )

    def class_pattern(self):
        return re.compile(
            r'^\s*(?:public|private|protected|internal|abstract|sealed|static|partial|\s)*'
            r'class\s+(\w+)\s*(?::\s*(\w+))?'
        )

    def method_pattern(self):
        return re.compile(
            r'^\s*(?:public|private|protected|internal|static|virtual|override|abstract|sealed|async|partial|\s)*'
            r'(?:[\w<>\[\]?, ]+?)\s+(\w+)\s*\([^)]*\)\s*[;{]'
        )

    def method_skip_names(self):
        return frozenset({'if', 'for', 'foreach', 'while', 'switch', 'return', 'new',
                          'class', 'struct', 'enum', 'interface', 'using', 'namespace'})

    def function_sig_pattern(self):
        return re.compile(
            r'^\s*(?:\[[\w()]+\]\s*)*'
            r'(?:public|private|protected|internal|static|virtual|override|abstract|sealed|async|partial|\s)*'
            r'(?:[\w<>\[\]?, ]+?)\s+'
            r'(\w+)\s*\('
        )

    def constructor_pattern(self):
        return re.compile(
            r'^\s*(?:public|private|protected|internal|static)\s+(\w+)\s*\('
        )

    def function_skip_names(self):
        return frozenset({
            'if', 'else', 'for', 'foreach', 'while', 'switch', 'catch', 'return',
            'case', 'throw', 'new', 'typeof', 'sizeof', 'nameof',
            'class', 'struct', 'interface', 'enum', 'namespace', 'using',
        })

    def strip_comments_and_strings(self, text):
        # C# uses same comment syntax as C++, plus verbatim strings @""
        result = []
        i = 0
        in_block = False
        in_line = False
        in_dq = False
        in_verbatim = False
        in_char = False

        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""

            if in_block:
                if ch == '*' and nxt == '/':
                    in_block = False
                    i += 2
                else:
                    if ch == '\n':
                        result.append('\n')
                    i += 1
                continue

            if in_line:
                if ch == '\n':
                    in_line = False
                    result.append('\n')
                i += 1
                continue

            if in_verbatim:
                if ch == '"' and nxt == '"':
                    i += 2  # escaped quote inside verbatim
                elif ch == '"':
                    in_verbatim = False
                    i += 1
                else:
                    if ch == '\n':
                        result.append('\n')
                    i += 1
                continue

            if in_dq:
                if ch == '\\' and nxt:
                    i += 2
                elif ch == '"':
                    in_dq = False
                    i += 1
                else:
                    i += 1
                continue

            if in_char:
                if ch == '\\' and nxt:
                    i += 2
                elif ch == "'":
                    in_char = False
                    i += 1
                else:
                    i += 1
                continue

            if ch == '/' and nxt == '*':
                in_block = True
                i += 2
                continue
            if ch == '/' and nxt == '/':
                in_line = True
                i += 2
                continue
            if ch == '@' and nxt == '"':
                in_verbatim = True
                i += 2
                continue
            if ch == '"':
                in_dq = True
                i += 1
                continue
            if ch == "'":
                in_char = True
                i += 1
                continue

            result.append(ch)
            i += 1

        return "".join(result)

    def strip_block_comments_from_lines(self, lines):
        return CppAdapter().strip_block_comments_from_lines(lines)

    def define_pattern(self):
        return re.compile(
            r'^\s*(?:public|private|protected|internal)?\s*'
            r'(?:static\s+)?(?:readonly\s+)?const\s+\w+\s+([A-Z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*;'
        )

    def call_patterns(self):
        return {
            'plain': re.compile(r'\b([A-Za-z_]\w*)\s*\('),
            'scoped': re.compile(r'\b([A-Za-z_]\w*)\.\s*([A-Za-z_]\w*)\s*\('),
            'member': re.compile(r'\.\s*([A-Za-z_]\w*)\s*\('),
        }

    def call_skip_names(self):
        return frozenset({
            "if", "else", "for", "foreach", "while", "switch", "catch", "return",
            "throw", "new", "typeof", "sizeof", "nameof", "is", "as",
            "Console", "String", "Math", "Convert",
        })

    def null_risk_calls(self):
        return ['Find', 'FindObjectOfType', 'GetComponent', 'FirstOrDefault',
                'SingleOrDefault', 'Find', 'FindById']

    def leak_patterns(self):
        return {
            'alloc': re.compile(r'\bnew\s+(?:FileStream|StreamReader|SqlConnection|HttpClient)\b'),
            'dealloc': re.compile(r'\.Dispose\s*\(|\.Close\s*\('),
            'smart_unique': re.compile(r'\busing\s*\('),
            'smart_shared': re.compile(r'\bIDisposable\b|\bIAsyncDisposable\b'),
        }

    def todo_pattern(self):
        return re.compile(r'(?://|/\*)\s*(TODO|FIXME|HACK|BUG|XXX|NOTE)\s*:?\s*(.*)', re.IGNORECASE)


# --- Adapter registry ---
ADAPTER_REGISTRY = {
    "cpp": CppAdapter,
    "c++": CppAdapter,
    "c": CppAdapter,
    "python": PythonAdapter,
    "py": PythonAdapter,
    "typescript": TypeScriptAdapter,
    "ts": TypeScriptAdapter,
    "javascript": TypeScriptAdapter,
    "js": TypeScriptAdapter,
    "rust": RustAdapter,
    "rs": RustAdapter,
    "go": GoAdapter,
    "golang": GoAdapter,
    "java": JavaAdapter,
    "csharp": CSharpAdapter,
    "c#": CSharpAdapter,
    "cs": CSharpAdapter,
}


def get_adapter(language_name):
    """Get language adapter by name. Returns CppAdapter as default."""
    cls = ADAPTER_REGISTRY.get(language_name.lower(), CppAdapter)
    return cls()


# ============================================================
# SCHEMA
# ============================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_metadata (
    id INTEGER PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    extractor_version TEXT NOT NULL,
    graph_build_revision TEXT,
    built_at TEXT NOT NULL,
    repo_root TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    project TEXT NOT NULL,         -- 'Game', 'Common', 'Server_Link', etc.
    category TEXT,                 -- 'core', 'event', 'ai', 'script', 'player', 'monster', 'network', 'db', 'util'
    description TEXT
);

CREATE TABLE IF NOT EXISTS singletons (
    id INTEGER PRIMARY KEY,
    macro TEXT NOT NULL,           -- e.g. 'sGameServer'
    class_name TEXT NOT NULL,      -- e.g. 'GameServer'
    header TEXT NOT NULL,          -- e.g. 'GameServer.h'
    project TEXT DEFAULT 'Game',
    category TEXT,                 -- 'core', 'inter_server', 'player_social', 'item', 'skill', 'event', 'vip', 'quest', 'misc', 'common'
    description TEXT
);

CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    header TEXT,
    cpp_file TEXT,
    parent_class TEXT,
    project TEXT DEFAULT 'Game',
    category TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS methods (
    id INTEGER PRIMARY KEY,
    class_name TEXT NOT NULL,
    method_name TEXT NOT NULL,
    file TEXT NOT NULL,
    line_hint INTEGER,            -- approximate line number
    category TEXT,                -- 'combat', 'db', 'network', 'trade', 'party', etc.
    description TEXT
);

CREATE TABLE IF NOT EXISTS packet_handlers (
    id INTEGER PRIMARY KEY,
    headcode_name TEXT NOT NULL,
    headcode_value INTEGER,
    headcode_hex TEXT,
    handler_method TEXT,
    source_file TEXT NOT NULL,    -- 'Player.cpp', 'ServerLink.cpp', etc.
    handler_type TEXT NOT NULL,   -- 'client', 'inter_server', 'login', 'connect'
    category TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    singleton_macro TEXT,
    cpp_file TEXT,
    header_file TEXT,
    def_file TEXT,
    ai_file TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS ai_handlers (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    target TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS inventory_scripts (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS constants (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    value TEXT,
    header TEXT NOT NULL,
    category TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS config_files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS player_files (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    domain TEXT NOT NULL,
    key_methods TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS dependencies (
    id INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    target_file TEXT NOT NULL,
    dep_type TEXT               -- 'include', 'singleton_use', 'call'
);

CREATE TABLE IF NOT EXISTS enums (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER,
    value_count INTEGER,        -- number of enum values
    values_preview TEXT,        -- first few values as comma-separated string
    category TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS structs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER,
    size_bytes INTEGER,         -- sizeof if detectable
    field_count INTEGER,
    is_packed INTEGER DEFAULT 0,-- 1 if inside #pragma pack(1)
    fields_preview TEXT,        -- first few fields
    category TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    line INTEGER NOT NULL,
    todo_type TEXT NOT NULL,    -- 'TODO', 'FIXME', 'HACK', 'BUG', 'XXX', 'NOTE'
    text TEXT NOT NULL,
    project TEXT
);

CREATE TABLE IF NOT EXISTS db_tables (
    id INTEGER PRIMARY KEY,
    table_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    line INTEGER,
    query_type TEXT,            -- 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE'
    context TEXT                -- snippet of surrounding code
);

CREATE TABLE IF NOT EXISTS prepared_statements (
    id INTEGER PRIMARY KEY,
    query_name TEXT NOT NULL,   -- e.g. 'QUERY_PARTY_DELETE'
    sql_text TEXT NOT NULL,     -- the SQL string
    connection_type TEXT,       -- 'CONNECTION_ASYNC', 'CONNECTION_SYNCH'
    source_file TEXT NOT NULL,
    line INTEGER
);

CREATE TABLE IF NOT EXISTS config_keys (
    id INTEGER PRIMARY KEY,
    key_name TEXT NOT NULL,     -- the config key string
    getter_type TEXT,           -- 'GetInt', 'GetString', 'GetBool', 'GetFloat'
    source_file TEXT NOT NULL,
    line INTEGER,
    context TEXT                -- surrounding code snippet
);

CREATE TABLE IF NOT EXISTS defines (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    value TEXT,
    file TEXT NOT NULL,
    line INTEGER,
    category TEXT
);

CREATE TABLE IF NOT EXISTS file_lines (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    project TEXT,
    line_count INTEGER NOT NULL,
    category TEXT
);

CREATE TABLE IF NOT EXISTS leak_risks (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    project TEXT,
    new_count INTEGER DEFAULT 0,
    delete_count INTEGER DEFAULT 0,
    make_unique_count INTEGER DEFAULT 0,
    make_shared_count INTEGER DEFAULT 0,
    risk_score INTEGER DEFAULT 0,     -- new_count - delete_count - smart_ptr_count
    sample_lines TEXT                 -- first few 'new' lines for reference
);

CREATE TABLE IF NOT EXISTS null_risks (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    line INTEGER NOT NULL,
    function_call TEXT NOT NULL,      -- e.g. 'FindPlayer', 'GetWorld'
    pointer_var TEXT,                 -- the variable being assigned
    risk_type TEXT,                   -- 'unchecked_deref', 'no_null_check'
    context TEXT
);

CREATE TABLE IF NOT EXISTS raw_pointers (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    line INTEGER NOT NULL,
    class_name TEXT,
    member_type TEXT NOT NULL,        -- e.g. 'Player*'
    member_name TEXT NOT NULL,        -- e.g. 'm_pOwner'
    has_destructor_delete INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS unsafe_casts (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    line INTEGER NOT NULL,
    cast_expr TEXT NOT NULL,          -- the (Type*)ptr expression
    context TEXT
);

CREATE TABLE IF NOT EXISTS dead_methods (
    id INTEGER PRIMARY KEY,
    class_name TEXT NOT NULL,
    method_name TEXT NOT NULL,
    header_file TEXT NOT NULL,
    header_line INTEGER,
    ref_count INTEGER DEFAULT 0       -- 0 = truly dead
);

CREATE TABLE IF NOT EXISTS duplicate_blocks (
    id INTEGER PRIMARY KEY,
    block_hash TEXT NOT NULL,
    file_a TEXT NOT NULL,
    line_a INTEGER NOT NULL,
    file_b TEXT NOT NULL,
    line_b INTEGER NOT NULL,
    line_count INTEGER NOT NULL,      -- how many lines are duplicated
    preview TEXT                       -- first line of the block
);

CREATE TABLE IF NOT EXISTS research_assets (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    asset_type TEXT NOT NULL,          -- 'markdown', 'asm', 'text', 'binary', 'other'
    size_bytes INTEGER NOT NULL,
    title TEXT,
    summary TEXT,
    symbol_refs TEXT,
    protocol_refs TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS research_mentions (
    id INTEGER PRIMARY KEY,
    asset_path TEXT NOT NULL,
    symbol TEXT NOT NULL,
    mention_type TEXT NOT NULL,        -- 'symbol', 'protocol', 'asm_label'
    context TEXT
);

-- Full-text search virtual table
CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
    entity_type,    -- 'singleton', 'class', 'method', 'handler', 'event', 'ai', 'constant', 'config', 'file'
    name,
    file,
    category,
    description,
    tokenize='unicode61'
);

CREATE INDEX IF NOT EXISTS idx_singletons_macro ON singletons(macro);
CREATE INDEX IF NOT EXISTS idx_singletons_class ON singletons(class_name);
CREATE INDEX IF NOT EXISTS idx_methods_class ON methods(class_name);
CREATE INDEX IF NOT EXISTS idx_methods_name ON methods(method_name);
CREATE INDEX IF NOT EXISTS idx_handlers_name ON packet_handlers(headcode_name);
CREATE INDEX IF NOT EXISTS idx_handlers_type ON packet_handlers(handler_type);
CREATE INDEX IF NOT EXISTS idx_constants_name ON constants(name);
CREATE INDEX IF NOT EXISTS idx_deps_source ON dependencies(source_file);
CREATE INDEX IF NOT EXISTS idx_deps_target ON dependencies(target_file);
CREATE INDEX IF NOT EXISTS idx_enums_name ON enums(name);
CREATE INDEX IF NOT EXISTS idx_structs_name ON structs(name);
CREATE INDEX IF NOT EXISTS idx_todos_type ON todos(todo_type);
CREATE INDEX IF NOT EXISTS idx_todos_file ON todos(file);
CREATE INDEX IF NOT EXISTS idx_db_tables_name ON db_tables(table_name);
CREATE INDEX IF NOT EXISTS idx_classes_name ON classes(name);
CREATE INDEX IF NOT EXISTS idx_classes_parent ON classes(parent_class);
CREATE INDEX IF NOT EXISTS idx_prepared_name ON prepared_statements(query_name);
CREATE INDEX IF NOT EXISTS idx_config_keys_name ON config_keys(key_name);
CREATE INDEX IF NOT EXISTS idx_defines_name ON defines(name);
CREATE INDEX IF NOT EXISTS idx_file_lines_count ON file_lines(line_count DESC);
CREATE INDEX IF NOT EXISTS idx_leak_risks_score ON leak_risks(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_null_risks_file ON null_risks(file);
CREATE INDEX IF NOT EXISTS idx_raw_pointers_file ON raw_pointers(file);
CREATE INDEX IF NOT EXISTS idx_unsafe_casts_file ON unsafe_casts(file);
CREATE INDEX IF NOT EXISTS idx_dead_methods_class ON dead_methods(class_name);
CREATE INDEX IF NOT EXISTS idx_research_assets_name ON research_assets(file_name);
CREATE INDEX IF NOT EXISTS idx_research_assets_type ON research_assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_research_mentions_symbol ON research_mentions(symbol);
CREATE INDEX IF NOT EXISTS idx_research_mentions_asset ON research_mentions(asset_path);
CREATE TABLE IF NOT EXISTS crash_risks (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    line INTEGER NOT NULL,
    risk_type TEXT NOT NULL,      -- 'null_chain', 'div_zero', 'use_after_free', 'unchecked_target', 'unchecked_world', 'unchecked_summoner'
    severity TEXT NOT NULL,       -- 'critical', 'high', 'medium'
    expression TEXT NOT NULL,     -- the risky code expression
    context TEXT                  -- full line of code
);

CREATE INDEX IF NOT EXISTS idx_crash_risks_file ON crash_risks(file);
CREATE INDEX IF NOT EXISTS idx_crash_risks_type ON crash_risks(risk_type);
CREATE INDEX IF NOT EXISTS idx_crash_risks_severity ON crash_risks(severity);

CREATE INDEX IF NOT EXISTS idx_duplicate_blocks_hash ON duplicate_blocks(block_hash);

CREATE TABLE IF NOT EXISTS infinite_loop_risks (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    line INTEGER NOT NULL,
    risk_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    expression TEXT NOT NULL,
    context TEXT
);

CREATE INDEX IF NOT EXISTS idx_infinite_loop_risks_file ON infinite_loop_risks(file);
CREATE INDEX IF NOT EXISTS idx_infinite_loop_risks_type ON infinite_loop_risks(risk_type);

CREATE TABLE IF NOT EXISTS file_summaries (
    id INTEGER PRIMARY KEY,
    file TEXT UNIQUE NOT NULL,
    project TEXT,
    summary TEXT NOT NULL,
    category TEXT
);

CREATE TABLE IF NOT EXISTS function_index (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    function_name TEXT NOT NULL,
    class_name TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT,
    project TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS semantic_tags (
    id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,   -- 'file', 'function', 'class', 'handler', 'event', 'config'
    entity_name TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER,
    tag TEXT NOT NULL,
    confidence INTEGER DEFAULT 100,
    evidence TEXT
);

CREATE TABLE IF NOT EXISTS semantic_profiles (
    id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER,
    security_sensitive INTEGER DEFAULT 0,
    hot_path_candidate INTEGER DEFAULT 0,
    optimization_candidate INTEGER DEFAULT 0,
    network_surface INTEGER DEFAULT 0,
    fragile_surface INTEGER DEFAULT 0,
    test_surface INTEGER DEFAULT 0,
    risk_score INTEGER DEFAULT 0,
    gain_score INTEGER DEFAULT 0,
    evidence TEXT
);

CREATE TABLE IF NOT EXISTS audit_coverage (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    project TEXT,
    unit_test_refs INTEGER DEFAULT 0,
    has_summary INTEGER DEFAULT 0,
    has_function_index INTEGER DEFAULT 0,
    todo_count INTEGER DEFAULT 0,
    crash_risk_count INTEGER DEFAULT 0,
    null_risk_count INTEGER DEFAULT 0,
    duplicate_pair_count INTEGER DEFAULT 0,
    dead_method_count INTEGER DEFAULT 0,
    leak_risk_score INTEGER DEFAULT 0,
    semantic_risk_max INTEGER DEFAULT 0,
    semantic_gain_max INTEGER DEFAULT 0,
    coverage_score INTEGER DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS history_metrics (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    commit_count INTEGER DEFAULT 0,
    recent_commit_count INTEGER DEFAULT 0,
    unique_authors INTEGER DEFAULT 0,
    bugfix_commits INTEGER DEFAULT 0,
    perf_commits INTEGER DEFAULT 0,
    audit_commits INTEGER DEFAULT 0,
    last_commit_date TEXT,
    churn_score INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    queue_type TEXT NOT NULL,
    priority_score INTEGER DEFAULT 0,
    rationale TEXT
);

CREATE TABLE IF NOT EXISTS ownership_metrics (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    primary_author TEXT,
    primary_author_share INTEGER DEFAULT 0,
    author_count INTEGER DEFAULT 0,
    blamed_lines INTEGER DEFAULT 0,
    most_recent_line_date TEXT,
    bus_factor_risk INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS test_function_map (
    id INTEGER PRIMARY KEY,
    test_file TEXT NOT NULL,
    target_file TEXT NOT NULL,
    function_name TEXT NOT NULL,
    class_name TEXT,
    mapping_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS call_edges (
    id INTEGER PRIMARY KEY,
    caller_symbol TEXT NOT NULL,
    caller_file TEXT NOT NULL,
    caller_project TEXT,
    callee_symbol TEXT NOT NULL,
    callee_file TEXT NOT NULL,
    callee_project TEXT,
    confidence INTEGER DEFAULT 0,
    call_count INTEGER DEFAULT 1,
    evidence TEXT,
    UNIQUE(caller_symbol, caller_file, callee_symbol, callee_file)
);

CREATE TABLE IF NOT EXISTS symbol_metadata (
    id INTEGER PRIMARY KEY,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    project TEXT,
    class_name TEXT,
    summary TEXT,
    semantic_tags TEXT,
    hot_path INTEGER DEFAULT 0,
    ct_sensitive INTEGER DEFAULT 0,
    batchable INTEGER DEFAULT 0,
    gpu_candidate INTEGER DEFAULT 0,
    risk_level TEXT DEFAULT 'low',
    review_priority INTEGER DEFAULT 0,
    risk_score INTEGER DEFAULT 0,
    gain_score INTEGER DEFAULT 0,
    audit_coverage_score INTEGER DEFAULT 0,
    change_frequency INTEGER DEFAULT 0,
    line_span INTEGER DEFAULT 0,
    loop_count INTEGER DEFAULT 0,
    branch_count INTEGER DEFAULT 0,
    caller_count INTEGER DEFAULT 0,
    callee_count INTEGER DEFAULT 0,
    reasons TEXT,
    UNIQUE(symbol_name, file_path)
);

CREATE TABLE IF NOT EXISTS analysis_scores (
    id INTEGER PRIMARY KEY,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    hotness_score INTEGER DEFAULT 0,
    complexity_score INTEGER DEFAULT 0,
    fanin_score INTEGER DEFAULT 0,
    fanout_score INTEGER DEFAULT 0,
    optimization_score INTEGER DEFAULT 0,
    gpu_score INTEGER DEFAULT 0,
    ct_risk_score INTEGER DEFAULT 0,
    audit_gap_score INTEGER DEFAULT 0,
    perf_priority INTEGER DEFAULT 0,
    safe_priority INTEGER DEFAULT 0,
    overall_priority INTEGER DEFAULT 0,
    reasons TEXT NOT NULL,
    UNIQUE(symbol_name, file_path)
);

CREATE TABLE IF NOT EXISTS symbol_audit_coverage (
    id INTEGER PRIMARY KEY,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    covered_by_tests INTEGER DEFAULT 0,
    test_count INTEGER DEFAULT 0,
    mapping_types TEXT,
    review_queue_types TEXT,
    audit_modules TEXT,
    coverage_score INTEGER DEFAULT 0,
    last_status TEXT DEFAULT 'unknown',
    historical_failures INTEGER DEFAULT 0,
    evidence TEXT,
    UNIQUE(symbol_name, file_path)
);

CREATE TABLE IF NOT EXISTS ai_tasks (
    id INTEGER PRIMARY KEY,
    task_type TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    rationale TEXT,
    UNIQUE(task_type, symbol_name, file_path)
);

CREATE VIEW IF NOT EXISTS v_bottleneck_queue AS
SELECT
    a.symbol_name,
    a.file_path,
    a.hotness_score,
    a.complexity_score,
    a.fanin_score,
    a.fanout_score,
    a.optimization_score,
    a.gpu_score,
    a.ct_risk_score,
    a.audit_gap_score,
    a.perf_priority,
    a.safe_priority,
    a.overall_priority,
    a.reasons,
    sm.summary,
    sm.semantic_tags,
    sm.hot_path,
    sm.ct_sensitive,
    sm.batchable,
    sm.gpu_candidate,
    sm.risk_level,
    sm.review_priority,
    sm.audit_coverage_score,
    sm.change_frequency
FROM analysis_scores a
LEFT JOIN symbol_metadata sm
    ON sm.symbol_name = a.symbol_name AND sm.file_path = a.file_path;

CREATE INDEX IF NOT EXISTS idx_file_summaries_file ON file_summaries(file);
CREATE INDEX IF NOT EXISTS idx_function_index_file ON function_index(file);
CREATE INDEX IF NOT EXISTS idx_function_index_name ON function_index(function_name);
CREATE INDEX IF NOT EXISTS idx_function_index_class ON function_index(class_name);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_semantic_tags_tag ON semantic_tags(tag);
CREATE INDEX IF NOT EXISTS idx_semantic_tags_file ON semantic_tags(file);
CREATE INDEX IF NOT EXISTS idx_semantic_tags_entity ON semantic_tags(entity_type, entity_name);
CREATE INDEX IF NOT EXISTS idx_semantic_profiles_risk ON semantic_profiles(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_profiles_gain ON semantic_profiles(gain_score DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_profiles_file ON semantic_profiles(file);
CREATE INDEX IF NOT EXISTS idx_audit_coverage_file ON audit_coverage(file);
CREATE INDEX IF NOT EXISTS idx_audit_coverage_score ON audit_coverage(coverage_score);
CREATE INDEX IF NOT EXISTS idx_history_metrics_file ON history_metrics(file);
CREATE INDEX IF NOT EXISTS idx_history_metrics_churn ON history_metrics(churn_score DESC);
CREATE INDEX IF NOT EXISTS idx_review_queue_type ON review_queue(queue_type);
CREATE INDEX IF NOT EXISTS idx_review_queue_priority ON review_queue(priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_ownership_metrics_file ON ownership_metrics(file);
CREATE INDEX IF NOT EXISTS idx_test_function_map_target ON test_function_map(target_file);
CREATE INDEX IF NOT EXISTS idx_test_function_map_test ON test_function_map(test_file);
CREATE INDEX IF NOT EXISTS idx_call_edges_caller ON call_edges(caller_symbol, caller_file);
CREATE INDEX IF NOT EXISTS idx_call_edges_callee ON call_edges(callee_symbol, callee_file);
CREATE INDEX IF NOT EXISTS idx_call_edges_confidence ON call_edges(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_symbol_metadata_priority ON symbol_metadata(review_priority DESC);
CREATE INDEX IF NOT EXISTS idx_symbol_metadata_hot ON symbol_metadata(hot_path, gpu_candidate, ct_sensitive);
CREATE INDEX IF NOT EXISTS idx_symbol_metadata_risk ON symbol_metadata(risk_score DESC, gain_score DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_scores_priority ON analysis_scores(overall_priority DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_scores_perf ON analysis_scores(perf_priority DESC);
CREATE INDEX IF NOT EXISTS idx_symbol_audit_coverage_symbol ON symbol_audit_coverage(symbol_name, file_path);
CREATE INDEX IF NOT EXISTS idx_symbol_audit_coverage_score ON symbol_audit_coverage(coverage_score);
CREATE INDEX IF NOT EXISTS idx_ai_tasks_status_priority ON ai_tasks(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_ai_tasks_type ON ai_tasks(task_type);

CREATE TABLE IF NOT EXISTS file_hashes (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    last_scan_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS function_bodies (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    function_name TEXT NOT NULL,
    class_name TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    line_count INTEGER NOT NULL,
    project TEXT
);

CREATE INDEX IF NOT EXISTS idx_function_bodies_name ON function_bodies(function_name);
CREATE INDEX IF NOT EXISTS idx_function_bodies_file ON function_bodies(file);

CREATE TABLE IF NOT EXISTS function_summaries (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL,
    function_name TEXT NOT NULL,
    class_name TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    summary TEXT,
    params TEXT,
    return_type TEXT,
    side_effects TEXT,
    body_hash TEXT NOT NULL,
    stale INTEGER DEFAULT 0,
    generated_at TEXT,
    generator TEXT,
    UNIQUE(file, function_name, start_line)
);

CREATE INDEX IF NOT EXISTS idx_function_summaries_file ON function_summaries(file);
CREATE INDEX IF NOT EXISTS idx_function_summaries_name ON function_summaries(function_name);
CREATE INDEX IF NOT EXISTS idx_function_summaries_stale ON function_summaries(stale);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    file TEXT,
    function_name TEXT,
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    alternatives TEXT,
    author TEXT,
    tags TEXT,
    status TEXT DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_decisions_file ON decisions(file);
CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);
CREATE INDEX IF NOT EXISTS idx_decisions_tags ON decisions(tags);

-- ============================================================
-- DATAFLOW GRAPH (type-level data flow: Scalar→Point→Sig etc.)
-- ============================================================
CREATE TABLE IF NOT EXISTS dataflow_edges (
    id INTEGER PRIMARY KEY,
    from_type TEXT NOT NULL,
    to_type TEXT NOT NULL,
    via_function TEXT NOT NULL,
    via_file TEXT NOT NULL,
    project TEXT,
    flow_kind TEXT DEFAULT 'transform',  -- transform | consume | produce | alias
    confidence INTEGER DEFAULT 50,
    UNIQUE(from_type, to_type, via_function, via_file)
);
CREATE INDEX IF NOT EXISTS idx_dataflow_from  ON dataflow_edges(from_type);
CREATE INDEX IF NOT EXISTS idx_dataflow_to    ON dataflow_edges(to_type);
CREATE INDEX IF NOT EXISTS idx_dataflow_func  ON dataflow_edges(via_function);

-- ============================================================
-- INVARIANTS (assert / precondition / postcondition / comment constraints)
-- ============================================================
CREATE TABLE IF NOT EXISTS invariants (
    id INTEGER PRIMARY KEY,
    symbol_name TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER NOT NULL,
    invariant_text TEXT NOT NULL,
    invariant_type TEXT NOT NULL,  -- assert|static_assert|precondition|comment_constraint|postcondition
    severity TEXT DEFAULT 'info',  -- info|warning|critical
    confidence INTEGER DEFAULT 60,
    related_type TEXT              -- which data type this constrains, e.g. "Scalar"
);
CREATE INDEX IF NOT EXISTS idx_invariants_symbol ON invariants(symbol_name);
CREATE INDEX IF NOT EXISTS idx_invariants_file   ON invariants(file);
CREATE INDEX IF NOT EXISTS idx_invariants_type   ON invariants(invariant_type);
CREATE INDEX IF NOT EXISTS idx_invariants_rtype  ON invariants(related_type);

-- ============================================================
-- BACKEND MAP (CPU ↔ CUDA ↔ OpenCL ↔ Metal parity tracking)
-- ============================================================
CREATE TABLE IF NOT EXISTS backend_map (
    id INTEGER PRIMARY KEY,
    operation_name TEXT NOT NULL UNIQUE,
    cpu_symbol TEXT, cpu_file TEXT,
    cuda_symbol TEXT, cuda_file TEXT,
    opencl_symbol TEXT, opencl_file TEXT,
    metal_symbol TEXT, metal_file TEXT,
    parity_status TEXT DEFAULT 'unknown',  -- full|partial|cpu-only|missing-metal|missing-opencl|missing-cuda
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_backend_map_op     ON backend_map(operation_name);
CREATE INDEX IF NOT EXISTS idx_backend_map_parity ON backend_map(parity_status);
"""


# ============================================================
# CONFIG SYSTEM
# ============================================================

_DEFAULT_CATEGORY_RULES = [
    # Common project rules
    {"match": "contains", "pattern": "Database", "category": "db", "source_dir": "Common"},
    {"match": "contains", "pattern": "MySQL", "category": "db", "source_dir": "Common"},
    {"match": "contains", "pattern": "Query", "category": "db", "source_dir": "Common"},
    {"match": "contains", "pattern": "Transaction", "category": "db", "source_dir": "Common"},
    {"match": "contains", "pattern": "Statement", "category": "db", "source_dir": "Common"},
    {"match": "contains", "pattern": "Field", "category": "db", "source_dir": "Common"},
    {"match": "contains", "pattern": "Socket", "category": "network", "source_dir": "Common"},
    {"match": "contains", "pattern": "TCP", "category": "network", "source_dir": "Common"},
    {"match": "contains", "pattern": "Acceptor", "category": "network", "source_dir": "Common"},
    {"match": "contains", "pattern": "Buffer", "category": "network", "source_dir": "Common"},
    {"match": "contains", "pattern": "Packet", "category": "network", "source_dir": "Common"},
    {"match": "contains", "pattern": "Log", "category": "logging", "source_dir": "Common"},
    {"match": "contains", "pattern": "Appender", "category": "logging", "source_dir": "Common"},
    {"match": "contains", "pattern": "Timer", "category": "util", "source_dir": "Common"},
    {"match": "contains", "pattern": "Util", "category": "util", "source_dir": "Common"},
    {"match": "contains", "pattern": "Random", "category": "util", "source_dir": "Common"},
    {"match": "contains", "pattern": "Byte", "category": "util", "source_dir": "Common"},
    {"match": "contains", "pattern": "Enc", "category": "security", "source_dir": "Common"},
    {"match": "contains", "pattern": "MD5", "category": "security", "source_dir": "Common"},
    {"match": "contains", "pattern": "Sha", "category": "security", "source_dir": "Common"},
    {"match": "contains", "pattern": "base64", "category": "security", "source_dir": "Common"},
    {"match": "contains", "pattern": "Security", "category": "security", "source_dir": "Common"},
    # Game project rules
    {"match": "prefix", "pattern": "ai_", "category": "ai"},
    {"match": "prefix", "pattern": "Player", "category": "player"},
    {"match": "prefix", "pattern": "Monster", "category": "monster"},
    {"match": "contains", "pattern": "Script", "category": "script"},
    {"match": "prefix", "pattern": "World", "category": "world"},
    {"match": "suffix", "pattern": "Def.h", "category": "definition"},
    {"match": "suffix", "pattern": "Packet.h", "category": "packet"},
]


def _find_config_file():
    """Search for source_graph.toml in standard locations."""
    candidates = [
        SCRIPT_DIR / "source_graph.toml",
        REPO_ROOT / "source_graph.toml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _parse_toml(path):
    """Parse a TOML config file. Requires Python 3.11+ tomllib or fallback."""
    if tomllib is not None:
        with open(path, "rb") as f:
            return tomllib.load(f)
    # Minimal fallback for older Python: try json with toml-like keys
    # (users on <3.11 can use JSON config instead)
    json_path = path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[!] Python 3.11+ required for TOML config, or provide {json_path}")
    sys.exit(1)


def _detect_language(directory):
    """Detect the primary language in a directory by counting files."""
    lang_exts = {
        "cpp":        ["*.cpp", "*.h", "*.cc", "*.cxx", "*.hpp", "*.c"],
        "python":     ["*.py"],
        "typescript":  ["*.ts", "*.tsx", "*.js", "*.jsx"],
        "rust":       ["*.rs"],
        "go":         ["*.go"],
        "java":       ["*.java"],
        "csharp":     ["*.cs"],
    }
    counts = {}
    for lang, exts in lang_exts.items():
        total = 0
        for ext in exts:
            total += len(list(directory.glob(ext)))
            # Also check one level of subdirectories
            total += len(list(directory.glob(f"*/{ext}")))
        if total > 0:
            counts[lang] = total
    if not counts:
        return "cpp", ["*.cpp", "*.h"]  # default fallback
    best = max(counts, key=counts.get)
    adapter = get_adapter(best)
    return best, adapter.extensions


def _auto_detect_config():
    """Auto-detect project structure and language when no config file exists."""
    skip_dirs = {
        "tools", "build", "bin", "obj", "Debug", "Release", "x64", ".vs",
        "packages", "UnitTests", "node_modules", "__pycache__", "target",
        "dist", "out", ".git", ".idea", ".vscode", "vendor", "venv",
        ".env", "env", ".tox", ".mypy_cache", ".pytest_cache",
    }

    # First detect project-level language
    lang, exts = _detect_language(REPO_ROOT)

    cfg = {
        "project": {"name": REPO_ROOT.name, "language": lang},
        "source_dirs": [],
        "external_paths": {},
        "category_rules": _DEFAULT_CATEGORY_RULES if lang == "cpp" else [],
    }

    # Detect source directories containing files of the detected language
    for subdir in sorted(REPO_ROOT.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith(".") or subdir.name in skip_dirs:
            continue
        file_count = 0
        for ext in exts:
            file_count += len(list(subdir.glob(ext)))
        if file_count > 0:
            cfg["source_dirs"].append({
                "label": subdir.name,
                "path": subdir.name,
                "extensions": exts,
            })

    if not cfg["source_dirs"]:
        # Check root directory itself for source files
        file_count = sum(len(list(REPO_ROOT.glob(ext))) for ext in exts)
        if file_count > 0:
            cfg["source_dirs"].append({
                "label": REPO_ROOT.name,
                "path": ".",
                "extensions": exts,
            })
        # Also check common project structures: src/, lib/, app/
        for common_dir in ("src", "lib", "app", "pkg", "cmd", "internal"):
            d = REPO_ROOT / common_dir
            if d.exists() and d.is_dir():
                file_count = sum(len(list(d.rglob(ext))) for ext in exts)
                if file_count > 0:
                    cfg["source_dirs"].append({
                        "label": common_dir,
                        "path": common_dir,
                        "extensions": exts,
                    })

    if not cfg["source_dirs"]:
        cfg["source_dirs"].append({
            "label": REPO_ROOT.name,
            "path": ".",
            "extensions": exts,
        })
    return cfg


def load_config():
    """Load project config and set global variables."""
    global SOURCE_DIRS, SOURCE_DIR_ADAPTERS, SOURCE_DIR_RECURSIVE
    global CATEGORY_RULES, EXTERNAL_PATHS, SEED_DATA, CONFIG_PATH
    global GAME_DIR, COMMON_DIR, RUNTIME_DATA_ROOT, RESEARCH_DIR
    global LANG_ADAPTER
    global _NON_RECURSIVE_SOURCE_DIRS, EXCLUDE_SUBDIR_NAMES

    config_path = _find_config_file()
    CONFIG_PATH = config_path

    if config_path:
        cfg = _parse_toml(config_path)
        config_base = config_path.parent
    else:
        cfg = _auto_detect_config()
        config_base = REPO_ROOT

    # Language adapter
    project_cfg = cfg.get("project", {})
    language = project_cfg.get("language", "cpp")
    LANG_ADAPTER = get_adapter(language)
    if not _QUIET:
        print(f"[*] Language: {LANG_ADAPTER.name} ({language})")

    # Excluded subdirectory name components
    EXCLUDE_SUBDIR_NAMES = set(cfg.get("project", {}).get("exclude_subdir_names", []))

    # Source directories
    SOURCE_DIRS = []
    SOURCE_DIR_ADAPTERS = {}
    SOURCE_DIR_RECURSIVE = {}
    _NON_RECURSIVE_SOURCE_DIRS = set()
    default_exts = LANG_ADAPTER.extensions
    for sd in cfg.get("source_dirs", []):
        label = sd["label"]
        path = Path(sd["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        exts = sd.get("extensions", default_exts)
        optional = sd.get("optional", False)
        adapter = get_adapter(sd["language"]) if sd.get("language") else LANG_ADAPTER
        recursive = sd.get("recursive", True)
        if path.exists() or not optional:
            SOURCE_DIRS.append((label, path, exts))
            SOURCE_DIR_ADAPTERS[label] = adapter
            SOURCE_DIR_RECURSIVE[label] = recursive
            if not recursive and path.exists():
                _NON_RECURSIVE_SOURCE_DIRS.add(str(path.resolve()))

    # Category rules
    if "category_rules" in cfg:
        CATEGORY_RULES = cfg["category_rules"]
    else:
        CATEGORY_RULES = _DEFAULT_CATEGORY_RULES

    # External paths
    ext = cfg.get("external_paths", {})
    EXTERNAL_PATHS = {}
    for key, val in ext.items():
        p = Path(val)
        if not p.is_absolute():
            p = REPO_ROOT / p
        EXTERNAL_PATHS[key] = p

    # Seed data
    SEED_DATA = {}
    for table in ("singletons", "player_files", "events", "ai_handlers",
                   "inventory_scripts", "constants", "config_files", "packet_handlers"):
        if table in cfg:
            SEED_DATA[table] = cfg[table]

    # Legacy aliases for backward compat
    for label, path, _exts in SOURCE_DIRS:
        if label == "Game":
            GAME_DIR = path
        elif label == "Common":
            COMMON_DIR = path
    if "runtime_data" in EXTERNAL_PATHS:
        RUNTIME_DATA_ROOT = EXTERNAL_PATHS["runtime_data"]
    if "research_dir" in EXTERNAL_PATHS:
        RESEARCH_DIR = EXTERNAL_PATHS["research_dir"]


def _project_adapter(project):
    return SOURCE_DIR_ADAPTERS.get(project, LANG_ADAPTER)


def create_db(preserve_persistent=False):
    """Create fresh database with schema.

    If preserve_persistent=True, keeps decisions and function_summaries tables
    (marks summaries as stale instead of deleting).
    """
    if DB_PATH.exists():
        if preserve_persistent:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            # Preserve persistent tables
            persistent_tables = {"decisions", "function_summaries"}
            existing = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            # Mark summaries stale
            if "function_summaries" in existing:
                conn.execute("UPDATE function_summaries SET stale = 1")
                conn.commit()
            # Save persistent data
            saved = {}
            for tbl in persistent_tables:
                if tbl in existing:
                    cur = conn.execute(f"SELECT * FROM {tbl}")
                    rows = cur.fetchall()
                    col_names = [d[0] for d in cur.description]
                    saved[tbl] = (col_names, rows)
            conn.close()
            DB_PATH.unlink()
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA)
            # Restore persistent data
            for tbl, (cols, rows) in saved.items():
                if rows:
                    placeholders = ",".join("?" * len(cols))
                    col_list = ",".join(cols)
                    conn.executemany(
                        f"INSERT OR IGNORE INTO {tbl} ({col_list}) VALUES ({placeholders})",
                        [tuple(r) for r in rows]
                    )
            conn.commit()
            return conn
        else:
            DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def populate_graph_metadata(conn):
    """Store graph schema/build provenance for drift checks and debugging."""
    revision = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )
        revision = result.stdout.strip() or None
    except Exception:
        revision = None

    conn.execute(
        "INSERT INTO graph_metadata (schema_version, extractor_version, graph_build_revision, built_at, repo_root) VALUES (?,?,?,?,?)",
        (
            SCHEMA_VERSION,
            EXTRACTOR_VERSION,
            revision,
            datetime.now(timezone.utc).isoformat(),
            str(REPO_ROOT),
        )
    )


# ============================================================
# DATA POPULATION — All known entities from graph
# ============================================================

def populate_singletons(conn):
    """Insert all 120+ singletons (Game + Common)."""
    data = [
        # Core Infrastructure
        ("sGameServer", "GameServer", "GameServer.h", "Game", "core", "Main server config and state"),
        ("sMain", "MainApplication", "MainApp.h", "Game", "core", "Application entry point and main loop"),
        ("sObjectMgr", "CObjectMgr", "ObjectManager.h", "Game", "core", "Find players/monsters by ID, character map"),
        ("sWorldMgr", "WorldMgr", "WorldManager.h", "Game", "core", "World data loader, map flags, exp rates"),
        ("sMonsterManager", "MonsterManager", "MonsterManager.h", "Game", "core", "Monster templates, skills, AI configs, respawn"),
        # Inter-Server
        ("sAuthServer", "AuthServer", "AuthServer.h", "Game", "inter_server", "LoginServer communication, auth"),
        ("sConnectServer", "ConnectServer", "ConnectServer.h", "Game", "inter_server", "ConnectServer comm, channels"),
        ("sServerLink", "ServerLink", "ServerLink.h", "Game", "inter_server", "Inter-server communication hub"),
        ("sServerToServer", "ServerToServer", "ServerToServer.h", "Game", "inter_server", "Server-to-server direct messaging"),
        ("sPacketRecorder", "PacketRecorder", "PacketRecorder.h", "Game", "inter_server", "Packet recording and replay"),
        # Player & Social
        ("sCharacterBase", "CharacterBaseMgr", "CharacterBase.h", "Game", "player_social", "Character data structures"),
        ("sGuildMgr", "CGuildMgr", "GuildMgr.h", "Game", "player_social", "Guild management"),
        ("sPartyMgr", "PartyMgr", "PartyMgr.h", "Game", "player_social", "Party management"),
        ("sGuildMatching", "GuildMatching", "GuildMatching.h", "Game", "player_social", "Guild matching system"),
        ("sPartyMatching", "PartyMatching", "PartyMatching.h", "Game", "player_social", "Party matching system"),
        ("sGenMgr", "CGenMgr", "GenMgr.h", "Game", "player_social", "Loyalty faction system"),
        ("sDuelMgr", "DuelMgr", "DuelMgr.h", "Game", "player_social", "Duel management"),
        ("sDuelBetItem", "CDuelBetIem", "DuelBetItem.h", "Game", "player_social", "Duel bet items"),
        ("sGuildWarMgr", "GuildWarMgr", "GuildWar.h", "Game", "player_social", "Guild wars"),
        # Items & Equipment
        ("sItemMgr", "CItemMgr", "ItemMgr.h", "Game", "item", "Item loading, templates, management"),
        ("sMixMgr", "MixMgr", "MixMgr.h", "Game", "item", "Mix/craft system"),
        ("sJewelMix", "JewelMix", "JewelMix.h", "Game", "item", "Jewel combination"),
        ("sPentagramSystem", "PentagramSystem", "PentagramSystem.h", "Game", "item", "Pentagram system"),
        ("sSocketSystem", "SocketSystem", "SocketSystem.h", "Game", "item", "Socket item system"),
        ("sWingSystem", "WingSystem", "WingSystem.h", "Game", "item", "Wing system"),
        ("sExcellentSystem", "ExcellentSystem", "ExcellentSystem.h", "Game", "item", "Excellent option system"),
        ("sNewLegendarySystem", "NewLegendarySystem", "NewLegendarySystem.h", "Game", "item", "Legendary item system"),
        ("sNewItemBagSystem", "NewItemBagSystem", "NewItemBagSystem.h", "Game", "item", "Item bag system"),
        ("sItemDisassambly", "ItemDisassembleSystem", "ItemDisassembleSystem.h", "Game", "item", "Item disassemble"),
        ("sCashShopMgr", "CashShopMgr", "CashShop.h", "Game", "item", "Cash shop microtransactions"),
        ("sPetExperience", "CPetItemExperience", "Item.h", "Game", "item", "Pet item experience"),
        ("sDropSystem", "DropSystem", "DropSystem.h", "Game", "item", "Drop system"),
        ("sCustomJewelsManager", "CustomJewelsManager", "CustomJewelsManager.h", "Game", "item", "Custom jewels"),
        # Skills & Combat
        ("sSkillMgr", "SkillMgr", "SkillMgr.h", "Game", "skill", "Skill data management"),
        ("sNewSkillSystem", "FiveClassSkillSystem", "FiveClassSkillSystem.h", "Game", "skill", "5th class skill system"),
        ("sMasteryChangeOption", "MasteryChangeOption", "MasteryChangeOption.h", "Game", "skill", "Mastery change options"),
        # VIP & Progression
        ("sVipMgr", "CVipMgr", "VIPMgr.h", "Game", "vip", "VIP level system"),
        ("sResetSystemMgr", "CResetMgr", "ResetSystemMgr.h", "Game", "vip", "Reset system manager"),
        ("sResetSystem", "ResetSystem", "ResetSystem.h", "Game", "vip", "Reset system"),
        ("sDynamicExperienceMgr", "CDynamicExpMgr", "DynamicExperienceMgr.h", "Game", "vip", "Dynamic exp rates"),
        ("sLevelUpCompensationMgr", "LevelUpCompensationMgr", "LevelUpCompensation.h", "Game", "vip", "Level-up compensation"),
        ("sExpConfigs", "ExpConfigs", "ExpConfigs.h", "Game", "vip", "Exp rate configs"),
        ("sHuntingRecord", "HuntingRecord", "HuntingRecord.h", "Game", "vip", "Hunting records"),
        ("sCustomRankTitle", "CCustomRankTitle", "CustomRankTitle.h", "Game", "vip", "Custom rank titles"),
        ("sCustomRankServer", "CCustomRankServer", "CustomRankServer.h", "Game", "vip", "Custom rank server"),
        # Quest & Event Management
        ("sQuestMgr", "CQuestMgr", "QuestMgr.h", "Game", "quest", "Quest data management"),
        ("sEventMgr", "CEventMgr", "EventManager.h", "Game", "quest", "Event scheduling engine"),
        ("sEventInventory", "EventInventory", "EventInventory.h", "Game", "quest", "Event inventory system"),
        ("sNewEventWindow", "NewEventWindow", "NewEventWindow.h", "Game", "quest", "New event window UI"),
        ("sMessage", "SystemMessage", "System_Message.h", "Game", "quest", "System messages"),
        ("sNoticeSystem", "CNoticeSystem", "NoticeSystem.h", "Game", "quest", "Notice system"),
        ("sRuudSystem", "RuudSystem", "RuudSystem.h", "Game", "quest", "Ruud currency system"),
        ("sPlayTimeEventMgr", "PlayTimeEventMgr", "PlayTimeEvent.h", "Game", "quest", "Play time events"),
        ("sHappyHour", "HappyHour", "HappyHour.h", "Game", "quest", "Happy hour boost events"),
        ("sMuPassEvent", "MuPassEvent", "MuPassEvent.h", "Game", "quest", "Battle pass system"),
        # Events/Dungeons
        ("sBloodCastleMgr", "CBloodCastleMgr", "BloodCastle.h", "Game", "event", "Blood Castle dungeon"),
        ("sDevilSquareMgr", "DevilSquareMgr", "DevilSquare.h", "Game", "event", "Devil Square event"),
        ("sChaosCastleMgr", "CChaosCastleMgr", "ChaosCastle.h", "Game", "event", "Chaos Castle event"),
        ("sChaosCastleSurvivalMgr", "ChaosCastleSurvivalMgr", "ChaosCastleSurvival.h", "Game", "event", "Chaos Castle Survival"),
        ("sImperialFortressMgr", "ImperialFortressMgr", "ImperialFortress.h", "Game", "event", "Imperial Fortress"),
        ("sCastleSiege", "CCastleSiege", "CastleSiege.h", "Game", "event", "Castle Siege warfare"),
        ("sIllusionTemple", "IllusionTemple", "IllusionTemple.h", "Game", "event", "Illusion Temple"),
        ("sDoppelganger", "Doppelganger", "Doppelganger.h", "Game", "event", "Doppelganger dungeon"),
        ("sInvasionMgr", "InvasionMgr", "Invasion.h", "Game", "event", "Monster invasion events"),
        ("sCrywolf", "Crywolf", "Crywolf.h", "Game", "event", "Crywolf altar defense"),
        ("sKanturuMgr", "KanturuMgr", "Kanturu.h", "Game", "event", "Kanturu 3-stage tower"),
        ("sRaklion", "Raklion", "Raklion.h", "Game", "event", "Raklion event"),
        ("sDungeon", "Dungeon", "Dungeon.h", "Game", "event", "General dungeon framework"),
        ("sAllBossEvent", "AllBossTogetherEvent", "AllBossTogetherEvent.h", "Game", "event", "All Boss Together event"),
        ("sBattleSoccerMgr", "CBattleSoccerMgr", "BattleSoccer.h", "Game", "event", "Battle Soccer mini-game"),
        ("sProtectorOfAcheron", "ProtectorOfAcheron", "ProtectorOfAcheron.h", "Game", "event", "Acheron raid boss"),
        ("sTormentedSquare", "TormentedSquare", "TormentedSquare.h", "Game", "event", "Tormented Square waves"),
        ("sTormentedSquareSurvival", "TormentedSquareSurvival", "TormentedSquareSurvival.h", "Game", "event", "Tormented Square Survival"),
        ("sGremoryCase", "GremoryCase", "GremoryCase.h", "Game", "event", "Gremory Case item container"),
        ("sEvomon", "Evomon", "Evomon.h", "Game", "event", "Evomon evolution event"),
        ("sLastManStanding", "LastManStanding", "LastManStanding.h", "Game", "event", "Last Man Standing FFA"),
        ("sMiniBomb", "MiniBomb", "MiniBomb.h", "Game", "event", "MiniBomb mini-game"),
        ("sJewelBingo", "JewelBingo", "JewelBingo.h", "Game", "event", "Jewel Bingo lottery"),
        ("sMossMerchant", "MossMerchant", "MossMerchant.h", "Game", "event", "Moss Merchant NPC"),
        ("sSwampOfDarkness", "SwampOfDarkness", "SwampOfDarkness.h", "Game", "event", "Swamp of Darkness"),
        ("sNumericBaseball", "NumericBaseball", "NumericBaseball.h", "Game", "event", "Numeric Baseball mini-game"),
        ("sLabyrinthDimensions", "LabyrinthDimensions", "LabyrinthDimensions.h", "Game", "event", "Labyrinth of Dimensions"),
        ("sNixiesLake", "NixiesLake", "NixiesLake.h", "Game", "event", "Nixies Lake dungeon"),
        ("sWorldBoss", "WorldBoss", "WorldBoss.h", "Game", "event", "Open-world raid bosses"),
        ("sArkaWar", "ArkaWar", "ArkaWar.h", "Game", "event", "Arka War PvP faction war"),
        ("sScramble", "Scramble", "Scramble.h", "Game", "event", "Scramble box collection"),
        ("sCastleDeep", "CastleDeep", "CastleDeep.h", "Game", "event", "Castle Deep multi-floor dungeon"),
        # Misc Game singletons
        ("sTeleport", "TeleportManager", "TeleportManager.h", "Game", "misc", "Teleport/gate system"),
        ("sShopMgr", "ShopMgr", "ShopMgr.h", "Game", "misc", "NPC shop management"),
        ("sFormulaMgr", "FormulaMgr", "FormulaData.h", "Game", "misc", "Damage/exp formulas"),
        ("sCommandMgr", "CommandMgr", "CommandMgr.h", "Game", "misc", "Admin/GM commands"),
        ("sParticleMgr", "ParticleMgr", "ParticleMgr.h", "Game", "misc", "Particle effects"),
        ("sMiniMap", "MiniMap", "MiniMap.h", "Game", "misc", "Mini-map system"),
        ("sSummonScroll", "SummonScroll", "SummonScroll.h", "Game", "misc", "Summon scroll system"),
        ("sArtifactmMgr", "ArtifactMgr", "ArtifactMgr.h", "Game", "misc", "Artifact system"),
        ("sBlessingBoxMgr", "BlessingBoxMgr", "BlessingBox.h", "Game", "misc", "Blessing box rewards"),
        ("sHelperPlusMgr", "HelperPlusMgr", "HelperPlusMgr.h", "Game", "misc", "Helper Plus auto-hunt zones"),
        ("sMonsterSoul", "MonsterSoul", "MonsterSoul.h", "Game", "misc", "Monster soul converter"),
        ("sScriptAI", "ScriptAIMgr", "ScriptAI.h", "Game", "misc", "Script-based AI"),
        ("sMiningSystem", "MiningSystem", "MiningSystem.h", "Game", "misc", "Mining/gathering system"),
        ("sMuRoomy", "MuRoomy", "MuRoomy.h", "Game", "misc", "Mu Room event"),
        ("sRegExMngr", "RegExMngr", "RegExMngr.h", "Game", "misc", "Regex manager"),
        ("sHttpRequest", "MyHttpRequest", "MyHttpRequest.h", "Game", "misc", "HTTP request handler"),
        ("sDiscord", "Discord", "Discord.h", "Game", "misc", "Discord webhook integration"),
        ("sScriptLoader", "CScriptLoader", "CScriptLoader.h", "Game", "misc", "Script file loader"),
        ("sClientLuaMgr", "ClientLuaMgr", "ClientLuaMgr.h", "Game", "misc", "Client Lua bridge"),
        ("sPath", "CPath", "CPath.h", "Game", "misc", "File path manager"),
        ("sSGlobals", "GSGlobalObjects", "GSGlobalObjects.h", "Game", "misc", "Global objects container"),
        ("sNewExpRecoverySystem", "NewExpRecoverySystem", "NewExpRecoverySystem.h", "Game", "misc", "Exp recovery system"),
        ("sDungeonRace", "DungeonRace", "DungeonRace.h", "Game", "misc", "Dungeon race event"),
        ("sLosttowerRace", "LosttowerRace", "LosttowerRace.h", "Game", "misc", "Lost Tower race"),
        ("sPartyUpdateQueue", "PartyUpdateQueue", "GamePCH.h", "Game", "misc", "Party update queue"),
        # Common singletons
        ("sLog", "Log", "Log.h", "Common", "common", "Async logging system"),
        ("sEncDec", "EncodeDecode", "EncDec.h", "Common", "common", "XOR packet encryption"),
        ("sLargeRandom", "CLargeRandom", "LargeRandom.h", "Common", "common", "MT19937 RNG"),
        ("sSecurity", "Security", "Security.h", "Common", "common", "MAC + system security"),
        ("sCustomUtil", "CCustomUtil", "Util.h", "Common", "common", "Utility functions"),
        ("sOpcodeMgr", "OpcodeMgr", "CustomPacket.h", "Common", "common", "Opcode management"),
        ("sMiniDump", "MiniDump", "MiniDump.h", "Common", "common", "Crash dump generation"),
        ("sConfig", "Config", "Config.h", "Common", "common", "INI config reader"),
    ]
    conn.executemany(
        "INSERT INTO singletons (macro, class_name, header, project, category, description) VALUES (?,?,?,?,?,?)",
        data
    )


def populate_player_files(conn):
    """Insert Player partial file mapping."""
    data = [
        ("Player.cpp", "core", "Player(), ~Player(), protocol_core(), PlayerSaveTransaction", "Core lifecycle, protocol dispatch, save transaction"),
        ("PlayerDB.cpp", "database", "LoadDBNew(), LoadDBInfoNew(), LoadDBItemNew(), LoadDBSkillNew()", "Database loading (~30 methods)"),
        ("PlayerSocket.cpp", "network", "sendPacket(), SendPacket(), CloseSocket(), kick()", "Network I/O"),
        ("PlayerSkillUse.cpp", "combat", "NormalAttack(), NormalMagicAttack()", "Combat/skill execution"),
        ("PlayerCharacter.cpp", "init", "ClearCharacter()", "Character init/reset"),
        ("PlayerChat.cpp", "chat", "ChatRequest(), ChatProcess()", "Chat routing"),
        ("PlayerTrade.cpp", "trade", "TradeRequest(), TradeRequestAnswer(), TradeBegin()", "Player-to-player trade"),
        ("PlayerParty.cpp", "party", "PartyRequest()", "Party system"),
        ("PlayerGuild.cpp", "guild", "GuildCreateRequest(), GuildJoinRequest/Result()", "Guild operations"),
        ("PlayerDuel.cpp", "duel", "DuelRequest(), DuelRequestAnswer()", "PVP duels"),
        ("PlayerPVP.cpp", "pvp", "AttackAllowedToPlayer()", "PVP rules (29+ world checks)"),
        ("PlayerFriend.cpp", "friend", "IsFriend(), AddFriend(), FriendAddRequest()", "Friends list (max 50)"),
        ("PlayerMix.cpp", "mix", "ChaosMixButton(), HarmonyMixConfirmOption()", "Crafting/alchemy"),
        ("PlayerCashShop.cpp", "cashShop", "CashShopSendPath(), CashShopGetItem()", "Microtransactions"),
        ("PlayerHelper.cpp", "helper", "HelperSendSettings(), HandleHelperPlusRun()", "Auto-hunt helper + HelperPlus"),
        ("PlayerDarkSpirit.cpp", "darkSpirit", "DarkSpiritCalculate(), DarkSpiritRun()", "Dark Lord pet AI"),
        ("PlayerGen.cpp", "gen", "GenJoinRequest(), GenJoinResult()", "Loyalty factions"),
        ("PlayerArtifact.cpp", "artifact", "ArtifactInsertIn(), GetFullsetStatus()", "Artifact equipment"),
        ("PlayerInterface.cpp", "interface", "InterfaceSharedCheck(), TransactionSerialCheck()", "UI window state"),
        ("PlayerSkillTree.cpp", "skillTree", "MasterSendStatus(), MasterSkillPointAdd()", "Master skill tree"),
        ("PlayerMajesticSkillTree.cpp", "majestic", "MajesticSkillTreeSend()", "Majestic (4th class) skills"),
        ("PlayerQuestEvolution.cpp", "questEvo", "QuestEvolutionUpdateState()", "Class evolution quests"),
        ("PlayerQuestGuided.cpp", "questGuided", "QuestGuidedSend(), QuestGuidedMonsterKill()", "Tutorial quests"),
        ("PlayerQuestMU.cpp", "questMU", "GetQuestMU(), AddQuestMU()", "Open-world quests"),
        ("PlayerPersonalStore.cpp", "personalStore", "PersonalStoreClose(), UpdatePersonalStore()", "Offline shop"),
        ("PlayerMuPassEvent.cpp", "muPass", "EnableMuPassEvent(), SendMuPassEventData()", "Battle pass"),
        ("PlayerStatFruit.cpp", "statFruit", "UsePlusStatFruit(), UseStatFruitResult()", "Stat fruits"),
        ("PlayerWingMixSystem.cpp", "wingMix", "OpenWingMixWindow(), GrantWingOptions()", "Wing socket upgrades"),
    ]
    conn.executemany(
        "INSERT INTO player_files (file, domain, key_methods, description) VALUES (?,?,?,?)",
        data
    )


def populate_events(conn):
    """Insert all event/dungeon systems."""
    data = [
        ("AllBossTogether", "sAllBossEvent", "AllBossTogetherEvent.cpp", "AllBossTogetherEvent.h", None, "ai_all_boss_together.cpp", "4 bosses simultaneous spawn"),
        ("ArkaWar", "sArkaWar", "ArkaWar.cpp", "ArkaWar.h", None, "ai_arka_war.cpp", "PvP faction war"),
        ("BattleSoccer", "sBattleSoccerMgr", "BattleSoccer.cpp", "BattleSoccer.h", "BattleSoccerDef.h", "ai_soccer_ball.cpp", "Soccer mini-game"),
        ("BloodCastle", "sBloodCastleMgr", "BloodCastle.cpp", "BloodCastle.h", "BloodCastleDef.h", "ai_blood_castle.cpp", "Gate-Trap-Boss dungeon"),
        ("CastleDeep", "sCastleDeep", "CastleDeep.cpp", "CastleDeep.h", None, "ai_castle_deep.cpp", "Multi-floor dungeon"),
        ("CastleSiege", "sCastleSiege", "CastleSiege.cpp", "CastleSiege.h", "CastleSiegeDef.h", "ai_castle_siege.cpp", "Guild siege warfare"),
        ("ChaosCastle", "sChaosCastleMgr", "ChaosCastle.cpp", "ChaosCastle.h", "ChaosCastleDef.h", "ai_chaos_castle.cpp", "Kill-count event"),
        ("ChaosCastleSurvival", "sChaosCastleSurvivalMgr", "ChaosCastleSurvival.cpp", "ChaosCastleSurvival.h", None, None, "Survival waves"),
        ("Crywolf", "sCrywolf", "Crywolf.cpp", "Crywolf.h", "CrywolfDef.h", "ai_crywolf.cpp", "Altar defense event"),
        ("DevilSquare", "sDevilSquareMgr", "DevilSquare.cpp", "DevilSquare.h", "DevilSquareDef.h", "ai_devil_square.cpp", "Kill-counter event"),
        ("Doppelganger", "sDoppelganger", "Doppelganger.cpp", "Doppelganger.h", None, "ai_doppelganger.cpp", "Clone boss dungeon"),
        ("Dungeon", "sDungeon", "Dungeon.cpp", "Dungeon.h", "DungeonDef.h", None, "General dungeon framework"),
        ("DungeonInstance", None, "DungeonInstance.cpp", "DungeonInstance.h", None, "ai_instanced_dungeon.cpp", "Private party instances"),
        ("DungeonRace", "sDungeonRace", "DungeonRace.cpp", "DungeonRace.h", None, None, "Timed race event"),
        ("Evomon", "sEvomon", "Evomon.cpp", "Evomon.h", None, "ai_evomon.cpp", "Evolution boss event"),
        ("HappyHour", "sHappyHour", "HappyHour.cpp", "HappyHour.h", None, None, "Exp/Zen/Drop boost"),
        ("IllusionTemple", "sIllusionTemple", "IllusionTemple.cpp", "IllusionTemple.h", "IllusionTempleDef.h", None, "Puzzle dungeon"),
        ("ImperialFortress", "sImperialFortressMgr", "ImperialFortress.cpp", "ImperialFortress.h", "ImperialFortressDef.h", "ai_imperial_fortress.cpp", "5-zone fortress"),
        ("Invasion", "sInvasionMgr", "Invasion.cpp", "Invasion.h", None, "ai_invasion.cpp", "Random monster invasion"),
        ("JewelBingo", "sJewelBingo", "JewelBingo.cpp", "JewelBingo.h", None, None, "Jewel lottery game"),
        ("Kanturu", "sKanturuMgr", "Kanturu.cpp", "Kanturu.h", "KanturuDef.h", "ai_kanturu.cpp", "3-stage tower event"),
        ("LabyrinthDimensions", "sLabyrinthDimensions", "LabyrinthDimensions.cpp", "LabyrinthDimensions.h", "LabyrinthDimensionsDef.h", "ai_labyrinth_of_dimensions.cpp", "Chaos maze event"),
        ("LastManStanding", "sLastManStanding", "LastManStanding.cpp", "LastManStanding.h", None, None, "Battle royale FFA"),
        ("LosttowerRace", "sLosttowerRace", "LosttowerRace.cpp", "LosttowerRace.h", None, None, "Lost Tower ascent"),
        ("MiniBomb", "sMiniBomb", "MiniBomb.cpp", "MiniBomb.h", None, None, "Bomb mini-game"),
        ("MiningSystem", "sMiningSystem", "MiningSystem.cpp", "MiningSystem.h", None, None, "Item mining/gathering"),
        ("MossMerchant", "sMossMerchant", "MossMerchant.cpp", "MossMerchant.h", None, "ai_moss_merchant.cpp", "Rare item exchange NPC"),
        ("MuRoomy", "sMuRoomy", "MuRoomy.cpp", "MuRoomy.h", None, None, "Room event mini-game"),
        ("NixiesLake", "sNixiesLake", "NixiesLake.cpp", "NixiesLake.h", None, "ai_nixies_lake.cpp", "Water hazard dungeon"),
        ("ProtectorOfAcheron", "sProtectorOfAcheron", "ProtectorOfAcheron.cpp", "ProtectorOfAcheron.h", None, "ai_protector_of_acheron.cpp", "Raid boss"),
        ("Scramble", "sScramble", "Scramble.cpp", "Scramble.h", "ScrambleDef.h", None, "Box collection event"),
        ("SwampOfDarkness", "sSwampOfDarkness", "SwampOfDarkness.cpp", "SwampOfDarkness.h", None, "ai_swamp_of_darkness.cpp", "Poison/trap dungeon"),
        ("TormentedSquare", "sTormentedSquare", "TormentedSquare.cpp", "TormentedSquare.h", "TormentedSquareDef.h", "ai_tormented_square.cpp", "Wave dungeon"),
        ("TormentedSquareSurvival", "sTormentedSquareSurvival", "TormentedSquareSurvival.cpp", "TormentedSquareSurvival.h", None, None, "Survival mode"),
        ("WorldBoss", "sWorldBoss", "WorldBoss.cpp", "WorldBoss.h", None, "ai_world_boss.cpp", "Open-world raid bosses"),
    ]
    conn.executemany(
        "INSERT INTO events (name, singleton_macro, cpp_file, header_file, def_file, ai_file, description) VALUES (?,?,?,?,?,?,?)",
        data
    )


def populate_ai_handlers(conn):
    """Insert AI handler files."""
    data = [
        ("ai_all_boss_together.cpp", "AllBossTogether", "All Boss Together event mob AI"),
        ("ai_arka_war.cpp", "ArkaWar", "Arka War event AI"),
        ("ai_blood_castle.cpp", "BloodCastle", "Blood Castle gates/statues/bosses"),
        ("ai_castle_deep.cpp", "CastleDeep", "Castle Deep instance mobs"),
        ("ai_castle_siege.cpp", "CastleSiege", "Castle Siege NPCs & defense"),
        ("ai_champion.cpp", "Champion", "Champion/boss monster specials"),
        ("ai_chaos_castle.cpp", "ChaosCastle", "Chaos Castle mobs"),
        ("ai_crywolf.cpp", "Crywolf", "Crywolf altar guardians"),
        ("ai_custom_boss.cpp", "CustomBoss", "Custom boss encounters"),
        ("ai_devil_square.cpp", "DevilSquare", "Devil Square spawns"),
        ("ai_doppelganger.cpp", "Doppelganger", "Doppelganger dungeon mobs"),
        ("ai_evomon.cpp", "Evomon", "Evomon evolution bosses"),
        ("ai_ferea.cpp", "Ferea", "Ferea raid boss (Gods of Darkness)"),
        ("ai_ground_darkness.cpp", "GroundDarkness", "Swamp of Darkness mobs"),
        ("ai_guard.cpp", "Guard", "Guard NPCs (gates, turrets)"),
        ("ai_imperial_fortress.cpp", "ImperialFortress", "Imperial Fortress traps & mobs"),
        ("ai_instanced_dungeon.cpp", "InstancedDungeon", "Dungeon instance spawning"),
        ("ai_invasion.cpp", "Invasion", "Invasion event mobs"),
        ("ai_kalima_gate.cpp", "KalimaGate", "Kalima gate guardians"),
        ("ai_kanturu.cpp", "Kanturu", "Kanturu event enemies"),
        ("ai_kundun.cpp", "Kundun", "Kundun final boss mechanics"),
        ("ai_labyrinth_of_dimensions.cpp", "LabyrinthDimensions", "Labyrinth maze mobs"),
        ("ai_majestic_debuff.cpp", "MajesticDebuff", "Majestic skill debuffs"),
        ("ai_medusa.cpp", "Medusa", "Medusa raid boss"),
        ("ai_moss_merchant.cpp", "MossMerchant", "Moss Merchant NPC AI"),
        ("ai_moving_npc.cpp", "MovingNPC", "NPCs with patrol routes"),
        ("ai_nars.cpp", "Nars", "Nars vendor AI"),
        ("ai_nixies_lake.cpp", "NixiesLake", "Nixie's Lake mobs"),
        ("ai_personal_merchant.cpp", "PersonalMerchant", "Personal store NPCs"),
        ("ai_protector_of_acheron.cpp", "ProtectorOfAcheron", "Acheron raid boss"),
        ("ai_quest.cpp", "Quest", "Quest-specific mob behaviors"),
        ("ai_raklion.cpp", "Raklion", "Raklion event mobs"),
        ("ai_soccer_ball.cpp", "SoccerBall", "Battle Soccer ball physics"),
        ("ai_special_map.cpp", "SpecialMap", "Special map mobs"),
        ("ai_summon.cpp", "Summon", "Summoned creature AI"),
        ("ai_summoner_debuff.cpp", "SummonerDebuff", "Summoner debuff mechanics"),
        ("ai_summon_player.cpp", "SummonPlayer", "Player summon (DL) AI"),
        ("ai_swamp_of_darkness.cpp", "SwampOfDarkness", "Swamp water/traps"),
        ("ai_tormented_square.cpp", "TormentedSquare", "Tormented Square wave mobs"),
        ("ai_trap.cpp", "Trap", "Environmental trap triggers"),
        ("ai_uruk_mountain.cpp", "UrukMountain", "Uruk Mountain mobs"),
        ("ai_world_boss.cpp", "WorldBoss", "World Boss AI"),
        ("ai_world_boss_new.cpp", "WorldBossNew", "New Gen World Boss AI"),
    ]
    conn.executemany(
        "INSERT INTO ai_handlers (file, target, description) VALUES (?,?,?)",
        data
    )


def populate_inventory_scripts(conn):
    """Insert inventory/script files."""
    data = [
        ("InventoryScript.cpp", "Main player inventory: item grid, wear slots"),
        ("WarehouseScript.cpp", "Bank warehouse multi-tab storage"),
        ("EventInventoryScript.cpp", "Event temporary bags"),
        ("ArtifactInventoryScript.cpp", "Artifact bag slots"),
        ("ExpRecoveryInventoryScript.cpp", "Exp recovery items"),
        ("GremoryCaseScript.cpp", "Gremory Case container"),
        ("MuunScript.cpp", "Muun pet inventory"),
        ("SkillBookInventoryScript.cpp", "Skill book collection"),
        ("WingCoreInventoryScript.cpp", "Wing core container"),
        ("PersonalStoreScript.cpp", "Player shop display"),
        ("NPCSellScript.cpp", "NPC shop layout"),
        ("StoreScript.cpp", "Base storage class (parent)"),
    ]
    conn.executemany(
        "INSERT INTO inventory_scripts (file, description) VALUES (?,?)",
        data
    )


def populate_constants(conn):
    """Insert key constants and defines."""
    data = [
        # CommonDef.h
        ("MAX_VIEWPORT", "120", "CommonDef.h", "limits", "Max viewport objects"),
        ("normal_inventory_size", "64", "CommonDef.h", "limits", "Normal inventory slots"),
        ("inventory_size", "239", "CommonDef.h", "limits", "Total inventory size"),
        ("SkillBookInventorySize", "540", "CommonDef.h", "limits", "Skill book inventory"),
        ("EVENT_INVENTORY_SIZE", "32", "CommonDef.h", "limits", "Event inventory size"),
        # Common.h / Define.h
        ("MAX_BUFFER_SIZE", "524288", "Common.h", "limits", "Network I/O buffer (512KB)"),
        ("MAX_QUERY_LEN", "65536", "Common.h", "limits", "Single SQL query size (64KB)"),
        ("MAX_CHARACTER_LENGTH", "10", "Common.h", "limits", "Character name max"),
        ("MAX_ACCOUNT_LENGTH", "10", "Common.h", "limits", "Account name max"),
        ("MAX_GUILD_MEMBER", "80", "Common.h", "limits", "Guild size cap"),
        ("MAX_CHARACTER_PER_ACCOUNT", "15", "Common.h", "limits", "Characters per account"),
        ("MAX_MULTI_WAREHOUSE", "5", "Common.h", "limits", "Warehouse slots (normal)"),
        ("MAX_SERVER_PER_GROUP", "20", "Common.h", "limits", "Channels per server group"),
        # BuffDef.h
        ("MAX_BUFF", "32", "BuffDef.h", "limits", "Max active buffs"),
        # PlayerDef.h
        ("FRIEND_MAX", "50", "PlayerDef.h", "limits", "Max friends"),
        ("MAIL_MAX", "150", "PlayerDef.h", "limits", "Max mails"),
        ("MAX_SKILL", "60", "SkillDef.h", "limits", "Playable skill slots"),
        # Key buff IDs
        ("BUFF_GREATER_ATTACK", "1", "BuffDef.h", "buff", "Greater attack buff"),
        ("BUFF_GREATER_DEFENSE", "2", "BuffDef.h", "buff", "Greater defense buff"),
        ("BUFF_MANA_SHIELD", "4", "BuffDef.h", "buff", "Mana shield"),
        ("BUFF_POISON", "55", "BuffDef.h", "buff", "Poison debuff"),
        ("BUFF_ICE", "56", "BuffDef.h", "buff", "Ice debuff"),
        ("BUFF_STUN", "61", "BuffDef.h", "buff", "Stun debuff"),
        # World IDs
        ("WORLD_LORENCIA", "0", "WorldDef.h", "world", "Lorencia"),
        ("WORLD_DUNGEON", "1", "WorldDef.h", "world", "Dungeon"),
        ("WORLD_DEVIAS", "2", "WorldDef.h", "world", "Devias"),
        ("WORLD_NORIA", "3", "WorldDef.h", "world", "Noria"),
        ("WORLD_LOSTTOWER", "4", "WorldDef.h", "world", "Lost Tower"),
        ("WORLD_ATLANS", "7", "WorldDef.h", "world", "Atlans"),
        ("WORLD_TARKAN", "8", "WorldDef.h", "world", "Tarkan"),
        ("WORLD_DEVIL_SQUARE", "9", "WorldDef.h", "world", "Devil Square"),
        ("WORLD_ICARUS", "10", "WorldDef.h", "world", "Icarus"),
        ("WORLD_BLOOD_CASTLE_1", "11", "WorldDef.h", "world", "Blood Castle 1"),
        ("WORLD_CASTLE_SIEGE", "30", "WorldDef.h", "world", "Castle Siege"),
        ("WORLD_CRYWOLF", "34", "WorldDef.h", "world", "Crywolf"),
        ("WORLD_AIDA", "33", "WorldDef.h", "world", "Aida"),
        ("WORLD_ELBELAND", "51", "WorldDef.h", "world", "Elbeland"),
    ]
    conn.executemany(
        "INSERT INTO constants (name, value, header, category, description) VALUES (?,?,?,?,?)",
        data
    )


def populate_config_files(conn):
    """Insert config file paths."""
    data = [
        ("ServerMap.xml", "Server topology, ports, default spawn"),
        ("FilterText.xml", "Profanity filter"),
        ("Notice.xml", "Server notices"),
        ("GoblinPoint.xml", "Goblin Point level restrictions"),
        ("Commands/Commands.xml", "Admin/GM commands"),
        ("Commands/LuaCommands.xml", "Lua commands"),
        ("Lang/LangBase.xml", "Server language strings"),
        ("Lang/ClientTexts.xml", "Client text strings"),
        ("FormulaData.xml", "Damage/exp formulas"),
        ("World/WorldTemplate.xml", "World definitions & properties"),
        ("World/WorldAI.xml", "World AI path data"),
        ("World/WorldBuff.xml", "World-specific buffs"),
        ("World/WorldAreaRestriction.xml", "Area access rules"),
        ("World/WorldAttribute.xml", "Map tile attributes"),
        ("World/WorldExpParty.xml", "Party exp bonuses"),
        ("Item/ItemList.xml", "Item database"),
        ("Item/ItemDecomposition.xml", "Decomposition recipes"),
        ("Artifact/ArtifactSystem.xml", "Artifact mechanics"),
        ("EventData/HelperPlus.xml", "Helper Plus zones/spots"),
        ("game_common.conf", "Main server INI config"),
    ]
    conn.executemany(
        "INSERT INTO config_files (path, description) VALUES (?,?)",
        data
    )


def populate_packet_handlers(conn):
    """Insert known packet handler mappings."""
    # Client handlers (Player::protocol_core)
    client = [
        ("HEADCODE_ACCOUNT_DATA_IN", 67, "0x43", "LoginRequest", "Player.cpp", "client", "account", "Account login"),
        ("HEADCODE_MISC_CHARACTER_DATA_IN", 66, "0x42", "CharacterCreate/Delete/Select", "Player.cpp", "client", "account", "Character management"),
        ("HEADCODE_MOVE_IN", 215, "0xD7", "CharacterMove", "Player.cpp", "client", "movement", "Player movement"),
        ("HEADCODE_POSITION_SET_IN", 16, "0x10", "PositionSet", "Player.cpp", "client", "movement", "Position sync"),
        ("HEADCODE_ATTACK_NORMAL_IN", 223, "0xDF", "NormalAttack", "Player.cpp", "client", "combat", "Normal attack"),
        ("HEADCODE_NORMAL_MAGIC_ATTACK_IN", 52, "0x34", "NormalMagicAttack", "Player.cpp", "client", "combat", "Magic attack"),
        ("HEADCODE_DURATION_MAGIC_ATTACK_IN", 30, "0x1E", "DurationMagicAttack", "Player.cpp", "client", "combat", "Duration skill"),
        ("HEADCODE_ATTACK_MULTI_TARGET_IN", 211, "0xD3", "MultiTargetMagicAttack", "Player.cpp", "client", "combat", "Multi-target attack"),
        ("HEADCODE_CANCEL_MAGIC_IN", 27, "0x1B", "MagicCancel", "Player.cpp", "client", "combat", "Cancel magic"),
        ("HEADCODE_ITEM_GET_IN", 178, "0xB2", "ItemGet", "Player.cpp", "client", "item", "Pick up item"),
        ("HEADCODE_ITEM_DROP_IN", 134, "0x86", "ItemDrop", "Player.cpp", "client", "item", "Drop item"),
        ("HEADCODE_ITEM_MOVE_IN", 97, "0x61", "ItemMove", "Player.cpp", "client", "item", "Move item"),
        ("HEADCODE_ITEM_USE_IN", 150, "0x96", "ItemUse", "Player.cpp", "client", "item", "Use item"),
        ("HEADCODE_CHAT_IN", 114, "0x72", "ChatRequest", "Player.cpp", "client", "chat", "Chat message"),
        ("HEADCODE_WHISPER_IN", 60, "0x3C", "WhisperRequest", "Player.cpp", "client", "chat", "Whisper message"),
        ("HEADCODE_TRADE_REQUEST_IN", 75, "0x4B", "TradeRequest", "Player.cpp", "client", "trade", "Trade request"),
        ("HEADCODE_NPC_TALK_IN", 145, "0x91", "TalkToNpc", "Player.cpp", "client", "npc", "NPC interaction"),
        ("HEADCODE_CLOSE_INTERFACE_IN", 131, "0x83", "CloseInterface", "Player.cpp", "client", "npc", "Close NPC UI"),
        ("HEADCODE_PARTY_REQUEST_IN", 36, "0x24", "PartyRequest", "Player.cpp", "client", "party", "Party request"),
        ("HEADCODE_TIME_CHECK_IN", 50, "0x32", "Ping", "Player.cpp", "client", "misc", "Client keepalive"),
    ]
    conn.executemany(
        "INSERT INTO packet_handlers (headcode_name, headcode_value, headcode_hex, handler_method, source_file, handler_type, category, description) VALUES (?,?,?,?,?,?,?,?)",
        client
    )

    # Inter-server handlers (ServerLink)
    interserver = [
        ("HEADCODE_SERVER_LINK_ON_CONNECT", 0, "0x00", "HandleHeadcodeOnConnect", "ServerLink.cpp", "inter_server", "connection", "Handshake"),
        ("HEADCODE_SERVER_LINK_GUILD_CHAT", 1, "0x01", "HandleHeadcodeGuildChat", "ServerLink.cpp", "inter_server", "guild", "Guild chat sync"),
        ("HEADCODE_SERVER_LINK_ALLIANCE_CHAT", 2, "0x02", "HandleAllianceChat", "ServerLink.cpp", "inter_server", "guild", "Alliance chat sync"),
        ("HEADCODE_SERVER_LINK_GUILD_ADD", 7, "0x07", "GuildCreateResult", "ServerLink.cpp", "inter_server", "guild", "Guild create result"),
        ("HEADCODE_SERVER_LINK_GUILD_REMOVE", 8, "0x08", "GuildDeleteResult", "ServerLink.cpp", "inter_server", "guild", "Guild delete result"),
        ("HEADCODE_SERVER_LINK_CHARACTER_ON_OFF", 25, "0x19", "CharacterOnOff", "ServerLink.cpp", "inter_server", "character", "Online/offline status"),
        ("HEADCODE_SERVER_LINK_PARTY_DESTROY", 57, "0x39", "CharacterPartyDestroyHandler", "ServerLink.cpp", "inter_server", "party", "Party destroy sync"),
        ("HEADCODE_SERVER_LINK_PARTY_DELETE_MEMBER", 58, "0x3A", "CharacterPartyDeleteMemberHandler", "ServerLink.cpp", "inter_server", "party", "Party member delete"),
        ("HEADCODE_SERVER_LINK_EVENT_NOTIFICATION", 45, "0x2D", "EventNotification", "ServerLink.cpp", "inter_server", "event", "Event notification"),
    ]
    conn.executemany(
        "INSERT INTO packet_handlers (headcode_name, headcode_value, headcode_hex, handler_method, source_file, handler_type, category, description) VALUES (?,?,?,?,?,?,?,?)",
        interserver
    )

    # Login server handlers (AuthServer)
    login = [
        ("HEADCODE_LOGIN_SERVER_CONNECT", 0, "0x00", "HandleHeadcodeOnConnect", "AuthServer.cpp", "login", "connection", "LoginServer handshake"),
        ("HEADCODE_LOGIN_SERVER_ACCOUNT_LOGIN", 3, "0x03", "PlayerLoginResult", "AuthServer.cpp", "login", "account", "Login result"),
        ("HEADCODE_LOGIN_SERVER_ACCOUNT_SERVER_MOVE", 5, "0x05", "PlayerServerMoveResult", "AuthServer.cpp", "login", "account", "Server move result"),
        ("HEADCODE_LOGIN_SERVER_ACCOUNT_SERVER_AUTH", 6, "0x06", "PlayerServerMoveAuthResult", "AuthServer.cpp", "login", "account", "Server move auth"),
        ("HEADCODE_LOGIN_SERVER_ACCOUNT_KICK", 7, "0x07", "PlayerAccountKick", "AuthServer.cpp", "login", "account", "Kick account"),
        ("HEADCODE_LOGIN_SERVER_COMPLETE_BAN", 9, "0x09", "AccountCompleteBan", "AuthServer.cpp", "login", "account", "Complete ban"),
        ("HEADCODE_LOGIN_SERVER_ACCOUNT_OFFLINE_LOGIN", 14, "0x0E", "PlayerOfflineLoginResult", "AuthServer.cpp", "login", "account", "Offline login"),
    ]
    conn.executemany(
        "INSERT INTO packet_handlers (headcode_name, headcode_value, headcode_hex, handler_method, source_file, handler_type, category, description) VALUES (?,?,?,?,?,?,?,?)",
        login
    )

    # ConnectServer handlers
    connect = [
        ("HEADCODE_CONNECT_SERVER_CONNECT", 0, "0x00", "HandleHeadcodeOnConnect", "ConnectServer.cpp", "connect", "connection", "ConnectServer handshake"),
        ("HEADCODE_CONNECT_SERVER_FLAG", 2, "0x02", "HandleHeadcodeFlag", "ConnectServer.cpp", "connect", "connection", "Server flag update"),
        ("HEADCODE_CONNECT_SERVER_CHANNELS", 3, "0x03", "HandleHeadcodeChannel", "ConnectServer.cpp", "connect", "connection", "Channel list update"),
    ]
    conn.executemany(
        "INSERT INTO packet_handlers (headcode_name, headcode_value, headcode_hex, handler_method, source_file, handler_type, category, description) VALUES (?,?,?,?,?,?,?,?)",
        connect
    )


# ============================================================
# LIVE SCANNER — Scans actual source files
# ============================================================

def _rel_name(f, base_dir):
    """Return POSIX-style relative path from base_dir, e.g. 'editor/codeeditor.cpp'."""
    try:
        return str(f.relative_to(base_dir)).replace("\\", "/")
    except ValueError:
        return f.name


def scan_files(conn):
    """Scan source directories and register all source files."""
    for project, base_dir, extensions in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for ext in extensions:
            for f in base_dir.rglob(ext):
                rel = _rel_name(f, base_dir)
                category = categorize_file(rel, project)
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO files (path, project, category) VALUES (?,?,?)",
                        (rel, project, category)
                    )
                except sqlite3.IntegrityError:
                    pass


def categorize_file(name, project):
    """Auto-categorize a file by its name pattern using config rules."""
    for rule in CATEGORY_RULES:
        # Optional source_dir filter
        rule_dir = rule.get("source_dir")
        if rule_dir and rule_dir != project:
            continue
        match_type = rule["match"]
        pattern = rule["pattern"]
        if match_type == "prefix" and name.startswith(pattern):
            return rule["category"]
        if match_type == "suffix" and name.endswith(pattern):
            return rule["category"]
        if match_type == "contains" and pattern in name:
            return rule["category"]
        if match_type == "regex" and re.search(pattern, name):
            return rule["category"]
    # Default: 'common' for Common project, 'core' for everything else
    if project == "Common":
        return "common"
    return "core"


def populate_seed_data(conn, cfg_seed=None):
    """Insert seed data from config. Falls back to hardcoded data if no config."""
    seed = cfg_seed or SEED_DATA

    if "singletons" in seed and seed["singletons"]:
        items = seed["singletons"]
        if isinstance(items[0], dict):
            data = [(s["macro"], s["class_name"], s["header"],
                     s.get("project", "Game"), s.get("category"), s.get("description"))
                    for s in items]
        else:
            data = items
        conn.executemany(
            "INSERT INTO singletons (macro, class_name, header, project, category, description) VALUES (?,?,?,?,?,?)",
            data
        )

    if "player_files" in seed and seed["player_files"]:
        items = seed["player_files"]
        if isinstance(items[0], dict):
            data = [(p["file"], p["domain"], p.get("key_methods", ""), p.get("description", ""))
                    for p in items]
        else:
            data = items
        conn.executemany(
            "INSERT INTO player_files (file, domain, key_methods, description) VALUES (?,?,?,?)",
            data
        )

    if "events" in seed and seed["events"]:
        items = seed["events"]
        if isinstance(items[0], dict):
            data = [(e["name"], e.get("singleton_macro"), e.get("cpp_file"),
                     e.get("header_file"), e.get("def_file"), e.get("ai_file"),
                     e.get("description", ""))
                    for e in items]
        else:
            data = items
        conn.executemany(
            "INSERT INTO events (name, singleton_macro, cpp_file, header_file, def_file, ai_file, description) VALUES (?,?,?,?,?,?,?)",
            data
        )

    if "ai_handlers" in seed and seed["ai_handlers"]:
        items = seed["ai_handlers"]
        if isinstance(items[0], dict):
            data = [(a["file"], a["target"], a.get("description", "")) for a in items]
        else:
            data = items
        conn.executemany(
            "INSERT INTO ai_handlers (file, target, description) VALUES (?,?,?)",
            data
        )

    if "inventory_scripts" in seed and seed["inventory_scripts"]:
        items = seed["inventory_scripts"]
        if isinstance(items[0], dict):
            data = [(s["file"], s.get("description", "")) for s in items]
        else:
            data = items
        conn.executemany(
            "INSERT INTO inventory_scripts (file, description) VALUES (?,?)",
            data
        )

    if "constants" in seed and seed["constants"]:
        items = seed["constants"]
        if isinstance(items[0], dict):
            data = [(c["name"], c.get("value"), c["header"],
                     c.get("category"), c.get("description", ""))
                    for c in items]
        else:
            data = items
        conn.executemany(
            "INSERT INTO constants (name, value, header, category, description) VALUES (?,?,?,?,?)",
            data
        )

    if "config_files" in seed and seed["config_files"]:
        items = seed["config_files"]
        if isinstance(items[0], dict):
            data = [(c["path"], c.get("description", "")) for c in items]
        else:
            data = items
        conn.executemany(
            "INSERT INTO config_files (path, description) VALUES (?,?)",
            data
        )

    if "packet_handlers" in seed and seed["packet_handlers"]:
        items = seed["packet_handlers"]
        if isinstance(items[0], dict):
            data = [(h["headcode_name"], h.get("headcode_value"), h.get("headcode_hex"),
                     h.get("handler_method"), h["source_file"], h["handler_type"],
                     h.get("category"), h.get("description", ""))
                    for h in items]
        else:
            data = items
        conn.executemany(
            "INSERT INTO packet_handlers (headcode_name, headcode_value, headcode_hex, handler_method, source_file, handler_type, category, description) VALUES (?,?,?,?,?,?,?,?)",
            data
        )


RESEARCH_SYMBOL_RE = re.compile(
    r"\b(?:"
    r"HEADCODE_[A-Z0-9_]+|"
    r"sub_[0-9A-Fa-f]+|"
    r"(?:GC|CG)[A-Za-z0-9_]+|"
    r"[A-Z][A-Za-z0-9_]+::[A-Za-z0-9_]+|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:Recv|Send|Handler|Manager|System|Window|Protocol|Packet|Attack|Damage|Viewport|Automata|AI)"
    r"[A-Za-z0-9_]*"
    r")\b"
)
RESEARCH_PROTOCOL_RE = re.compile(r"\b(?:HEADCODE_[A-Z0-9_]+|HC\s*\d+|0x[0-9A-Fa-f]{2,4})\b")
RESEARCH_ASM_LABEL_RE = re.compile(r"^\s*(sub_[0-9A-Fa-f]+):")


def _trim_text(text, limit=240):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _iter_research_files():
    if not RESEARCH_DIR.exists():
        return []
    return sorted(
        [path for path in RESEARCH_DIR.rglob("*") if path.is_file()],
        key=lambda item: str(item).lower(),
    )


def _research_asset_type(path):
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".txt", ".log"}:
        return "text"
    if suffix == ".asm":
        return "asm"
    if suffix in {".exe", ".dll", ".bin"}:
        return "binary"
    return "other"


def _extract_research_symbols(text, limit=80):
    seen = set()
    symbols = []
    for match in RESEARCH_SYMBOL_RE.finditer(text or ""):
        symbol = match.group(0).strip()
        if len(symbol) < 4 or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if len(symbols) >= limit:
            break
    return symbols


def _extract_protocol_refs(text, limit=40):
    seen = set()
    refs = []
    for match in RESEARCH_PROTOCOL_RE.finditer(text or ""):
        ref = match.group(0).strip()
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
        if len(refs) >= limit:
            break
    return refs


def _scan_markdown_research_file(path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    title = None
    summary = None
    mentions = []
    seen_mentions = set()
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if title is None and stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            continue
        if summary is None and not stripped.startswith("#") and not set(stripped) <= {"-", "="}:
            summary = _trim_text(stripped, 260)
        for symbol in _extract_research_symbols(stripped, limit=12):
            key = ("symbol", symbol)
            if key not in seen_mentions:
                seen_mentions.add(key)
                mentions.append((symbol, "symbol", _trim_text(stripped, 220)))
        for ref in _extract_protocol_refs(stripped, limit=8):
            key = ("protocol", ref)
            if key not in seen_mentions:
                seen_mentions.add(key)
                mentions.append((ref, "protocol", _trim_text(stripped, 220)))

    if title is None:
        title = path.stem.replace("_", " ")
    symbols = [entry[0] for entry in mentions if entry[1] == "symbol"]
    protocols = [entry[0] for entry in mentions if entry[1] == "protocol"]
    return {
        "title": title,
        "summary": summary or f"Research asset: {path.name}",
        "symbol_refs": symbols[:40],
        "protocol_refs": protocols[:20],
        "notes": f"lines={len(lines)}",
        "mentions": mentions,
    }


def _scan_asm_research_file(path):
    mentions = []
    seen_labels = set()
    label_count = 0
    preview = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = RESEARCH_ASM_LABEL_RE.match(line)
                if not match:
                    continue
                label = match.group(1)
                label_count += 1
                if label not in seen_labels:
                    seen_labels.add(label)
                    mentions.append((label, "asm_label", label))
                    if len(preview) < 16:
                        preview.append(label)
    except Exception:
        return None

    summary = f"Client disassembly listing with {label_count} labeled subroutines"
    notes = f"preview={', '.join(preview[:8])}" if preview else "raw disassembly"
    return {
        "title": path.name,
        "summary": summary,
        "symbol_refs": preview,
        "protocol_refs": [],
        "notes": notes,
        "mentions": mentions,
    }


def scan_research_assets(conn):
    """Index ReversingResearch notes and lightweight symbol references."""
    for path in _iter_research_files():
        rel_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        asset_type = _research_asset_type(path)
        parsed = None
        if asset_type in {"markdown", "text"}:
            parsed = _scan_markdown_research_file(path)
        elif asset_type == "asm":
            parsed = _scan_asm_research_file(path)

        title = path.stem
        summary = None
        symbol_refs = []
        protocol_refs = []
        notes = None
        mentions = []
        if parsed:
            title = parsed["title"] or title
            summary = parsed["summary"]
            symbol_refs = parsed["symbol_refs"]
            protocol_refs = parsed["protocol_refs"]
            notes = parsed["notes"]
            mentions = parsed["mentions"]
        else:
            summary = f"Research asset: {path.name}"

        conn.execute(
            "INSERT INTO research_assets (path, file_name, asset_type, size_bytes, title, summary, symbol_refs, protocol_refs, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rel_path,
                path.name,
                asset_type,
                path.stat().st_size,
                title,
                summary,
                ", ".join(symbol_refs[:20]),
                ", ".join(protocol_refs[:12]),
                notes,
            ),
        )

        for symbol, mention_type, context in mentions:
            conn.execute(
                "INSERT INTO research_mentions (asset_path, symbol, mention_type, context) VALUES (?,?,?,?)",
                (rel_path, symbol, mention_type, context),
            )


def scan_includes(conn):
    """Scan import/include directives to build dependency graph."""
    for project, base_dir, exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        adapter = _project_adapter(project)
        import_re = adapter.import_pattern()
        if not import_re:
            continue
        for ext in exts:
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            m = import_re.search(line)
                            if m:
                                # Get first non-None group
                                inc = next((g for g in m.groups() if g), None)
                                if not inc:
                                    continue
                                inc = inc.replace("\\", "/").split("/")[-1]
                                try:
                                    conn.execute(
                                        "INSERT INTO dependencies (source_file, target_file, dep_type) VALUES (?,?,?)",
                                        (_rel_name(f, base_dir), inc, "include")
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass


def scan_singleton_usage(conn):
    """Scan for singleton macro usage patterns (sXxx->) in .cpp files."""
    singleton_macros = set()
    for row in conn.execute("SELECT macro FROM singletons"):
        singleton_macros.add(row[0])

    pattern = re.compile(r'\b(' + '|'.join(re.escape(m) for m in singleton_macros) + r')->')

    for _label, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for f in base_dir.rglob("*.cpp"):
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                found = set(pattern.findall(content))
                for macro in found:
                    # Find which header the singleton is in
                    row = conn.execute("SELECT header FROM singletons WHERE macro=?", (macro,)).fetchone()
                    if row:
                        try:
                            conn.execute(
                                "INSERT INTO dependencies (source_file, target_file, dep_type) VALUES (?,?,?)",
                                (_rel_name(f, base_dir), row[0], "singleton_use")
                            )
                        except Exception:
                            pass
            except Exception:
                pass


def scan_enums(conn):
    """Scan source files for enum declarations and extract values."""
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        adapter = _project_adapter(project)
        enum_re = adapter.enum_pattern()
        if not enum_re:
            continue
        header_exts = adapter.header_extensions or adapter.extensions
        scan_exts = header_exts if header_exts else _exts
        for ext in scan_exts:
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = fh.readlines()
                    i = 0
                    while i < len(lines):
                        m = enum_re.match(lines[i])
                        if m:
                            enum_name = m.group(1)
                            enum_line = i + 1
                            values = []
                            if adapter.uses_braces:
                                j = i + 1
                                brace_found = '{' in lines[i]
                                while not brace_found and j < min(i + 3, len(lines)):
                                    if '{' in lines[j]:
                                        brace_found = True
                                    j += 1
                                if not brace_found:
                                    i += 1
                                    continue
                                while j < len(lines):
                                    line = lines[j].strip()
                                    if '}' in line:
                                        break
                                    vm = re.match(r'(\w+)\s*(?:=\s*([^,/\n]+))?\s*[,]?', line)
                                    if vm and not line.startswith('//') and not line.startswith('#'):
                                        values.append(vm.group(1))
                                    j += 1
                            else:
                                # Python/indent-based: read indented values
                                j = i + 1
                                base_indent = len(lines[i]) - len(lines[i].lstrip())
                                while j < len(lines):
                                    ln = lines[j]
                                    stripped = ln.strip()
                                    if not stripped or stripped.startswith('#'):
                                        j += 1
                                        continue
                                    indent = len(ln) - len(ln.lstrip())
                                    if indent <= base_indent:
                                        break
                                    vm = re.match(r'(\w+)\s*=', stripped)
                                    if vm:
                                        values.append(vm.group(1))
                                    j += 1
                            preview = ", ".join(values[:8])
                            if len(values) > 8:
                                preview += f", ... ({len(values)} total)"
                            cat = categorize_file(_rel_name(f, base_dir), project)
                            try:
                                conn.execute(
                                    "INSERT INTO enums (name, file, line, value_count, values_preview, category) VALUES (?,?,?,?,?,?)",
                                    (enum_name, _rel_name(f, base_dir), enum_line, len(values), preview, cat)
                                )
                            except Exception:
                                pass
                        i += 1
                except Exception:
                    pass


def scan_structs(conn):
    """Scan source files for struct/dataclass/record declarations."""
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        adapter = _project_adapter(project)
        struct_re = adapter.struct_pattern()
        if not struct_re:
            continue
        scan_exts = adapter.header_extensions or adapter.extensions
        for ext in (scan_exts if scan_exts else _exts):
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = fh.readlines()
                    in_pack = False
                    i = 0
                    while i < len(lines):
                        line = lines[i].strip()
                        # Track C++ #pragma pack state
                        if adapter.name == "cpp" and '#pragma' in line and 'pack' in line:
                            if 'pack(1)' in line or 'pack(push' in line:
                                in_pack = True
                            elif 'pack()' in line or 'pack(pop' in line:
                                in_pack = False
                            i += 1
                            continue
                        m = struct_re.match(lines[i])
                        if m:
                            struct_name = m.group(1)
                            struct_line = i + 1
                            fields = []
                            if adapter.uses_braces:
                                j = i + 1
                                brace_found = '{' in lines[i]
                                while not brace_found and j < min(i + 3, len(lines)):
                                    if '{' in lines[j]:
                                        brace_found = True
                                    j += 1
                                if not brace_found:
                                    i += 1
                                    continue
                                brace_depth = 1
                                while j < len(lines) and brace_depth > 0:
                                    fl = lines[j].strip()
                                    brace_depth += fl.count('{') - fl.count('}')
                                    if brace_depth <= 0:
                                        break
                                    fm = re.match(r'((?:const\s+)?(?:unsigned\s+)?\w+(?:\s*\*)?)\s+(\w+)(?:\[.*\])?\s*;', fl)
                                    if fm and brace_depth == 1:
                                        fields.append(f"{fm.group(1)} {fm.group(2)}")
                                    j += 1
                            else:
                                # Indent-based (Python dataclass): read indented fields
                                j = i + 1
                                base_indent = len(lines[i]) - len(lines[i].lstrip())
                                while j < len(lines):
                                    ln = lines[j]
                                    stripped = ln.strip()
                                    if not stripped or stripped.startswith('#'):
                                        j += 1
                                        continue
                                    indent = len(ln) - len(ln.lstrip())
                                    if indent <= base_indent and stripped not in ('', 'pass'):
                                        break
                                    fm = re.match(r'(\w+)\s*(?::\s*([\w\[\], ]+))?', stripped)
                                    if fm and ':' in stripped:
                                        fields.append(f"{fm.group(2) or '?'} {fm.group(1)}")
                                    j += 1
                            preview = "; ".join(fields[:6])
                            if len(fields) > 6:
                                preview += f"; ... ({len(fields)} total)"
                            cat = "packet" if in_pack else "data"
                            try:
                                conn.execute(
                                    "INSERT INTO structs (name, file, line, field_count, is_packed, fields_preview, category) VALUES (?,?,?,?,?,?,?)",
                                    (struct_name, _rel_name(f, base_dir), struct_line, len(fields), 1 if in_pack else 0, preview, cat)
                                )
                            except Exception:
                                pass
                        i += 1
                except Exception:
                    pass


def scan_classes(conn):
    """Scan source files for class/interface/impl declarations and inheritance."""
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        adapter = _project_adapter(project)
        class_re = adapter.class_pattern()
        if not class_re:
            continue
        scan_exts = adapter.header_extensions or adapter.extensions
        for ext in (scan_exts if scan_exts else _exts):
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            m = class_re.match(line)
                            if m:
                                class_name = m.group(1)
                                parent = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                                # Try to find a companion implementation file
                                impl_ext = ".cpp" if adapter.name == "cpp" else f.suffix
                                impl_name = f.stem + impl_ext
                                impl_path = f.parent / impl_name
                                try:
                                    conn.execute(
                                        "INSERT INTO classes (name, header, cpp_file, parent_class, project) VALUES (?,?,?,?,?)",
                                        (class_name, _rel_name(f, base_dir), impl_name if impl_path.exists() else None, parent, project)
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass


def scan_methods(conn):
    """Scan source files for method/function declarations in classes."""
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        adapter = _project_adapter(project)
        method_re = adapter.method_pattern()
        class_re = adapter.class_pattern()
        if not method_re:
            continue
        skip_names = adapter.method_skip_names()
        scan_exts = adapter.header_extensions or adapter.extensions
        for ext in (scan_exts if scan_exts else _exts):
            for f in base_dir.rglob(ext):
                try:
                    current_class = None
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, 1):
                            # Track current class context
                            if class_re:
                                cm = class_re.match(line)
                                if cm:
                                    current_class = cm.group(1)
                                    continue
                            m = method_re.match(line)
                            if m:
                                # For Python: group(1) is method name; for C++: group(2)
                                method_name = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
                                if not method_name or method_name in skip_names:
                                    continue
                                try:
                                    conn.execute(
                                        "INSERT INTO methods (class_name, method_name, file, line_hint) VALUES (?,?,?,?)",
                                        (current_class, method_name, _rel_name(f, base_dir), lineno)
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass


def scan_todos(conn):
    """Scan source files for TODO, FIXME, HACK, BUG, XXX, NOTE comments."""
    for project, base_dir, exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        todo_re = _project_adapter(project).todo_pattern()
        for ext in exts:
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, 1):
                            m = todo_re.search(line)
                            if m:
                                todo_type = m.group(1).upper()
                                text = m.group(2).strip().rstrip('*/')
                                if len(text) < 3:
                                    continue
                                try:
                                    conn.execute(
                                        "INSERT INTO todos (file, line, todo_type, text, project) VALUES (?,?,?,?,?)",
                                        (_rel_name(f, base_dir), lineno, todo_type, text[:200], project)
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass


def scan_db_tables(conn):
    """Scan source files for SQL table references in queries."""
    # Match common SQL patterns: FROM table, INTO table, UPDATE table, JOIN table, TABLE table
    sql_re = re.compile(
        r'(?:FROM|INTO|UPDATE|JOIN|TABLE)\s+`?(\w+)`?',
        re.IGNORECASE
    )
    # Also match DirectDB and MuDatabase query strings
    query_type_re = re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b', re.IGNORECASE)

    seen = set()
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for ext in ("*.cpp", "*.h"):
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, 1):
                            # Only look at lines with SQL-like content
                            if not any(kw in line.upper() for kw in ('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'CREATE TABLE', 'ALTER TABLE')):
                                continue
                            tables = sql_re.findall(line)
                            qt_match = query_type_re.search(line)
                            query_type = qt_match.group(1).upper() if qt_match else None
                            for table_name in tables:
                                # Filter out common false positives
                                if table_name.upper() in ('SET', 'WHERE', 'AND', 'OR', 'NOT', 'NULL', 'VALUES', 'INTO', 'AS', 'ON', 'BY', 'IF', 'IN', 'IS'):
                                    continue
                                key = (table_name, _rel_name(f, base_dir), query_type)
                                if key in seen:
                                    continue
                                seen.add(key)
                                context = line.strip()[:150]
                                try:
                                    conn.execute(
                                        "INSERT INTO db_tables (table_name, source_file, line, query_type, context) VALUES (?,?,?,?,?)",
                                        (table_name, _rel_name(f, base_dir), lineno, query_type, context)
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass


def scan_prepared_statements(conn):
    """Scan MuDatabase.cpp (and similar) for PrepareStatement calls."""
    prep_re = re.compile(
        r'PrepareStatement\s*\(\s*(QUERY_\w+)\s*,\s*"([^"]+)"\s*,\s*(CONNECTION_\w+)',
        re.DOTALL
    )
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for ext in ("*.cpp", "*.h"):
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    for m in prep_re.finditer(content):
                        query_name = m.group(1)
                        sql_text = m.group(2).strip()
                        conn_type = m.group(3)
                        # Calculate line number
                        lineno = content[:m.start()].count('\n') + 1
                        try:
                            conn.execute(
                                "INSERT INTO prepared_statements (query_name, sql_text, connection_type, source_file, line) VALUES (?,?,?,?,?)",
                                (query_name, sql_text, conn_type, _rel_name(f, base_dir), lineno)
                            )
                        except Exception:
                            pass
                except Exception:
                    pass


def scan_config_keys(conn):
    """Scan source files for sConfig->GetXxx(\"key\") calls."""
    cfg_re = re.compile(
        r'sConfig->\s*(Get(?:Int|String|Bool|Float|IntDefault))\s*\(\s*"([^"]+)"'
    )
    seen = set()
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for ext in ("*.cpp", "*.h"):
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, 1):
                            for m in cfg_re.finditer(line):
                                getter = m.group(1)
                                key = m.group(2)
                                dedup = (key, _rel_name(f, base_dir), getter)
                                if dedup in seen:
                                    continue
                                seen.add(dedup)
                                context = line.strip()[:150]
                                try:
                                    conn.execute(
                                        "INSERT INTO config_keys (key_name, getter_type, source_file, line, context) VALUES (?,?,?,?,?)",
                                        (key, getter, _rel_name(f, base_dir), lineno, context)
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass


def scan_defines(conn):
    """Scan source files for constant/define declarations."""
    cpp_skip_patterns = {'_H', '_H_', 'INCLUDED', 'GUARD'}

    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        adapter = _project_adapter(project)
        define_re = adapter.define_pattern()
        if not define_re:
            continue
        scan_exts = adapter.header_extensions if adapter.header_extensions else adapter.extensions
        for ext in (scan_exts if scan_exts else _exts):
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, 1):
                            m = define_re.match(line)
                            if m:
                                name = m.group(1)
                                value = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
                                if not name:
                                    continue
                                # C++ specific: skip include guards and function-like macros
                                if adapter.name == "cpp":
                                    if any(name.endswith(s) for s in cpp_skip_patterns):
                                        continue
                                    if name.startswith('_') and name.endswith('_'):
                                        continue
                                    if '(' in name:
                                        continue
                                    if value.startswith('\\'):
                                        continue
                                # Auto-categorize
                                cat = "misc"
                                name_upper = name.upper()
                                if any(k in name_upper for k in ("MAX", "MIN", "SIZE", "COUNT", "LIMIT")):
                                    cat = "limits"
                                elif any(k in name_upper for k in ("WORLD", "MAP", "ZONE")):
                                    cat = "world"
                                elif "ITEM" in name_upper:
                                    cat = "item"
                                elif "SKILL" in name_upper:
                                    cat = "skill"
                                elif "BUFF" in name_upper:
                                    cat = "buff"
                                elif any(k in name_upper for k in ("PACKET", "HEADCODE", "OPCODE")):
                                    cat = "packet"
                                try:
                                    conn.execute(
                                        "INSERT INTO defines (name, value, file, line, category) VALUES (?,?,?,?,?)",
                                        (name, value[:100], _rel_name(f, base_dir), lineno, cat)
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass


def scan_file_lines(conn):
    """Count lines in each source file for complexity metrics."""
    for project, base_dir, exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for ext in exts:
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        count = sum(1 for _ in fh)
                    category = categorize_file(_rel_name(f, base_dir), project)
                    conn.execute(
                        "INSERT INTO file_lines (file, project, line_count, category) VALUES (?,?,?,?)",
                        (_rel_name(f, base_dir), project, count, category)
                    )
                except Exception:
                    pass


# Extensions that are meaningful for leak analysis per adapter language.
# Prevents scanning .rs/.js/.php/.md/.dart etc. with C++ 'new' regex.
_LEAK_SCAN_EXTS = {
    "cpp":        {".cpp", ".cc", ".c", ".h", ".hpp", ".hh", ".cu", ".cuh", ".mm", ".m", ".ipp", ".inl"},
    "python":     {".py"},
    "rust":       {".rs"},
    "typescript": {".ts", ".tsx", ".js", ".jsx"},
}

# Projects where leak tracking is not meaningful:
# - scripts/tools: Python tooling runs under GC
# - audit: exploit PoC tests use intentional raw allocations in controlled scopes
_LEAK_SKIP_PROJECTS = {"scripts", "tools", "audit"}

def scan_leak_risks(conn):
    """Analyze resource alloc/dealloc balance per file to find potential leaks."""
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        if project in _LEAK_SKIP_PROJECTS:
            continue
        adapter = _project_adapter(project)
        patterns = adapter.leak_patterns()
        if not patterns:
            continue
        alloc_re = patterns['alloc']
        dealloc_re = patterns['dealloc']
        smart1_re = patterns.get('smart_unique')
        smart2_re = patterns.get('smart_shared')
        allowed_suffixes = _LEAK_SCAN_EXTS.get(adapter.name, set())
        scan_exts = [e for e in _exts if e not in (adapter.header_extensions or [])]
        if not scan_exts:
            scan_exts = _exts
        for ext in scan_exts:
            for f in base_dir.rglob(ext):
                # Only scan files whose extension matches the adapter's language
                if allowed_suffixes and not any(str(f).endswith(s) for s in allowed_suffixes):
                    continue
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = fh.readlines()
                    content = "".join(lines)
                    alloc_count = len(alloc_re.findall(content))
                    dealloc_count = len(dealloc_re.findall(content))
                    smart1_count = len(smart1_re.findall(content)) if smart1_re else 0
                    smart2_count = len(smart2_re.findall(content)) if smart2_re else 0
                    if alloc_count == 0 and dealloc_count == 0:
                        continue
                    risk = alloc_count - dealloc_count - smart1_count - smart2_count
                    samples = []
                    for i, line in enumerate(lines):
                        if alloc_re.search(line) and len(samples) < 5:
                            samples.append(f"L{i+1}: {line.strip()[:80]}")
                    conn.execute(
                        "INSERT INTO leak_risks (file, project, new_count, delete_count, make_unique_count, make_shared_count, risk_score, sample_lines) VALUES (?,?,?,?,?,?,?,?)",
                        (_rel_name(f, base_dir), project, alloc_count, dealloc_count, smart1_count, smart2_count, risk, "\n".join(samples))
                    )
                except Exception:
                    pass


# Projects that are dev tooling — not the library itself.  Skip from
# null-risk scanning because their Python .get()/.find()/.search()
# calls inflate the metric without representing library quality.
_NULL_SKIP_PROJECTS = {"scripts", "tools"}

def scan_null_risks(conn):
    """Find nullable-returning calls used without null checks."""
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        if project in _NULL_SKIP_PROJECTS:
            continue
        adapter = _project_adapter(project)
        risky_calls = adapter.null_risk_calls()
        if not risky_calls:
            continue
        call_pattern = re.compile(
            r'(?:\w+\s*(?:->|\.)\s*)?(' + '|'.join(re.escape(c) for c in risky_calls) + r')\s*\([^)]*\)'
        )
        deref_op = '->' if adapter.name == "cpp" else '.'
        for ext in [e for e in _exts if e not in (adapter.header_extensions or [])]:
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = fh.readlines()
                    for i, line in enumerate(lines):
                        for m in call_pattern.finditer(line):
                            func_call = m.group(1)
                            # Python dict.get(key, default) with a default arg is always safe
                            if func_call == 'get' and adapter.name == 'python':
                                call_args = m.group(0)
                                if ',' in call_args:
                                    continue
                            assign_match = re.search(r'(\w+)\s*=\s*' + re.escape(m.group(0)), line)
                            ptr_var = assign_match.group(1) if assign_match else None
                            if ptr_var:
                                window = "".join(lines[i+1:i+4]) if i+1 < len(lines) else ""
                                null_re = adapter.null_check_pattern(ptr_var)
                                has_check = bool(null_re.search(window)) if null_re else False
                                if not has_check and 'if' not in line and '?' not in line:
                                    conn.execute(
                                        "INSERT INTO null_risks (file, line, function_call, pointer_var, risk_type, context) VALUES (?,?,?,?,?,?)",
                                        (_rel_name(f, base_dir), i+1, func_call, ptr_var, 'no_null_check', line.strip()[:150])
                                    )
                            else:
                                # Direct dereference on same line
                                if deref_op in line[m.end():]:
                                    conn.execute(
                                        "INSERT INTO null_risks (file, line, function_call, pointer_var, risk_type, context) VALUES (?,?,?,?,?,?)",
                                        (_rel_name(f, base_dir), i+1, func_call, None, 'unchecked_deref', line.strip()[:150])
                                    )
                except Exception:
                    pass


def scan_raw_pointers(conn):
    """Find raw pointer / unmanaged resource class members (RAII/ownership candidates)."""
    for _label, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        adapter = _project_adapter(_label)
        member_re = adapter.raw_pointer_pattern()
        class_re = adapter.class_pattern()
        if not member_re:
            continue
        scan_exts = adapter.header_extensions or adapter.extensions
        for ext in (scan_exts if scan_exts else _exts):
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = fh.readlines()
                    current_class = None
                    for i, line in enumerate(lines):
                        if class_re:
                            cm = class_re.match(line)
                            if cm:
                                current_class = cm.group(1)
                        m = member_re.match(line)
                        if m and current_class:
                            member_type = m.group(1).strip()
                            member_name = m.group(2).strip()
                            if member_type in ('char*', 'const char*', 'void*'):
                                continue
                            conn.execute(
                                "INSERT INTO raw_pointers (file, line, class_name, member_type, member_name) VALUES (?,?,?,?,?)",
                                (_rel_name(f, base_dir), i+1, current_class, member_type, member_name)
                            )
                except Exception:
                    pass


def scan_unsafe_casts(conn):
    """Find unsafe type casts (language-specific)."""
    safe_types = {'void', 'char', 'unsigned', 'const', 'BYTE', 'LPBYTE', 'LPSTR', 'LPCSTR', 'LPVOID'}
    for project, base_dir, exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        cast_re = _project_adapter(project).unsafe_cast_pattern()
        if not cast_re:
            continue
        for ext in exts:
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, 1):
                            stripped = line.strip()
                            if stripped.startswith('//') or stripped.startswith('/*'):
                                continue
                            for m in cast_re.finditer(line):
                                cast_type = m.group(1).strip().rstrip('*').strip()
                                if cast_type.lower() in {s.lower() for s in safe_types}:
                                    continue
                                expr = m.group(0)
                                conn.execute(
                                    "INSERT INTO unsafe_casts (file, line, cast_expr, context) VALUES (?,?,?,?)",
                                    (_rel_name(f, base_dir), lineno, expr[:80], stripped[:150])
                                )
                except Exception:
                    pass


def scan_crash_risks(conn):
    """Detect crash-prone patterns: null chains, division by zero, use-after-free.

    For non-C++ languages, runs adapter-provided crash risk patterns only.
    """
    # Run adapter-provided generic crash risk patterns for any language
    adapter_patterns = LANG_ADAPTER.crash_risk_patterns()
    if adapter_patterns and LANG_ADAPTER.name != "cpp":
        for project, base_dir, exts in SOURCE_DIRS:
            if not base_dir.exists():
                continue
            for ext in exts:
                for f in base_dir.rglob(ext):
                    try:
                        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                            lines = fh.readlines()
                        for i, line in enumerate(lines):
                            for pattern_re, risk_type, severity in adapter_patterns:
                                for m in pattern_re.finditer(line):
                                    conn.execute(
                                        "INSERT INTO crash_risks (file, line, risk_type, severity, expression, context) VALUES (?,?,?,?,?,?)",
                                        (_rel_name(f, base_dir), i+1, risk_type, severity, m.group(0)[:100], line.strip()[:200])
                                    )
                    except Exception:
                        pass
        return

    # C++ specific analysis below
    if LANG_ADAPTER.name != "cpp":
        return

    # --- Pattern 1: Unchecked chain dereferences ---
    # Matches: expr->Method()->AnotherMethod()  (triple+ chain without null guard)
    chain_re = re.compile(
        r'(\w+(?:->|\.)(?:Get\w+|To\w+)\s*\([^)]*\))\s*->\s*(\w+)'
    )
    # Specific dangerous chains: GetTarget()-> GetSummoner()-> GetWorld()->
    target_chain_re = re.compile(
        r'(?:->|\.)(?:GetTarget|GetSummoner|GetWorld|GetOwner|GetParty|GetGuild|GetSummoned|GetInterfaceState)\s*\(\)\s*->\s*(\w+)'
    )
    # Division patterns:  / varname  or  / expr  (excluding /= and // comments)
    div_re = re.compile(
        r'(?<!/)\s/\s+(?!/)(?!\*)(?!0[^.])([a-zA-Z_]\w*)'
    )
    # Use-after patterns: ->Method() after ->Remove() or SetTarget(nullptr) on same object
    remove_then_use_re = re.compile(
        r'->(?:Remove|SetTarget\s*\(\s*nullptr\s*\))'
    )

    cpp_impl_exts = ["*.cpp", "*.cc", "*.cxx"]
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for cpp_ext in cpp_impl_exts:
            for f in base_dir.rglob(cpp_ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = fh.readlines()

                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        # Skip comments and blank lines
                        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*') or not stripped:
                            continue

                        # --- Dangerous chain dereferences ---
                        for m in target_chain_re.finditer(line):
                            window = "".join(lines[max(0,i-5):i+1])
                            call_name = m.group(0).split('(')[0].split('.')[-1].split('>')[-1]
                            has_guard = False
                            guard_funcs = ['GetTarget', 'GetSummoner', 'GetWorld', 'GetOwner',
                                           'GetParty', 'GetGuild', 'GetSummoned', 'GetInterfaceState']
                            for gf in guard_funcs:
                                if gf in call_name:
                                    if re.search(rf'if\s*\(?\s*!?\s*\w+(?:->|\.)' + gf, window):
                                        has_guard = True
                                    if re.search(rf'{gf}\s*\(\)\s*&&', window) or re.search(rf'&&\s*\w+(?:->|\.)' + gf, window):
                                        has_guard = True
                                    if re.search(rf'=\s*\w+(?:->|\.)' + gf + r'\s*\([^)]*\)', "".join(lines[max(0,i-3):i])):
                                        prev_window = "".join(lines[max(0,i-3):i+1])
                                        if 'if' in prev_window and ('nullptr' in prev_window or '!' in prev_window or 'NULL' in prev_window):
                                            has_guard = True
                            if not has_guard:
                                expr = m.group(0).strip()
                                severity = 'critical' if any(c in call_name for c in ['GetTarget', 'GetSummoner', 'GetWorld']) else 'high'
                                conn.execute(
                                    "INSERT INTO crash_risks (file, line, risk_type, severity, expression, context) VALUES (?,?,?,?,?,?)",
                                    (_rel_name(f, base_dir), i+1, 'null_chain', severity, expr[:100], stripped[:200])
                                )

                        # --- Division by zero ---
                        if '/' in stripped and not stripped.startswith('#') and '//' not in stripped.split('/', 1)[0] + '/' and '/*' not in stripped:
                            no_strings = re.sub(r'"[^"]*"', '""', stripped)
                            no_strings = re.sub(r"'[^']*'", "''", no_strings)
                            no_strings = re.sub(r'//.*$', '', no_strings)
                            if '/' in no_strings:
                                for dm in re.finditer(r'\b(\w+)\s*/\s*(\w+)\b', no_strings):
                                    divisor = dm.group(2)
                                    if divisor in ('2', '3', '4', '5', '8', '10', '16', '32', '64', '100', '128', '255', '256',
                                                   '1000', '1024', '2048', '4096', '0x', 'sizeof', 'MAX', 'CLOCKS_PER_SEC',
                                                   'MINUTE', 'HOUR', 'IN_MILLISECONDS', 'MAX_SERVER_PER_GROUP'):
                                        continue
                                    if divisor.isdigit():
                                        continue
                                    # ALL_CAPS_SNAKE identifiers are compile-time constants/macros — cannot be zero at runtime
                                    if re.fullmatch(r'[A-Z][A-Z0-9_]+', divisor):
                                        continue
                                    # Constant shift/mask: division by 64U, 32U, etc. — never zero
                                    dividend = dm.group(1)
                                    if re.fullmatch(r'\d+[UuLl]*', divisor):
                                        continue
                                    # C++ keywords / casts used as divisor tokens — not variables
                                    if divisor in ('static_cast', 'reinterpret_cast', 'dynamic_cast', 'const_cast'):
                                        continue
                                    # Float scientific-notation literals (1e3, 1e6, 1e9, etc.) — never zero
                                    if re.fullmatch(r'\d+e\d+', divisor) or re.fullmatch(r'\d+e\d+', dividend):
                                        continue
                                    # _hi / _lo suffixed variables in multi-precision arithmetic
                                    # (e.g. Knuth long-division v_hi): these are normalized to be
                                    # non-zero by the algorithm that sets them.
                                    if re.fullmatch(r'\w+_hi', divisor):
                                        continue
                                    # Bench / test / audit files are never production code —
                                    # div_zero in them is always a false positive (ratio
                                    # calculations, loop counters, timing divisions).
                                    rel_name = _rel_name(f, base_dir)
                                    _is_bench = any(p in rel_name for p in ('bench', 'Bench', 'benchmark', 'perf'))
                                    _is_test_file = any(p in rel_name for p in ('test_', '/tests/', 'esp32_test',
                                                                                 'unified_audit', 'selftest',
                                                                                 'audit_ct'))
                                    if _is_bench or _is_test_file:
                                        continue
                                    # printf/fprintf timing output inside any function whose name
                                    # contains 'bench', 'perf', or 'timing' — always safe
                                    if ('printf' in stripped or 'cerr' in stripped) and re.search(
                                            r'(bench|perf|timing|profile|measure)', stripped, re.I):
                                        continue
                                    # Divisor names ending with _calls, _count, _iters, _ops,
                                    # _ns, _ms, _us inside diagnostic output — iteration counters.
                                    # Check window because printf/cerr may span multiple lines.
                                    if divisor.endswith(('_calls', '_count', '_iters', '_ops',
                                                         '_ns', '_ms', '_us', '_iterations', '_per_sig')):
                                        _diag_window = "".join(lines[max(0,i-3):i+1])
                                        if 'printf' in _diag_window or 'cerr' in _diag_window or 'cout' in _diag_window:
                                            continue
                                    # ANY printf/cerr/cout line with `total_X / Y_calls` pattern
                                    # is timing/profiling output — the divisor is always a
                                    # non-zero iteration count.
                                    if ('printf' in stripped or 'cerr' in stripped or 'cout' in stripped):
                                        if re.search(r'(total|sum|elapsed|time|cycles?|avg|skip)\w*\s*/\s*\w+', no_strings):
                                            continue
                                        # Single-letter divisors (n, i, k) in diagnostic output
                                        # are loop/batch counters — never zero in practice
                                        if re.fullmatch(r'[a-zA-Z]', divisor):
                                            continue
                                    full_match_end = dm.end()
                                    if full_match_end < len(no_strings) and no_strings[full_match_end] == '=':
                                        continue
                                    window = "".join(lines[max(0,i-5):i+1])
                                    if re.search(rf'{re.escape(divisor)}\s*>\s*0|{re.escape(divisor)}\s*!=\s*0|{re.escape(divisor)}\s*>=\s*1|if\s*\(\s*{re.escape(divisor)}\s*\)', window):
                                        continue
                                    if re.search(rf'{re.escape(divisor)}\s*[>!]=?\s*0\s*\?', stripped) or re.search(rf'{re.escape(divisor)}\s*\?', stripped):
                                        continue
                                    conn.execute(
                                        "INSERT INTO crash_risks (file, line, risk_type, severity, expression, context) VALUES (?,?,?,?,?,?)",
                                        (_rel_name(f, base_dir), i+1, 'div_zero', 'critical', f'/ {divisor}', stripped[:200])
                                    )

                        # --- Use after Remove/SetTarget(nullptr) ---
                        if '->Remove()' in stripped:
                            obj_match = re.search(r'(\w+)->Remove\(\)', stripped)
                            if obj_match:
                                obj_name = obj_match.group(1)
                                for j in range(i+1, min(i+6, len(lines))):
                                    next_line = lines[j].strip()
                                    if next_line.startswith('//') or not next_line:
                                        continue
                                    if f'{obj_name}->' in next_line and 'return' not in lines[j-1].strip():
                                        conn.execute(
                                            "INSERT INTO crash_risks (file, line, risk_type, severity, expression, context) VALUES (?,?,?,?,?,?)",
                                            (_rel_name(f, base_dir), j+1, 'use_after_free', 'critical', f'{obj_name}->... after Remove()', next_line[:200])
                                        )
                                        break
                                    if 'return' in next_line or 'break' in next_line:
                                        break

                except Exception:
                    pass


def scan_infinite_loop_risks(conn):
    """Detect potential infinite loop patterns.
    
    Risk types:
    - while_true_no_break: while(true/1)/for(;;) without break/return/goto in body
    - retry_no_limit: Retry/reconnect loop without max attempt counter
    - loop_var_unused: Loop condition variable never modified inside body
    - recursive_no_base: Simple recursive call without visible base case
    """
    # Patterns for unconditional loops
    infinite_loop_re = re.compile(r'^\s*(?:while\s*\(\s*(?:true|1|TRUE)\s*\)|for\s*\(\s*;\s*;\s*\))\s*$')
    # while(condition) — extract condition variable
    while_var_re = re.compile(r'^\s*while\s*\(\s*(!?\s*)(\w+)\s*\)\s*$')
    # Retry pattern keywords
    retry_re = re.compile(r'(?:retry|reconnect|reattempt|try_again)', re.IGNORECASE)

    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for ext in ("*.cpp", "*.h"):
            for filepath in base_dir.rglob(ext):
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = fh.readlines()
                    _scan_infinite_loops_in_file(conn, _rel_name(filepath, base_dir), lines)
                except Exception:
                    pass


def _scan_infinite_loops_in_file(conn, filename, lines):
    """Scan a single file for infinite loop risks."""
    infinite_loop_re = re.compile(r'^\s*(?:while\s*\(\s*(?:true|1|TRUE)\s*\)|for\s*\(\s*;\s*;\s*\))')
    while_var_re = re.compile(r'^\s*while\s*\(\s*!?\s*(\w+)\s*\)')
    retry_kw_re = re.compile(r'(?:retry|reconnect|reattempt|try.?again)', re.IGNORECASE)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip comments
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*') or not stripped:
            i += 1
            continue

        # --- Pattern 1: while(true)/for(;;) without break/return ---
        if infinite_loop_re.match(line):
            body_end = _find_brace_end(lines, i)
            if body_end is not None and body_end > i:
                body = ''.join(lines[i+1:body_end])
                body_no_comments = re.sub(r'//.*$', '', body, flags=re.MULTILINE)
                body_no_comments = re.sub(r'/\*.*?\*/', '', body_no_comments, flags=re.DOTALL)
                has_exit = bool(re.search(r'\b(?:break|return|goto|exit|throw|continue)\b', body_no_comments))
                if not has_exit:
                    # Check for callback/event-driven patterns (acceptable)
                    is_main_loop = bool(re.search(r'(?:Sleep|sleep|select|poll|accept|recv|wait|WaitFor|SleepEx|std::this_thread)', body_no_comments))
                    severity = 'medium' if is_main_loop else 'critical'
                    conn.execute(
                        "INSERT INTO infinite_loop_risks (file, line, risk_type, severity, expression, context) VALUES (?,?,?,?,?,?)",
                        (filename, i+1, 'while_true_no_break', severity, stripped[:100], stripped[:200])
                    )
            i = (body_end or i) + 1
            continue

        # --- Pattern 2: Retry loop without max attempts ---
        if retry_kw_re.search(stripped) and ('while' in stripped or 'for' in stripped):
            body_end = _find_brace_end(lines, i)
            if body_end is not None and body_end > i:
                body = ''.join(lines[i:body_end+1])
                has_max = bool(re.search(r'(?:max|MAX|limit|count|attempt|tries|retry_count|num_retries)\s*[<>!=]|\+\+\s*\w*(?:count|attempt|tries|retry)', body, re.IGNORECASE))
                if not has_max:
                    conn.execute(
                        "INSERT INTO infinite_loop_risks (file, line, risk_type, severity, expression, context) VALUES (?,?,?,?,?,?)",
                        (filename, i+1, 'retry_no_limit', 'high', stripped[:100], stripped[:200])
                    )
            i = (body_end or i) + 1
            continue

        # --- Pattern 3: while(var) where var is never modified in body ---
        m = while_var_re.match(line)
        if m and '{' not in stripped:
            var_name = m.group(1)
            # Skip common safe patterns: iterators, pointers, known flags, streams, atomics
            if var_name not in ('true', 'false', '1', '0', 'TRUE', 'FALSE', 'it', 'iter', 'result', 'row', 'node', 'current', 'next', 'p', 'ptr', 'pNode',
                                'ss', 'stream', 'iss', 'oss', 'in', 'out', 'file', 'fin', 'fout', 'input', 'output'):
                body_end = _find_brace_end(lines, i)
                if body_end is not None and body_end - i > 2 and body_end - i < 200:
                    body = ''.join(lines[i+1:body_end])
                    body_no_comments = re.sub(r'//.*$', '', body, flags=re.MULTILINE)
                    body_no_comments = re.sub(r'/\*.*?\*/', '', body_no_comments, flags=re.DOTALL)
                    # Check if variable is ever modified (assigned, incremented, decremented, or passed by ref)
                    var_modified = bool(re.search(
                        rf'\b{re.escape(var_name)}\b\s*(?:[+\-*/%&|^]?=|\+\+|\-\-)|'
                        rf'(?:\+\+|\-\-)\s*\b{re.escape(var_name)}\b|'
                        rf'>>\s*\b{re.escape(var_name)}\b|'
                        rf'&\s*\b{re.escape(var_name)}\b',
                        body_no_comments
                    ))
                    has_break = bool(re.search(r'\b(?:break|return|goto)\b', body_no_comments))
                    # Skip atomic/stop-flag vars (modified from other threads)
                    is_atomic = bool(re.search(rf'(?:atomic|_stop|_running|_shutdown|_terminate)', var_name, re.IGNORECASE))
                    if not var_modified and not has_break and not is_atomic:
                        conn.execute(
                            "INSERT INTO infinite_loop_risks (file, line, risk_type, severity, expression, context) VALUES (?,?,?,?,?,?)",
                            (filename, i+1, 'loop_var_unused', 'high', f'while({var_name}) — never modified', stripped[:200])
                        )

        i += 1


def _find_brace_end(lines, start_line):
    """Find the closing brace matching the first opening brace at or after start_line."""
    depth = 0
    found_open = False
    for i in range(start_line, min(start_line + 500, len(lines))):
        for ch in lines[i]:
            if ch == '{':
                depth += 1
                found_open = True
            elif ch == '}':
                depth -= 1
                if found_open and depth == 0:
                    return i
    return None


def scan_dead_methods(conn):
    """Find methods declared in .h files that are never referenced in any .cpp file."""
    # First collect all .cpp content into a single searchable index
    all_cpp_content = {}
    for _label, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for ext in ("*.cpp", "*.cc", "*.cxx", "*.mm", "*.m"):
            for f in base_dir.rglob(ext):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        all_cpp_content[_rel_name(f, base_dir)] = fh.read()
                except Exception:
                    pass
    combined_cpp = "\n".join(all_cpp_content.values())

    # Check each method from the methods table
    rows = conn.execute("SELECT class_name, method_name, file, line_hint FROM methods").fetchall()
    skip_methods = {'operator', 'instance', 'Instance', 'Get', 'Set', 'Is', 'Has',
                    'begin', 'end', 'size', 'empty', 'clear', 'push_back', 'erase',
                    'find', 'insert', 'remove', 'update', 'reset', 'init', 'load'}
    for row in rows:
        class_name, method_name, header, line_hint = row
        if method_name in skip_methods or len(method_name) < 4:
            continue
        # Only C/C++ headers produce meaningful dead-method signals.
        # Python (.py), Rust (.rs), PHP (.php), etc. don't follow the
        # header-declaration → impl-definition split.
        if not header.endswith(('.h', '.hpp', '.hxx', '.hh')):
            continue
        # Check if method_name appears anywhere in .cpp files
        ref_count = combined_cpp.count(method_name)
        if ref_count == 0:
            try:
                conn.execute(
                    "INSERT INTO dead_methods (class_name, method_name, header_file, header_line, ref_count) VALUES (?,?,?,?,?)",
                    (class_name, method_name, header, line_hint, 0)
                )
            except Exception:
                pass


def scan_duplicate_blocks(conn):
    """Find duplicate code blocks (5+ consecutive identical lines) across files."""
    BLOCK_SIZE = 5  # minimum consecutive lines to count as duplicate
    block_hashes = defaultdict(list)  # hash -> [(file, start_line, first_line_text)]

    # Constant/data line: hex arrays, numeric arrays, byte sequences
    _const_line_re = re.compile(
        r'^[\s,{}\[\]()|]*(?:(?:0x[0-9a-fA-F]+|[0-9]{2,})[UuLl]*[\s,;)}\]|]*)+\s*(?://.*)?$'
    )

    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for f in base_dir.rglob("*.cpp"):
            fname = f.name
            # Skip test and fuzz files — structural boilerplate duplication is
            # expected and not actionable.
            if fname.startswith(("test_", "fuzz_")):
                continue
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = [l.strip() for l in fh.readlines()]
                # Slide window of BLOCK_SIZE lines
                for i in range(len(lines) - BLOCK_SIZE + 1):
                    block = lines[i:i+BLOCK_SIZE]
                    # Skip blocks that are mostly empty/trivial
                    non_empty = [l for l in block if l and l != '{' and l != '}' and l != 'break;' and not l.startswith('//')]
                    if len(non_empty) < 3:
                        continue
                    # Skip blocks that are entirely #include / #pragma directives
                    if all(l.startswith('#include') or l.startswith('#pragma') or not l for l in block):
                        continue
                    # Skip blocks that are only closing braces, namespace endings, or boilerplate
                    if all(l in ('', '{', '}', '};', 'return;', 'break;', 'default:') or l.startswith('//') for l in block):
                        continue
                    # Skip blocks that are mainly constant/literal initializers
                    # (e.g. secp256k1 field prime 0xFFFFFFFEFFFFFC2F, byte arrays)
                    _const_count = sum(1 for l in non_empty if _const_line_re.match(l))
                    if _const_count >= len(non_empty) * 0.6:
                        continue
                    text = "\n".join(block)
                    h = hashlib.md5(text.encode()).hexdigest()
                    block_hashes[h].append((_rel_name(f, base_dir), i+1, block[0][:80]))
            except Exception:
                pass

    # Only keep hashes that appear in different files
    for h, locations in block_hashes.items():
        files_set = set(loc[0] for loc in locations)
        if len(files_set) < 2:
            continue
        # Insert cross-file pairs (limit to first 3 pairs per hash)
        seen_pairs = set()
        pairs = 0
        for i, (fa, la, pa) in enumerate(locations):
            for j, (fb, lb, pb) in enumerate(locations[i+1:], i+1):
                if fa == fb:
                    continue
                pair_key = tuple(sorted([fa, fb]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                try:
                    conn.execute(
                        "INSERT INTO duplicate_blocks (block_hash, file_a, line_a, file_b, line_b, line_count, preview) VALUES (?,?,?,?,?,?,?)",
                        (h, fa, la, fb, lb, BLOCK_SIZE, pa)
                    )
                except Exception:
                    pass
                pairs += 1
                if pairs >= 3:
                    break
            if pairs >= 3:
                break


def scan_file_summaries(conn):
    """Generate one-line summaries for important source files."""
    comment_re = re.compile(r'^\s*(?://|/\*)\s*(.+?)(?:\*/\s*)?$')
    class_re = re.compile(r'^\s*class\s+(\w+)')

    dirs = [(label, path) for label, path, _exts in SOURCE_DIRS]

    for project, base_dir in dirs:
        if not base_dir.exists():
            continue
        for ext in ("*.cpp", "*.h"):
            for filepath in base_dir.rglob(ext):
                try:
                    summary = _extract_file_summary(filepath)
                    if summary:
                        category = _categorize_file(_rel_name(filepath, base_dir))
                        conn.execute(
                            "INSERT OR IGNORE INTO file_summaries (file, project, summary, category) VALUES (?,?,?,?)",
                            (_rel_name(filepath, base_dir), project, summary, category)
                        )
                except Exception:
                    pass


def _extract_file_summary(filepath):
    """Extract a one-line summary from a source file."""
    comment_re = re.compile(r'^\s*(?://|/\*)\s*(.+?)(?:\*/\s*)?$')
    class_re = re.compile(r'^\s*class\s+(\w+)')
    struct_re = re.compile(r'^\s*struct\s+(\w+)')
    singleton_re = re.compile(r'SingletonInstance\((\w+)\)')

    classes = []
    singletons = []
    first_comment = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            if lineno > 50:
                break
            stripped = line.strip()
            # Skip blank lines and includes
            if not stripped or stripped.startswith('#include') or stripped.startswith('#pragma'):
                continue
            # Capture first meaningful comment
            if first_comment is None:
                m = comment_re.match(line)
                if m:
                    text = m.group(1).strip().rstrip('*/')
                    # Skip boilerplate
                    if len(text) > 10 and not text.startswith('Copyright') and not text.startswith('==='):
                        first_comment = text[:120]
            # Track classes
            m = class_re.match(line)
            if m:
                classes.append(m.group(1))
            m = struct_re.match(line)
            if m and not m.group(1).startswith('_'):
                classes.append(m.group(1))
            m = singleton_re.search(line)
            if m:
                singletons.append(m.group(1))

    # Build summary
    if singletons:
        return f"Singleton: {', '.join(singletons[:3])}" + (f" — {first_comment}" if first_comment else "")
    if first_comment:
        return first_comment
    if classes:
        return f"Defines: {', '.join(classes[:4])}"
    return None


def _categorize_file(filename):
    """Categorize a file based on its name."""
    name = filename.lower()
    if 'player' in name:
        return 'player'
    if 'monster' in name or 'ai_' in name:
        return 'monster'
    if 'event' in name or 'castle' in name or 'blood' in name or 'devil' in name:
        return 'event'
    if 'packet' in name or 'protocol' in name or 'handler' in name:
        return 'network'
    if 'database' in name or 'query' in name or 'mysql' in name or 'db' in name:
        return 'db'
    if 'item' in name or 'inventory' in name or 'shop' in name:
        return 'item'
    if 'guild' in name or 'party' in name or 'friend' in name:
        return 'social'
    if 'skill' in name or 'magic' in name or 'buff' in name:
        return 'skill'
    if 'world' in name or 'map' in name or 'viewport' in name:
        return 'world'
    if 'config' in name or 'setting' in name:
        return 'config'
    if 'log' in name or 'appender' in name:
        return 'logging'
    if 'quest' in name:
        return 'quest'
    return 'core'


SEMANTIC_TAG_RULES = [
    ("security", ("auth", "login", "secure", "password", "warehouse", "otp", "pin", "secret", "seckey", "private key", "nonce")),
    ("crypto", ("secp256k1", "ecdsa", "schnorr", "bip340", "bip-340", "taproot", "xonly", "x-only", "ecdh", "scalar", "field", "point")),
    ("constant-time", ("constant-time", "constant time", "ct::", "ct_", "ct-verif", "dudect", "valgrind-ct", "side-channel", "side channel")),
    ("caas", ("caas", "continuous audit", "audit bundle", "audit_gate", "security_autonomy", "external_audit_bundle", "claim", "evidence")),
    ("audit", ("audit", "assurance", "exploit", "poc", "wycheproof", "fuzz", "traceability", "autonomy", "evidence")),
    ("bitcoin-core", ("bitcoin core", "bitcoin-core", "bitcoin_core", "libsecp", "libsecp256k1", "secp256k1_shim", "shim", "consensus")),
    ("compat-shim", ("compat", "shim", "libsecp256k1_shim", "secp256k1_context", "secp256k1_ecdsa", "secp256k1_schnorr")),
    ("strict-parser", ("parse_strict", "strict parse", "strict parsing", "parse_compact_strict", "parse_bytes_strict", "der parser", "canonical")),
    ("auth", ("auth", "login", "account", "session", "token")),
    ("warehouse", ("warehouse", "vault")),
    ("movement", ("move", "path", "teleport", "dir", "viewport", "position")),
    ("world-flag", ("world", "map", "terrain", "summon", "partymove", "party_move")),
    ("party", ("party", "member", "leader", "matching", "recruit")),
    ("boss-tab", ("neweventwindow", "bossfight", "ctrl+t", "ctrlt", "boss tab")),
    ("event", ("event", "castle", "blood", "devil", "kanturu", "raklion", "medusa", "kundun", "evomon")),
    ("ai", ("monsterai", "monstermovement", "ai_", "automata", "monster ai")),
    ("network-packet", ("packet", "protocol", "headcode", "serverlink", "sendpacket", "recv", "p2p", "network")),
    ("build-integration", ("cmake", "fetchcontent", "submodule", "build", "target_link_libraries", "add_subdirectory", "reproducible")),
    ("mix", ("mix", "chaos", "goblin")),
    ("inventory", ("inventory", "itembag", "item", "warehouse")),
    ("combat", ("attack", "damage", "skill", "combat", "threat")),
    ("db", ("query", "database", "mysql", "sql", "db")),
    ("quest", ("quest", "objective", "mission")),
    ("npc", ("npc", "guard", "merchant", "shop")),
]

HOT_PATH_TAGS = {"movement", "combat", "ai", "network-packet", "world-flag", "boss-tab", "crypto"}
OPTIMIZATION_TAGS = {"movement", "combat", "ai", "db", "network-packet", "world-flag", "crypto", "build-integration"}
SECURITY_TAGS = {"security", "auth", "warehouse", "crypto", "constant-time", "strict-parser"}
FRAGILE_TAGS = {"security", "auth", "warehouse", "boss-tab", "world-flag", "event", "mix", "bitcoin-core", "compat-shim", "strict-parser", "build-integration", "caas", "audit"}
NETWORK_TAGS = {"network-packet", "boss-tab", "bitcoin-core"}


def _collect_semantic_tags(*parts):
    text = " ".join(str(part or "") for part in parts).lower()
    tags = set()
    for tag, keywords in SEMANTIC_TAG_RULES:
        if any(keyword in text for keyword in keywords):
            tags.add(tag)
    return tags


def _emit_semantic_tag(conn, seen, entity_type, entity_name, file, line, tag, confidence, evidence):
    key = (entity_type, entity_name, file, line or 0, tag)
    if key in seen:
        return
    seen.add(key)
    conn.execute(
        "INSERT INTO semantic_tags (entity_type, entity_name, file, line, tag, confidence, evidence) VALUES (?,?,?,?,?,?,?)",
        (entity_type, entity_name, file, line, tag, confidence, evidence[:240] if evidence else None)
    )


def _project_relative_path(project, file_name):
    if project == "Game":
        return f"Game/{file_name}"
    if project == "Common":
        return f"Common/{file_name}"
    if project == "Server_Link":
        return f"Server_Link/{file_name}"
    if project == "LoginServer":
        return f"LoginServer/{file_name}"
    if project == "ConnectServer":
        return f"ConnectServer/{file_name}"
    return file_name


def _project_base_dir(project):
    for label, path, _exts in SOURCE_DIRS:
        if label == project:
            return path
    # Fallback to legacy paths
    if project == "Game":
        return GAME_DIR
    if project == "Common":
        return COMMON_DIR
    fallback = REPO_ROOT / project
    if fallback.exists():
        return fallback
    return None


def _read_project_file_lines(project, file_name, cache):
    key = (project, file_name)
    if key in cache:
        return cache[key]

    base_dir = _project_base_dir(project)
    if not base_dir:
        cache[key] = None
        return None

    path = base_dir / file_name
    if not path.exists():
        cache[key] = None
        return None

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            cache[key] = handle.readlines()
    except Exception:
        cache[key] = None
    return cache[key]


def _strip_cpp_comments_and_strings(text):
    result = []
    i = 0
    in_block_comment = False
    in_line_comment = False
    in_string = False
    in_char = False

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                result.append("\n")
            i += 1
            continue

        if in_string:
            if ch == "\\" and nxt:
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if in_char:
            if ch == "\\" and nxt:
                i += 2
                continue
            if ch == "'":
                in_char = False
            i += 1
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "'":
            in_char = True
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _function_label(class_name, function_name):
    return f"{class_name}::{function_name}" if class_name else function_name


def _extract_function_body_text(lines, start_line, end_line):
    snippet = "".join(lines[start_line - 1:end_line])
    if not snippet:
        return ""

    open_brace = snippet.find("{")
    close_brace = snippet.rfind("}")
    if open_brace >= 0 and close_brace > open_brace:
        snippet = snippet[open_brace + 1:close_brace]
    elif open_brace >= 0:
        snippet = snippet[open_brace + 1:]
    return snippet


CALL_SKIP_NAMES = frozenset({
    "if", "else", "for", "while", "switch", "catch", "return", "sizeof", "alignof", "decltype",
    "static_cast", "dynamic_cast", "reinterpret_cast", "const_cast", "ASSERT", "CHECK", "TEST_CASE",
    "min", "max", "defined", "new", "delete", "noexcept", "throw"
})


def _get_call_skip_names():
    """Return the union of the global C++ skip names and the adapter-specific skip names."""
    names = set(CALL_SKIP_NAMES)
    for adapter in SOURCE_DIR_ADAPTERS.values():
        names.update(adapter.call_skip_names())
    return frozenset(names)


def _resolve_call_target(caller_row, scope_name, callee_name, by_name):
    candidates = by_name.get(callee_name, [])
    if not candidates:
        return None, 0

    if scope_name:
        scoped = [row for row in candidates if (row["class_name"] or "") == scope_name]
        if len(scoped) == 1:
            return scoped[0], 95

    same_class = [
        row for row in candidates
        if (row["class_name"] or "") == (caller_row["class_name"] or "") and caller_row["class_name"]
    ]
    if len(same_class) == 1:
        return same_class[0], 90

    same_file = [row for row in candidates if row["file"] == caller_row["file"]]
    if len(same_file) == 1:
        return same_file[0], 85

    same_project = [row for row in candidates if row["project"] == caller_row["project"]]
    if len(same_project) == 1:
        return same_project[0], 72

    if len(candidates) == 1:
        return candidates[0], 80

    return None, 0


def scan_semantic_tags(conn):
    """Derive semantic tags from filenames, symbols, handlers, events, and summaries."""
    seen = set()

    file_rows = conn.execute(
        "SELECT f.path, f.category, COALESCE(fs.summary, '') AS summary "
        "FROM files f LEFT JOIN file_summaries fs ON f.path = fs.file"
    ).fetchall()
    for row in file_rows:
        file_name = row["path"]
        tags = _collect_semantic_tags(file_name, row["category"], row["summary"])
        for tag in tags:
            _emit_semantic_tag(
                conn, seen, "file", file_name, file_name, None, tag, 70,
                f"file/category/summary matched: {file_name}"
            )

    class_rows = conn.execute("SELECT name, header, cpp_file, parent_class, category, description FROM classes").fetchall()
    for row in class_rows:
        file_name = row["cpp_file"] or row["header"] or ""
        label = row["name"]
        tags = _collect_semantic_tags(label, row["parent_class"], row["category"], row["description"], file_name)
        for tag in tags:
            _emit_semantic_tag(
                conn, seen, "class", label, file_name or label, None, tag, 82,
                f"class metadata matched: {label}"
            )

    method_rows = conn.execute("SELECT class_name, method_name, file, line_hint, category, description FROM methods").fetchall()
    for row in method_rows:
        label = f"{row['class_name']}::{row['method_name']}"
        tags = _collect_semantic_tags(label, row["category"], row["description"], row["file"])
        for tag in tags:
            _emit_semantic_tag(
                conn, seen, "method", label, row["file"], row["line_hint"], tag, 88,
                f"method metadata matched: {label}"
            )

    function_rows = conn.execute("SELECT file, function_name, class_name, start_line, signature, project FROM function_index").fetchall()
    for row in function_rows:
        label = f"{row['class_name']}::{row['function_name']}" if row["class_name"] else row["function_name"]
        tags = _collect_semantic_tags(label, row["signature"], row["file"], row["project"])
        for tag in tags:
            _emit_semantic_tag(
                conn, seen, "function", label, row["file"], row["start_line"], tag, 92,
                f"function signature matched: {row['signature'] or label}"
            )

    handler_rows = conn.execute(
        "SELECT headcode_name, handler_method, source_file, handler_type, category, description FROM packet_handlers"
    ).fetchall()
    for row in handler_rows:
        label = f"{row['headcode_name']} {row['handler_method']}"
        tags = _collect_semantic_tags(label, row["handler_type"], row["category"], row["description"], row["source_file"])
        tags.add("network-packet")
        for tag in tags:
            _emit_semantic_tag(
                conn, seen, "handler", label, row["source_file"], None, tag, 95,
                f"packet handler matched: {label}"
            )

    event_rows = conn.execute("SELECT name, singleton_macro, cpp_file, header_file, def_file, ai_file, description FROM events").fetchall()
    for row in event_rows:
        label = row["name"]
        tags = _collect_semantic_tags(
            row["name"], row["singleton_macro"], row["cpp_file"], row["header_file"], row["def_file"], row["ai_file"], row["description"]
        )
        tags.add("event")
        for tag in tags:
            _emit_semantic_tag(
                conn, seen, "event", label, row["cpp_file"] or row["header_file"] or label, None, tag, 96,
                f"event metadata matched: {label}"
            )

    config_rows = conn.execute("SELECT path, description FROM config_files").fetchall()
    for row in config_rows:
        tags = _collect_semantic_tags(row["path"], row["description"], "config")
        for tag in tags:
            _emit_semantic_tag(
                conn, seen, "config", row["path"], row["path"], None, tag, 72,
                f"config path matched: {row['path']}"
            )


def scan_semantic_profiles(conn):
    """Aggregate semantic tags into coarse risk/gain profiles for reasoning queries."""
    rows = conn.execute(
        "SELECT entity_type, entity_name, file, COALESCE(line, 0) AS line, tag, confidence, COALESCE(evidence, '') AS evidence "
        "FROM semantic_tags ORDER BY entity_type, entity_name, file, line"
    ).fetchall()

    grouped = {}
    for row in rows:
        key = (row["entity_type"], row["entity_name"], row["file"], row["line"])
        grouped.setdefault(key, {"tags": set(), "evidence": []})
        grouped[key]["tags"].add(row["tag"])
        if row["evidence"]:
            grouped[key]["evidence"].append(row["evidence"])

    for (entity_type, entity_name, file_name, line), payload in grouped.items():
        tags = payload["tags"]
        evidence = "; ".join(payload["evidence"][:4])
        lower_name = entity_name.lower()
        lower_file = file_name.lower()

        test_surface = 1 if ("test" in lower_name or "test" in lower_file) else 0
        security_sensitive = 1 if tags & SECURITY_TAGS else 0
        hot_path_candidate = 1 if tags & HOT_PATH_TAGS else 0
        optimization_candidate = 1 if tags & OPTIMIZATION_TAGS else 0
        network_surface = 1 if tags & NETWORK_TAGS else 0
        fragile_surface = 1 if tags & FRAGILE_TAGS else 0

        if entity_type in {"function", "method"} and any(token in lower_name for token in ("update", "process", "attack", "move", "send")):
            hot_path_candidate = 1
            optimization_candidate = 1
        if entity_type == "handler":
            network_surface = 1
            fragile_surface = 1
        if entity_type == "event":
            fragile_surface = 1

        risk_score = (
            security_sensitive * 5 +
            fragile_surface * 3 +
            network_surface * 2 +
            (1 - test_surface) * 1
        )
        gain_score = (
            hot_path_candidate * 4 +
            optimization_candidate * 3 +
            network_surface * 2 +
            (1 if entity_type in {"function", "method"} else 0)
        )

        conn.execute(
            "INSERT INTO semantic_profiles (entity_type, entity_name, file, line, security_sensitive, hot_path_candidate, "
            "optimization_candidate, network_surface, fragile_surface, test_surface, risk_score, gain_score, evidence) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                entity_type, entity_name, file_name, line or None,
                security_sensitive, hot_path_candidate, optimization_candidate,
                network_surface, fragile_surface, test_surface,
                risk_score, gain_score, evidence[:240] if evidence else None
            )
        )


def scan_audit_coverage(conn):
    """Aggregate test/coverage/risk signals per file."""
    file_rows = conn.execute(
        "SELECT fl.file, fl.project FROM file_lines fl GROUP BY fl.file, fl.project ORDER BY fl.file"
    ).fetchall()

    for row in file_rows:
        file_name = row["file"]
        project = row["project"]
        unit_test_refs = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE target = ? AND edge_type = 'tests'",
            (file_name,)
        ).fetchone()[0]
        has_summary = 1 if conn.execute("SELECT 1 FROM file_summaries WHERE file = ? LIMIT 1", (file_name,)).fetchone() else 0
        has_function_index = 1 if conn.execute("SELECT 1 FROM function_index WHERE file = ? LIMIT 1", (file_name,)).fetchone() else 0
        todo_count = conn.execute("SELECT COUNT(*) FROM todos WHERE file = ?", (file_name,)).fetchone()[0]
        crash_risk_count = conn.execute("SELECT COUNT(*) FROM crash_risks WHERE file = ?", (file_name,)).fetchone()[0]
        null_risk_count = conn.execute("SELECT COUNT(*) FROM null_risks WHERE file = ?", (file_name,)).fetchone()[0]
        duplicate_pair_count = conn.execute(
            "SELECT COUNT(*) FROM duplicate_blocks WHERE file_a = ? OR file_b = ?",
            (file_name, file_name)
        ).fetchone()[0]
        dead_method_count = conn.execute("SELECT COUNT(*) FROM dead_methods WHERE header_file = ?", (file_name,)).fetchone()[0]
        leak_risk_row = conn.execute("SELECT risk_score FROM leak_risks WHERE file = ? LIMIT 1", (file_name,)).fetchone()
        leak_risk_score = leak_risk_row[0] if leak_risk_row else 0
        semantic_row = conn.execute(
            "SELECT COALESCE(MAX(risk_score), 0), COALESCE(MAX(gain_score), 0) FROM semantic_profiles WHERE file = ?",
            (file_name,)
        ).fetchone()
        semantic_risk_max, semantic_gain_max = semantic_row[0], semantic_row[1]

        coverage_score = 20
        coverage_score += min(unit_test_refs * 20, 40)
        coverage_score += 15 if has_summary else 0
        coverage_score += 15 if has_function_index else 0
        coverage_score -= min(todo_count * 2, 10)
        coverage_score -= min(crash_risk_count, 10)
        coverage_score -= min(null_risk_count // 5, 10)
        coverage_score -= min(duplicate_pair_count // 10, 10)
        coverage_score = max(0, min(100, coverage_score))

        notes = []
        if unit_test_refs == 0:
            notes.append("no unit test edges")
        if crash_risk_count > 0:
            notes.append(f"{crash_risk_count} crash risks")
        if null_risk_count > 0:
            notes.append(f"{null_risk_count} null risks")
        if duplicate_pair_count > 0:
            notes.append(f"{duplicate_pair_count} duplicate pairs")

        conn.execute(
            "INSERT INTO audit_coverage (file, project, unit_test_refs, has_summary, has_function_index, todo_count, "
            "crash_risk_count, null_risk_count, duplicate_pair_count, dead_method_count, leak_risk_score, "
            "semantic_risk_max, semantic_gain_max, coverage_score, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                file_name, project, unit_test_refs, has_summary, has_function_index, todo_count,
                crash_risk_count, null_risk_count, duplicate_pair_count, dead_method_count, leak_risk_score,
                semantic_risk_max, semantic_gain_max, coverage_score, "; ".join(notes)[:240]
            )
        )


def scan_history_metrics(conn):
    """Aggregate per-file git history/churn signals."""
    try:
        result = subprocess.run(
            [
                "git", "log", "--date=short", "--pretty=format:__COMMIT__|%ad|%an|%s",
                "--name-only", "--no-renames", "--", "Game", "Common", "Server_Link", "LoginServer", "ConnectServer"
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )
    except Exception:
        return

    cutoff = date.today() - timedelta(days=90)
    bugfix_keywords = ("fix", "bug", "crash", "regress", "issue", "hotfix")
    perf_keywords = ("perf", "optim", "speed", "fast", "hot path", "latency", "cache")
    audit_keywords = ("audit", "test", "fuzz", "coverage", "validate", "harden", "security")

    history = {}
    commit_date = None
    author = ""
    subject = ""

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("__COMMIT__|"):
            parts = line.split("|", 3)
            if len(parts) == 4:
                commit_date = datetime.strptime(parts[1], "%Y-%m-%d").date()
                author = parts[2]
                subject = parts[3].lower()
            continue

        file_name = os.path.basename(line)
        if not file_name:
            continue

        entry = history.setdefault(file_name, {
            "commit_count": 0,
            "recent_commit_count": 0,
            "authors": set(),
            "bugfix_commits": 0,
            "perf_commits": 0,
            "audit_commits": 0,
            "last_commit_date": None,
        })
        entry["commit_count"] += 1
        if commit_date and commit_date >= cutoff:
            entry["recent_commit_count"] += 1
        if author:
            entry["authors"].add(author)
        if any(keyword in subject for keyword in bugfix_keywords):
            entry["bugfix_commits"] += 1
        if any(keyword in subject for keyword in perf_keywords):
            entry["perf_commits"] += 1
        if any(keyword in subject for keyword in audit_keywords):
            entry["audit_commits"] += 1
        if commit_date and (entry["last_commit_date"] is None or commit_date > entry["last_commit_date"]):
            entry["last_commit_date"] = commit_date

    for file_name, entry in history.items():
        churn_score = (
            entry["commit_count"] +
            entry["recent_commit_count"] * 2 +
            entry["bugfix_commits"] * 2 +
            entry["perf_commits"] +
            len(entry["authors"])
        )
        conn.execute(
            "INSERT INTO history_metrics (file, commit_count, recent_commit_count, unique_authors, bugfix_commits, "
            "perf_commits, audit_commits, last_commit_date, churn_score) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                file_name,
                entry["commit_count"],
                entry["recent_commit_count"],
                len(entry["authors"]),
                entry["bugfix_commits"],
                entry["perf_commits"],
                entry["audit_commits"],
                entry["last_commit_date"].isoformat() if entry["last_commit_date"] else None,
                churn_score,
            )
        )


def scan_review_queue(conn):
    """Build actionable queues for review, hardening, and optimization."""
    rows = conn.execute(
        "SELECT a.file, a.project, a.coverage_score, a.unit_test_refs, a.crash_risk_count, a.null_risk_count, "
        "a.semantic_risk_max, a.semantic_gain_max, a.notes, "
        "COALESCE(h.churn_score, 0) AS churn_score, COALESCE(h.recent_commit_count, 0) AS recent_commit_count, "
        "COALESCE(h.bugfix_commits, 0) AS bugfix_commits, "
        "COALESCE(o.bus_factor_risk, 0) AS bus_factor_risk, COALESCE(o.primary_author_share, 0) AS primary_author_share, "
        "COALESCE(tf.tested_functions, 0) AS tested_functions "
        "FROM audit_coverage a "
        "LEFT JOIN history_metrics h ON a.file = h.file "
        "LEFT JOIN ownership_metrics o ON a.file = o.file "
        "LEFT JOIN (SELECT target_file, COUNT(DISTINCT function_name || '|' || COALESCE(class_name,'')) AS tested_functions "
        "           FROM test_function_map GROUP BY target_file) tf ON a.file = tf.target_file"
    ).fetchall()

    # File categories that are not actionable for testing/hardening queues:
    # tests, benchmarks, docs, build artifacts, scripts, fuzz targets, binding language dirs.
    _non_actionable_re = re.compile(
        r'(?:^test_|^tests?/|/tests?/|^bench[/_]|/bench[/_]|benchmark|\.md$|\.json$|\.yml$|\.yaml$|'
        r'\.build/|/examples?/|^scripts?/|_test\.cpp$|_test\.c$|selftest|\.txt$|\.cmake$|'
        r'unified_audit|audit_gate|audit_gap|export_assurance|preflight|\.py$|'
        r'esp32_test/build|source_graph_kit/|^audit_|'
        r'^fuzz|/fuzz[/_]|stm32_test/|^dart/|^go/|^kotlin/|^swift/|^java/|'
        r'^nodejs/|^php/|^ruby/|^react.native/|^python/|^ufsecp/|^ultrafast|'
        r'app/metal_test|_audit_runner|gpu_bench_)',
        re.IGNORECASE,
    )
    # Narrower filter for audit_gap: exclude only non-library tool/artifact files
    # (scripts, CI workflows, build outputs) — NOT security-critical binding dirs.
    _tool_artifact_re = re.compile(
        r'(?:\.py$|\.yml$|\.yaml$|\.build/|source_graph_kit/|^audit_|'
        r'unified_audit|audit_gate|audit_gap|export_assurance|preflight|'
        r'esp32_test/build|stm32_test/|^fuzz|/fuzz[/_]|\.derived/)',
        re.IGNORECASE,
    )

    for row in rows:
        file_name = row["file"]
        coverage_score = row["coverage_score"]
        semantic_risk = row["semantic_risk_max"]
        semantic_gain = row["semantic_gain_max"]
        churn_score = row["churn_score"]
        recent_commit_count = row["recent_commit_count"]
        crash_risk_count = row["crash_risk_count"]
        null_risk_count = row["null_risk_count"]
        bus_factor_risk = row["bus_factor_risk"]
        tested_functions = row["tested_functions"]

        # Skip non-actionable files for gain/testing queues
        _is_non_actionable = bool(_non_actionable_re.search(file_name))
        # Narrower check: skip only tool/artifact/CI files from audit_gap
        _is_tool_artifact = bool(_tool_artifact_re.search(file_name))

        if not _is_non_actionable and semantic_gain >= 7 and semantic_risk <= 3 and coverage_score >= 45:
            priority = semantic_gain * 5 + coverage_score - semantic_risk * 3
            conn.execute(
                "INSERT INTO review_queue (file, queue_type, priority_score, rationale) VALUES (?,?,?,?)",
                (file_name, "high_gain_low_risk", priority,
                 f"gain={semantic_gain} risk={semantic_risk} coverage={coverage_score}")
            )

        if not _is_tool_artifact and semantic_risk >= 5 and coverage_score < 45:
            priority = semantic_risk * 8 + max(0, 50 - coverage_score) + crash_risk_count * 3 + null_risk_count // 2
            conn.execute(
                "INSERT INTO review_queue (file, queue_type, priority_score, rationale) VALUES (?,?,?,?)",
                (file_name, "audit_gap", priority,
                 f"risk={semantic_risk} coverage={coverage_score} crash={crash_risk_count} null={null_risk_count}")
            )

        if recent_commit_count >= 2 and coverage_score < 55:
            priority = churn_score + max(0, 60 - coverage_score) + row["bugfix_commits"] * 3
            conn.execute(
                "INSERT INTO review_queue (file, queue_type, priority_score, rationale) VALUES (?,?,?,?)",
                (file_name, "churn_watch", priority,
                 f"recent={recent_commit_count} churn={churn_score} coverage={coverage_score}")
            )

        if crash_risk_count > 0 or null_risk_count >= 10:
            priority = crash_risk_count * 6 + null_risk_count // 2 + semantic_risk * 4
            conn.execute(
                "INSERT INTO review_queue (file, queue_type, priority_score, rationale) VALUES (?,?,?,?)",
                (file_name, "stability_hotspot", priority,
                 f"crash={crash_risk_count} null={null_risk_count} risk={semantic_risk}")
            )

        if bus_factor_risk >= 5 and semantic_gain >= 7:
            priority = bus_factor_risk * 8 + semantic_gain * 4 + max(0, 50 - coverage_score)
            conn.execute(
                "INSERT INTO review_queue (file, queue_type, priority_score, rationale) VALUES (?,?,?,?)",
                (file_name, "ownership_risk", priority,
                 f"bus_factor={bus_factor_risk} gain={semantic_gain} coverage={coverage_score} share={row['primary_author_share']}")
            )

        if not _is_non_actionable and semantic_gain >= 7 and tested_functions == 0:
            priority = semantic_gain * 6 + semantic_risk * 3 + max(0, 45 - coverage_score)
            conn.execute(
                "INSERT INTO review_queue (file, queue_type, priority_score, rationale) VALUES (?,?,?,?)",
                (file_name, "untested_hotspot", priority,
                 f"gain={semantic_gain} risk={semantic_risk} tested_functions={tested_functions} coverage={coverage_score}")
            )


def scan_ownership_metrics(conn):
    """Build coarse ownership/bus-factor signals from git blame."""
    files = conn.execute(
        "SELECT file, project FROM file_lines WHERE project IN ('Game', 'Common', 'Server_Link', 'LoginServer', 'ConnectServer')"
    ).fetchall()

    for row in files:
        file_name = row["file"]
        rel_path = _project_relative_path(row["project"], file_name)
        try:
            result = subprocess.run(
                ["git", "blame", "--line-porcelain", "--", rel_path],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=True,
            )
        except Exception:
            continue

        author_lines = defaultdict(int)
        total_lines = 0
        most_recent = None
        current_author = None
        current_time = None

        for raw_line in result.stdout.splitlines():
            if raw_line.startswith("author "):
                current_author = raw_line[7:].strip()
            elif raw_line.startswith("author-time "):
                try:
                    current_time = datetime.fromtimestamp(int(raw_line[12:].strip()), timezone.utc).date()
                except Exception:
                    current_time = None
            elif raw_line.startswith("\t"):
                total_lines += 1
                author_lines[current_author or "unknown"] += 1
                if current_time and (most_recent is None or current_time > most_recent):
                    most_recent = current_time

        if total_lines <= 0:
            continue

        primary_author, primary_lines = max(author_lines.items(), key=lambda item: item[1])
        primary_share = int((primary_lines * 100) / total_lines)
        author_count = len(author_lines)
        bus_factor_risk = 0
        if author_count <= 1:
            bus_factor_risk += 5
        elif author_count == 2:
            bus_factor_risk += 2
        if primary_share >= 80:
            bus_factor_risk += 4
        elif primary_share >= 60:
            bus_factor_risk += 2

        conn.execute(
            "INSERT INTO ownership_metrics (file, primary_author, primary_author_share, author_count, blamed_lines, most_recent_line_date, bus_factor_risk) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                file_name, primary_author, primary_share, author_count, total_lines,
                most_recent.isoformat() if most_recent else None, bus_factor_risk
            )
        )


def scan_test_function_map(conn):
    """Map tests to target functions using file edges and name mentions."""
    target_mentions = {}
    test_dir = REPO_ROOT / "UnitTests" / "Tests"
    mention_re = re.compile(r'\b([A-Za-z_]\w*)::([A-Za-z_~]\w*)\b')

    if test_dir.exists():
        for filepath in test_dir.glob("*.cpp"):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                target_mentions[_rel_name(filepath, test_dir)] = set(mention_re.findall(text))
            except Exception:
                target_mentions[_rel_name(filepath, test_dir)] = set()

    edges = conn.execute("SELECT source, target FROM edges WHERE edge_type = 'tests'").fetchall()
    for edge in edges:
        test_file = edge["source"]
        target_file = edge["target"]

        # Whole-file coverage mapping
        target_funcs = conn.execute(
            "SELECT function_name, class_name FROM function_index WHERE file = ? ORDER BY start_line",
            (target_file,)
        ).fetchall()
        for func in target_funcs:
            conn.execute(
                "INSERT INTO test_function_map (test_file, target_file, function_name, class_name, mapping_type) VALUES (?,?,?,?,?)",
                (test_file, target_file, func["function_name"], func["class_name"], "file_edge")
            )

        # Stronger mapping for explicit symbol mentions inside test source
        mentions = target_mentions.get(test_file, set())
        if not mentions:
            continue
        for class_name, function_name in mentions:
            match = conn.execute(
                "SELECT function_name, class_name FROM function_index WHERE file = ? AND function_name = ? AND COALESCE(class_name, '') = ? LIMIT 1",
                (target_file, function_name, class_name)
            ).fetchone()
            if match:
                conn.execute(
                    "INSERT INTO test_function_map (test_file, target_file, function_name, class_name, mapping_type) VALUES (?,?,?,?,?)",
                    (test_file, target_file, function_name, class_name, "explicit_mention")
                )

    # ---------- secp256k1-style test discovery ----------
    # Test files (test_*.cpp) in any source dir: match called function names
    # against function_index entries in non-test projects.
    _impl_projects = {"cpu", "include", "gpu", "cuda", "opencl", "metal"}
    call_re = re.compile(r'\b([a-zA-Z_]\w{3,})\s*\(')
    # Pre-load implementation function names for fast lookup
    impl_funcs = {}
    for row in conn.execute(
        "SELECT function_name, file, class_name FROM function_index WHERE project IN ({})".format(
            ",".join(f"'{p}'" for p in _impl_projects)
        )
    ).fetchall():
        impl_funcs.setdefault(row["function_name"], (row["file"], row["class_name"]))

    seen_mappings = set()
    for project, base_dir, _exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for pattern in ("test_*.cpp", "test_*.c"):
            for filepath in base_dir.rglob(pattern):
                test_rel = _rel_name(filepath, base_dir)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except Exception:
                    continue
                called = set(call_re.findall(text))
                for fn in called:
                    if fn in impl_funcs:
                        target_file, class_name = impl_funcs[fn]
                        key = (test_rel, target_file, fn)
                        if key in seen_mappings:
                            continue
                        seen_mappings.add(key)
                        try:
                            conn.execute(
                                "INSERT INTO test_function_map (test_file, target_file, function_name, class_name, mapping_type) VALUES (?,?,?,?,?)",
                                (test_rel, target_file, fn, class_name, "call_mention")
                            )
                        except Exception:
                            pass


def scan_call_edges(conn):
    """Heuristically build a function call graph from indexed function bodies."""
    rows = conn.execute(
        "SELECT file, function_name, class_name, start_line, end_line, project FROM function_index ORDER BY file, start_line"
    ).fetchall()
    by_name = defaultdict(list)
    for row in rows:
        by_name[row["function_name"]].append(row)

    file_cache = {}
    call_pattern_cache = {}
    skip_names = _get_call_skip_names()

    for row in rows:
        lines = _read_project_file_lines(row["project"], row["file"], file_cache)
        if not lines:
            continue

        body = _extract_function_body_text(lines, row["start_line"], row["end_line"])
        if not body:
            continue

        adapter = _project_adapter(row["project"])
        if row["project"] not in call_pattern_cache:
            call_pattern_cache[row["project"]] = adapter.call_patterns()
        call_pats = call_pattern_cache[row["project"]]
        plain_call_re = call_pats.get('plain', re.compile(r'\b([A-Za-z_~]\w*)\s*\('))
        scoped_call_re = call_pats.get('scoped', re.compile(r'\b([A-Za-z_]\w*)::([A-Za-z_~]\w*)\s*\('))
        member_call_re = call_pats.get('member', re.compile(r'(?:->|\.)\s*([A-Za-z_~]\w*)\s*\('))

        sanitized = adapter.strip_comments_and_strings(body)
        counts = defaultdict(int)
        scoped_counts = {}

        scoped_matches = scoped_call_re.findall(sanitized)
        for match in scoped_matches:
            if isinstance(match, tuple) and len(match) == 2:
                scope_name, callee_name = match
            else:
                continue
            if callee_name in skip_names:
                continue
            key = (scope_name, callee_name)
            scoped_counts[key] = scoped_counts.get(key, 0) + 1

        for callee_name in member_call_re.findall(sanitized):
            if callee_name in skip_names:
                continue
            counts[callee_name] += 1

        for callee_name in plain_call_re.findall(sanitized):
            if callee_name in skip_names:
                continue
            counts[callee_name] += 1

        caller_symbol = _function_label(row["class_name"], row["function_name"])
        inserted = set()

        for (scope_name, callee_name), call_count in scoped_counts.items():
            target_row, confidence = _resolve_call_target(row, scope_name, callee_name, by_name)
            if not target_row:
                continue
            callee_symbol = _function_label(target_row["class_name"], target_row["function_name"])
            dedup = (callee_symbol, target_row["file"])
            inserted.add(dedup)
            conn.execute(
                "INSERT OR REPLACE INTO call_edges (caller_symbol, caller_file, caller_project, callee_symbol, callee_file, callee_project, confidence, call_count, evidence) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    caller_symbol,
                    row["file"],
                    row["project"],
                    callee_symbol,
                    target_row["file"],
                    target_row["project"],
                    confidence,
                    call_count,
                    f"scoped-call:{scope_name}::{callee_name}"
                )
            )

        for callee_name, call_count in counts.items():
            target_row, confidence = _resolve_call_target(row, None, callee_name, by_name)
            if not target_row:
                continue
            callee_symbol = _function_label(target_row["class_name"], target_row["function_name"])
            dedup = (callee_symbol, target_row["file"])
            if dedup in inserted:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO call_edges (caller_symbol, caller_file, caller_project, callee_symbol, callee_file, callee_project, confidence, call_count, evidence) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    caller_symbol,
                    row["file"],
                    row["project"],
                    callee_symbol,
                    target_row["file"],
                    target_row["project"],
                    confidence,
                    call_count,
                    f"heuristic-call:{callee_name}"
                )
            )


def scan_symbol_metadata(conn):
    """Aggregate symbol-level metadata for reasoning and prioritization."""
    rows = conn.execute(
        "SELECT file, function_name, class_name, start_line, end_line, signature, project FROM function_index ORDER BY file, start_line"
    ).fetchall()

    file_cache = {}
    file_summaries = {
        row["file"]: row["summary"]
        for row in conn.execute("SELECT file, summary FROM file_summaries").fetchall()
    }
    audit_scores = {
        row["file"]: row["coverage_score"]
        for row in conn.execute("SELECT file, coverage_score FROM audit_coverage").fetchall()
    }
    history_metrics = {
        row["file"]: (row["recent_commit_count"], row["churn_score"])
        for row in conn.execute("SELECT file, recent_commit_count, churn_score FROM history_metrics").fetchall()
    }

    entity_tags = defaultdict(set)
    for row in conn.execute("SELECT entity_name, file, tag FROM semantic_tags").fetchall():
        entity_tags[(row["entity_name"], row["file"])].add(row["tag"])
    file_tags = defaultdict(set)
    for row in conn.execute("SELECT entity_name, file, tag FROM semantic_tags WHERE entity_type = 'file'").fetchall():
        file_tags[row["file"]].add(row["tag"])

    profiles = {}
    for row in conn.execute(
        "SELECT entity_name, file, security_sensitive, hot_path_candidate, optimization_candidate, network_surface, fragile_surface, risk_score, gain_score, evidence "
        "FROM semantic_profiles"
    ).fetchall():
        profiles[(row["entity_name"], row["file"])] = row

    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    for row in conn.execute(
        "SELECT caller_symbol, caller_file, callee_symbol, callee_file FROM call_edges"
    ).fetchall():
        outgoing[(row["caller_symbol"], row["caller_file"])] += 1
        incoming[(row["callee_symbol"], row["callee_file"])] += 1

    loop_re = re.compile(r'\bfor\s*\(|\bwhile\s*\(|\bdo\b')
    branch_re = re.compile(r'\bif\s*\(|\bswitch\s*\(|\?')

    for row in rows:
        symbol_name = _function_label(row["class_name"], row["function_name"])
        key = (symbol_name, row["file"])
        lines = _read_project_file_lines(row["project"], row["file"], file_cache)
        body = ""
        if lines:
            body = _extract_function_body_text(lines, row["start_line"], row["end_line"])
        sanitized = _strip_cpp_comments_and_strings(body)

        tags = set(file_tags.get(row["file"], set()))
        tags.update(entity_tags.get(key, set()))
        profile = profiles.get(key)
        line_span = row["end_line"] - row["start_line"] + 1
        loop_count = len(loop_re.findall(sanitized))
        branch_count = len(branch_re.findall(sanitized))
        caller_count = incoming.get(key, 0)
        callee_count = outgoing.get(key, 0)
        lower_symbol = symbol_name.lower()
        lower_file = row["file"].lower()

        hot_path = int(
            (profile["hot_path_candidate"] if profile else 0) or
            bool(tags & HOT_PATH_TAGS) or
            caller_count >= 3 or
            (loop_count >= 2 and line_span >= 40)
        )
        ct_sensitive = int(
            (profile["security_sensitive"] if profile else 0) or
            bool(tags & SECURITY_TAGS) or
            any(token in lower_symbol for token in ("auth", "login", "password", "secure", "warehouse", "token", "session"))
        )
        batchable = int(
            any(token in lower_symbol for token in ("batch", "scan", "verify", "broadcast", "sendall", "foreach")) or
            (loop_count >= 2 and caller_count >= 2)
        )
        gpu_candidate = int(
            (batchable and any(token in lower_symbol for token in ("batch", "scan", "verify", "field", "scalar", "point", "hash"))) or
            ("cuda" in lower_file or lower_file.endswith(".cu"))
        )

        base_risk = profile["risk_score"] if profile else 0
        base_gain = profile["gain_score"] if profile else 0
        audit_coverage_score = audit_scores.get(row["file"], 0)
        recent_commit_count, churn_score = history_metrics.get(row["file"], (0, 0))
        change_frequency = recent_commit_count + churn_score

        risk_score = base_risk + (4 if ct_sensitive else 0) + min(branch_count, 6) + (2 if tags & FRAGILE_TAGS else 0) + (1 if line_span > 120 else 0)
        gain_score = base_gain + (4 if hot_path else 0) + min(loop_count * 2, 6) + min(caller_count, 6) + (2 if batchable else 0) + (2 if gpu_candidate else 0)

        if risk_score >= 12:
            risk_level = "critical"
        elif risk_score >= 8:
            risk_level = "high"
        elif risk_score >= 4:
            risk_level = "medium"
        else:
            risk_level = "low"

        reasons = []
        if hot_path:
            reasons.append("hot-path heuristic matched")
        if ct_sensitive:
            reasons.append("security-sensitive symbol or file")
        if batchable:
            reasons.append("loop-heavy or batch-style workload")
        if gpu_candidate:
            reasons.append("candidate for parallel/offload review")
        if caller_count >= 3:
            reasons.append(f"high fan-in via {caller_count} callers")
        if audit_coverage_score < 45:
            reasons.append(f"low file coverage score {audit_coverage_score}")
        if change_frequency >= 10:
            reasons.append(f"recent churn signal {change_frequency}")
        if not reasons:
            reasons.append("baseline semantic metadata")

        review_priority = gain_score * 4 + max(0, 60 - audit_coverage_score) // 5 + min(caller_count, 6) - (risk_score // 3)
        summary = row["signature"] or file_summaries.get(row["file"], "") or symbol_name

        conn.execute(
            "INSERT OR REPLACE INTO symbol_metadata (symbol_name, file_path, project, class_name, summary, semantic_tags, hot_path, ct_sensitive, batchable, gpu_candidate, "
            "risk_level, review_priority, risk_score, gain_score, audit_coverage_score, change_frequency, line_span, loop_count, branch_count, caller_count, callee_count, reasons) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                symbol_name,
                row["file"],
                row["project"],
                row["class_name"],
                summary[:240],
                ", ".join(sorted(tags))[:240],
                hot_path,
                ct_sensitive,
                batchable,
                gpu_candidate,
                risk_level,
                review_priority,
                risk_score,
                gain_score,
                audit_coverage_score,
                change_frequency,
                line_span,
                loop_count,
                branch_count,
                caller_count,
                callee_count,
                json.dumps(reasons[:8])
            )
        )


def scan_analysis_scores(conn):
    """Score symbols for proactive bottleneck and audit queue generation."""
    rows = conn.execute(
        "SELECT symbol_name, file_path, hot_path, ct_sensitive, batchable, gpu_candidate, risk_level, review_priority, risk_score, gain_score, "
        "audit_coverage_score, change_frequency, line_span, loop_count, branch_count, caller_count, callee_count, reasons "
        "FROM symbol_metadata"
    ).fetchall()

    hot_name_tokens = ("mul", "verify", "scan", "hash", "point", "field", "batch", "move", "update", "send")

    for row in rows:
        lower_name = row["symbol_name"].lower()
        hotness_score = (
            (5 if row["hot_path"] else 0) +
            min(row["loop_count"] * 2, 8) +
            min(row["caller_count"], 8) +
            (2 if row["line_span"] > 80 else 0) +
            (3 if any(token in lower_name for token in hot_name_tokens) else 0)
        )
        complexity_score = row["branch_count"] + row["loop_count"] * 2 + (row["line_span"] // 40) + min(row["callee_count"], 6)
        fanin_score = row["caller_count"]
        fanout_score = row["callee_count"]
        optimization_score = row["gain_score"] + (3 if row["batchable"] else 0) + (2 if row["hot_path"] else 0)
        gpu_score = (4 if row["gpu_candidate"] else 0) + (2 if row["batchable"] else 0) + min(row["loop_count"], 4) - (1 if row["ct_sensitive"] else 0)
        ct_risk_score = (
            (5 if row["ct_sensitive"] else 0) +
            (3 if row["risk_level"] == "critical" else 2 if row["risk_level"] == "high" else 1 if row["risk_level"] == "medium" else 0) +
            (2 if row["audit_coverage_score"] < 40 else 0) +
            (2 if complexity_score > 10 else 0)
        )
        audit_gap_score = max(0, 60 - row["audit_coverage_score"]) // 10
        if row["hot_path"] and row["audit_coverage_score"] < 50:
            audit_gap_score += 2
        if row["ct_sensitive"] and row["audit_coverage_score"] < 60:
            audit_gap_score += 2

        perf_priority = hotness_score * 4 + fanin_score * 2 + gpu_score * 3
        safe_priority = row["risk_score"] * 3 + ct_risk_score * 2 + audit_gap_score * 3
        overall_priority = perf_priority + audit_gap_score * 3 - ct_risk_score * 2

        reasons = []
        try:
            reasons.extend(json.loads(row["reasons"] or "[]"))
        except Exception:
            pass
        if hotness_score >= 10:
            reasons.append(f"hotness={hotness_score}")
        if gpu_score >= 6:
            reasons.append(f"gpu_score={gpu_score}")
        if audit_gap_score >= 3:
            reasons.append(f"audit_gap={audit_gap_score}")
        if ct_risk_score >= 6:
            reasons.append(f"ct_risk={ct_risk_score}")
        if not reasons:
            reasons.append("baseline analysis score")

        conn.execute(
            "INSERT OR REPLACE INTO analysis_scores (symbol_name, file_path, hotness_score, complexity_score, fanin_score, fanout_score, optimization_score, "
            "gpu_score, ct_risk_score, audit_gap_score, perf_priority, safe_priority, overall_priority, reasons) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["symbol_name"],
                row["file_path"],
                hotness_score,
                complexity_score,
                fanin_score,
                fanout_score,
                optimization_score,
                gpu_score,
                ct_risk_score,
                audit_gap_score,
                perf_priority,
                safe_priority,
                overall_priority,
                json.dumps(reasons[:10])
            )
        )


def scan_symbol_audit_coverage(conn):
    """Build symbol-level audit memory from tests, queues, and semantic evidence."""
    queue_by_file = defaultdict(set)
    for row in conn.execute("SELECT file, queue_type FROM review_queue").fetchall():
        queue_by_file[row["file"]].add(row["queue_type"])

    tests_by_symbol = defaultdict(lambda: {"count": 0, "types": set(), "tests": set()})
    for row in conn.execute(
        "SELECT target_file, function_name, class_name, mapping_type, test_file FROM test_function_map"
    ).fetchall():
        symbol_name = _function_label(row["class_name"], row["function_name"])
        key = (symbol_name, row["target_file"])
        tests_by_symbol[key]["count"] += 1
        tests_by_symbol[key]["types"].add(row["mapping_type"])
        tests_by_symbol[key]["tests"].add(row["test_file"])

    tag_rows = conn.execute("SELECT entity_name, file, tag FROM semantic_tags").fetchall()
    tags_by_symbol = defaultdict(set)
    for row in tag_rows:
        tags_by_symbol[(row["entity_name"], row["file"])].add(row["tag"])

    rows = conn.execute(
        "SELECT symbol_name, file_path, risk_score, gain_score, audit_coverage_score, hot_path, ct_sensitive, reasons "
        "FROM symbol_metadata"
    ).fetchall()

    for row in rows:
        key = (row["symbol_name"], row["file_path"])
        test_info = tests_by_symbol.get(key, {"count": 0, "types": set(), "tests": set()})
        covered_by_tests = 1 if test_info["count"] > 0 else 0
        queue_types = queue_by_file.get(row["file_path"], set())
        semantic_tags = tags_by_symbol.get(key, set())

        audit_modules = set()
        if row["ct_sensitive"]:
            audit_modules.add("ct_review")
        if row["hot_path"]:
            audit_modules.add("perf_review")
        if covered_by_tests:
            audit_modules.add("unit_tests")
        if "network-packet" in semantic_tags:
            audit_modules.add("packet_surface")
        if "security" in semantic_tags or "auth" in semantic_tags:
            audit_modules.add("security_review")

        coverage_score = row["audit_coverage_score"]
        coverage_score += min(len(test_info["tests"]) * 10, 30)
        coverage_score += 10 if "explicit_mention" in test_info["types"] else 0
        coverage_score += 10 if row["ct_sensitive"] and "ct_review" in audit_modules else 0
        coverage_score -= 10 if not covered_by_tests and row["hot_path"] else 0
        coverage_score = max(0, min(100, coverage_score))

        if coverage_score >= 75:
            last_status = "covered"
        elif coverage_score >= 45:
            last_status = "partial"
        else:
            last_status = "gap"

        historical_failures = len([queue for queue in queue_types if queue in {"audit_gap", "stability_hotspot", "untested_hotspot"}])
        evidence = []
        if covered_by_tests:
            evidence.append(f"tests={len(test_info['tests'])}")
        if test_info["types"]:
            evidence.append(f"mappings={','.join(sorted(test_info['types']))}")
        if queue_types:
            evidence.append(f"queues={','.join(sorted(queue_types))}")
        if audit_modules:
            evidence.append(f"modules={','.join(sorted(audit_modules))}")
        if not evidence:
            evidence.append("no direct audit evidence")

        conn.execute(
            "INSERT OR REPLACE INTO symbol_audit_coverage (symbol_name, file_path, covered_by_tests, test_count, mapping_types, review_queue_types, audit_modules, coverage_score, last_status, historical_failures, evidence) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["symbol_name"],
                row["file_path"],
                covered_by_tests,
                len(test_info["tests"]),
                ", ".join(sorted(test_info["types"])),
                ", ".join(sorted(queue_types)),
                ", ".join(sorted(audit_modules)),
                coverage_score,
                last_status,
                historical_failures,
                "; ".join(evidence)[:240]
            )
        )


def scan_ai_tasks(conn):
    """Generate a proactive AI task queue from bottlenecks and audit gaps."""
    rows = conn.execute(
        "SELECT b.symbol_name, b.file_path, b.overall_priority, b.perf_priority, b.safe_priority, b.hotness_score, b.gpu_score, b.ct_risk_score, b.audit_gap_score, "
        "b.semantic_tags, b.summary, s.coverage_score, s.last_status, sm.batchable, sm.gpu_candidate, sm.ct_sensitive "
        "FROM v_bottleneck_queue b "
        "LEFT JOIN symbol_audit_coverage s ON s.symbol_name = b.symbol_name AND s.file_path = b.file_path "
        "LEFT JOIN symbol_metadata sm ON sm.symbol_name = b.symbol_name AND sm.file_path = b.file_path "
        "ORDER BY b.overall_priority DESC, b.hotness_score DESC"
    ).fetchall()

    created = set()
    for row in rows:
        tasks = []
        if row["overall_priority"] >= 45:
            tasks.append((
                "optimize",
                row["overall_priority"],
                f"Analyze {row['symbol_name']} as a performance bottleneck. Focus on loop structure, call fan-in, memory layout, and low-risk simplifications. Preserve gameplay behavior and constant-time constraints where applicable.",
                f"priority={row['overall_priority']} hot={row['hotness_score']} summary={row['summary'] or row['symbol_name']}"
            ))
        if row["audit_gap_score"] >= 3 or (row["last_status"] or "unknown") == "gap":
            tasks.append((
                "audit_expand",
                max(row["safe_priority"], row["overall_priority"]),
                f"Expand audit coverage for {row['symbol_name']}. Focus on edge cases, failure handling, missing tests, and crash-sensitive branches in {row['file_path']}.",
                f"audit_gap={row['audit_gap_score']} coverage={row['coverage_score']} status={row['last_status']}"
            ))
        if row["ct_sensitive"] and row["ct_risk_score"] >= 6:
            tasks.append((
                "ct_review",
                row["safe_priority"] + 5,
                f"Review {row['symbol_name']} for constant-time and security-sensitive behavior. Check control-flow divergence, secret-dependent branching, and correctness under edge conditions.",
                f"ct_risk={row['ct_risk_score']} tags={row['semantic_tags'] or ''}"
            ))
        if row["gpu_candidate"] and row["gpu_score"] >= 6:
            tasks.append((
                "gpu_candidate",
                row["perf_priority"] + 3,
                f"Evaluate {row['symbol_name']} as a GPU or parallel batch candidate. Focus on batchability, data independence, and acceptable risk for offload or job parallelism.",
                f"gpu_score={row['gpu_score']} batchable={row['batchable']}"
            ))

        for task_type, priority, prompt, rationale in tasks:
            dedup = (task_type, row["symbol_name"], row["file_path"])
            if dedup in created:
                continue
            created.add(dedup)
            conn.execute(
                "INSERT OR REPLACE INTO ai_tasks (task_type, symbol_name, file_path, prompt, status, priority, created_at, rationale) VALUES (?,?,?,?,?,?,?,?)",
                (
                    task_type,
                    row["symbol_name"],
                    row["file_path"],
                    prompt,
                    "pending",
                    priority,
                    datetime.now(timezone.utc).isoformat(),
                    rationale[:240]
                )
            )


def scan_function_index(conn):
    """Scan source files for function definitions with exact line ranges."""
    dirs = [(label, path, exts) for label, path, exts in SOURCE_DIRS]

    count = 0
    for project, base_dir, exts in dirs:
        if not base_dir.exists():
            continue
        for ext in exts:
            for filepath in base_dir.rglob(ext):
                try:
                    n = _scan_file_functions(conn, filepath, project, base_dir)
                    count += n
                except Exception:
                    pass
    return count


def _scan_file_functions(conn, filepath, project, base_dir=None):
    """Extract function definitions from a single file with line ranges."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        return 0

    fname = _rel_name(filepath, base_dir) if base_dir else filepath.name
    count = 0

    adapter = _project_adapter(project)
    cleaned = adapter.strip_block_comments_from_lines(lines)

    # Get patterns from adapter
    func_sig_re = adapter.function_sig_pattern()
    ctor_re = adapter.constructor_pattern()
    skip_names = adapter.function_skip_names()

    if not func_sig_re:
        return 0

    i = 0
    while i < len(cleaned):
        line = cleaned[i]
        stripped = line.strip()

        # Skip empty, preprocessor, pure comments
        if not stripped or stripped.startswith('#'):
            i += 1
            continue

        # Try to match function signature
        m = func_sig_re.match(line)
        class_name = None
        func_name = None

        if m:
            # Determine class_name and func_name from adapter pattern groups
            if m.lastindex and m.lastindex >= 2:
                class_name = m.group(1)   # e.g. ClassName:: prefix for C++
                func_name = m.group(2)
            elif m.lastindex == 1:
                func_name = m.group(1)
            else:
                func_name = stripped.split('(')[0].split()[-1] if '(' in stripped else None
        elif ctor_re:
            m = ctor_re.match(line)
            if m:
                class_name = m.group(1) if m.lastindex and m.lastindex >= 1 else None
                func_name = m.group(1) if m.lastindex == 1 else m.group(2) if m.lastindex >= 2 else None
            else:
                i += 1
                continue
        else:
            i += 1
            continue

        if not func_name or func_name in skip_names:
            i += 1
            continue

        # Find the opening scope marker (brace for C++/Java/etc., colon for Python)
        scope_line = adapter.find_opening_scope(cleaned, i)
        if scope_line is None or scope_line - i > 15:
            i += 1
            continue

        # For brace-based languages, check it's not a forward declaration
        if adapter.is_declaration_not_definition(cleaned, i, scope_line):
            i += 1
            continue

        # Found a function definition — find the end of its scope
        start_line = i + 1  # 1-based
        end_line = adapter.find_scope_end(cleaned, scope_line)
        if end_line is None:
            i += 1
            continue

        end_line_1based = end_line + 1  # convert to 1-based

        # Build signature from original (non-cleaned) lines
        sig_parts = []
        for si in range(i, min(scope_line + 1, len(lines))):
            sig_parts.append(lines[si].strip())
        signature = ' '.join(sig_parts)
        # Trim at opening scope marker
        for trim_ch in ('{', ':'):
            brace_idx = signature.find(trim_ch)
            if brace_idx > 0:
                signature = signature[:brace_idx].strip()
                break
        if len(signature) > 200:
            signature = signature[:200] + "..."

        try:
            conn.execute(
                "INSERT INTO function_index (file, function_name, class_name, start_line, end_line, signature, project) VALUES (?,?,?,?,?,?,?)",
                (fname, func_name, class_name, start_line, end_line_1based, signature, project)
            )
            count += 1
        except Exception:
            pass

        # Store function body in snippet cache
        try:
            body_lines = lines[start_line - 1:end_line_1based]
            body_text = "".join(body_lines)
            body_hash = hashlib.sha256(body_text.encode("utf-8", errors="ignore")).hexdigest()
            body_line_count = end_line_1based - start_line + 1
            conn.execute(
                "INSERT INTO function_bodies (file, function_name, class_name, start_line, end_line, body, body_hash, line_count, project) VALUES (?,?,?,?,?,?,?,?,?)",
                (fname, func_name, class_name, start_line, end_line_1based, body_text, body_hash, body_line_count, project)
            )
        except Exception:
            pass

        # Skip past this function
        i = end_line + 1

    return count


def _find_opening_brace(lines, start):
    """Find the line containing the opening '{' of a function definition."""
    paren_depth = 0
    found_open_paren = False
    for i in range(start, min(start + 20, len(lines))):
        for ch in lines[i]:
            if ch == '(':
                paren_depth += 1
                found_open_paren = True
            elif ch == ')':
                paren_depth -= 1
            elif ch == '{' and found_open_paren and paren_depth <= 0:
                return i
            elif ch == ';' and found_open_paren and paren_depth <= 0:
                return None
    return None


def _find_function_end(lines, brace_line):
    """Find the closing '}' of a function by counting braces from the opening '{' line."""
    depth = 0
    for i in range(brace_line, len(lines)):
        for ch in lines[i]:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i
    return None


def scan_edges(conn):
    """Scan for relationships between files: tests, extends, implements."""
    # 1. Test edges: UnitTests/Tests/*.cpp → Game/*.cpp or Game/*.h
    test_dir = REPO_ROOT / "UnitTests" / "Tests"
    if test_dir.exists():
        include_re = re.compile(r'//\s*(?:Tests?\s+for|Reproduce\s+from)\s+(\S+\.(?:h|cpp))', re.IGNORECASE)
        # Also try filename-based matching: BloodCastleDefTests.cpp → BloodCastle
        test_name_re = re.compile(r'^(\w+?)(?:Def)?Tests?\.cpp$', re.IGNORECASE)
        for filepath in test_dir.glob("*.cpp"):
            test_file = _rel_name(filepath, test_dir)
            targets_found = set()
            # Check for explicit references in comments
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        if lineno > 30:
                            break
                        m = include_re.search(line)
                        if m:
                            targets_found.add(m.group(1))
            except Exception:
                pass

            # Filename-based matching
            m = test_name_re.match(test_file)
            if m:
                base = m.group(1)
                # Look for matching Game files
                for candidate_ext in ('.cpp', '.h'):
                    candidate = GAME_DIR / (base + candidate_ext)
                    if candidate.exists():
                        targets_found.add(base + candidate_ext)
                    # Also try with "Def" suffix for definition headers
                    candidate2 = GAME_DIR / (base + "Def" + candidate_ext)
                    if candidate2.exists():
                        targets_found.add(base + "Def" + candidate_ext)

            for target in targets_found:
                try:
                    conn.execute(
                        "INSERT INTO edges (source, target, edge_type, detail) VALUES (?,?,?,?)",
                        (test_file, target, "tests", f"Unit test file")
                    )
                except Exception:
                    pass

    # 2. Class hierarchy edges from classes table
    rows = conn.execute("SELECT name, parent_class, header FROM classes WHERE parent_class IS NOT NULL AND parent_class != ''").fetchall()
    for row in rows:
        try:
            conn.execute(
                "INSERT INTO edges (source, target, edge_type, detail) VALUES (?,?,?,?)",
                (row[0], row[1], "extends", f"in {row[2]}")
            )
        except Exception:
            pass


def build_fts_index(conn):
    """Rebuild full-text search index from all tables."""
    conn.execute("DELETE FROM fts_index")

    # Singletons
    for row in conn.execute("SELECT macro, class_name, header, category, description FROM singletons"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("singleton", f"{row[0]} {row[1]}", row[2], row[3] or "", row[4] or ""))

    # Events
    for row in conn.execute("SELECT name, cpp_file, singleton_macro, description FROM events"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("event", f"{row[0]} {row[2] or ''}", row[1] or "", "event", row[3] or ""))

    # AI handlers
    for row in conn.execute("SELECT file, target, description FROM ai_handlers"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("ai", row[1], row[0], "ai", row[2] or ""))

    # Player files
    for row in conn.execute("SELECT file, domain, key_methods, description FROM player_files"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("player_file", f"{row[1]} {row[2] or ''}", row[0], "player", row[3] or ""))

    # Constants
    for row in conn.execute("SELECT name, value, header, category, description FROM constants"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("constant", f"{row[0]}={row[1]}", row[2], row[3] or "", row[4] or ""))

    # Packet handlers
    for row in conn.execute("SELECT headcode_name, handler_method, source_file, handler_type, description FROM packet_handlers"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("handler", f"{row[0]} {row[1]}", row[2], row[3], row[4] or ""))

    # Config files
    for row in conn.execute("SELECT path, description FROM config_files"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("config", row[0], row[0], "config", row[1] or ""))

    # Enums
    for row in conn.execute("SELECT name, file, values_preview, category FROM enums"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("enum", row[0], row[1], row[3] or "", row[2] or ""))

    # Structs
    for row in conn.execute("SELECT name, file, fields_preview, category FROM structs"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("struct", row[0], row[1], row[3] or "", row[2] or ""))

    # Classes (hierarchy)
    for row in conn.execute("SELECT name, header, parent_class, project FROM classes"):
        desc = f"extends {row[2]}" if row[2] else "base class"
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("class", row[0], row[1] or "", row[3] or "", desc))

    # Methods
    for row in conn.execute("SELECT class_name, method_name, file FROM methods"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("method", f"{row[0]}::{row[1]}", row[2], "method", ""))

    # DB tables
    seen_tables = set()
    for row in conn.execute("SELECT DISTINCT table_name, source_file, query_type FROM db_tables"):
        key = row[0]
        if key not in seen_tables:
            seen_tables.add(key)
            conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                         ("db_table", row[0], row[1], "database", f"Used with {row[2] or 'query'}"))

    # TODOs (only add summary per file, not every single one)
    for row in conn.execute("SELECT file, todo_type, COUNT(*) as cnt FROM todos GROUP BY file, todo_type"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("todo", f"{row[1]}x{row[2]}", row[0], "todo", f"{row[2]} {row[1]} comments"))

    # Prepared statements
    for row in conn.execute("SELECT query_name, sql_text, connection_type, source_file FROM prepared_statements"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("prepared_stmt", row[0], row[3], "database", row[1][:100]))

    # Config keys (deduplicated)
    seen_keys = set()
    for row in conn.execute("SELECT key_name, getter_type, source_file FROM config_keys"):
        if row[0] not in seen_keys:
            seen_keys.add(row[0])
            conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                         ("config_key", row[0], row[2], "config", f"Read via {row[1]}"))

    # Defines
    for row in conn.execute("SELECT name, value, file, category FROM defines"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("define", f"{row[0]}={row[1]}", row[2], row[3] or "", ""))

    # Leak risks
    for row in conn.execute("SELECT file, project, risk_score, sample_lines FROM leak_risks WHERE risk_score > 0"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("leak_risk", f"risk={row[2]}", row[0], row[1], (row[3] or "")[:200]))

    # NULL risks
    for row in conn.execute("SELECT file, function_call, pointer_var, risk_type, context FROM null_risks"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("null_risk", row[1], row[0], row[3], (row[4] or "")[:200]))

    # Raw pointers
    for row in conn.execute("SELECT file, class_name, member_type, member_name FROM raw_pointers"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("raw_pointer", f"{row[1]}::{row[3]}", row[0], row[2], ""))

    # Unsafe casts
    for row in conn.execute("SELECT file, cast_expr, context FROM unsafe_casts"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("unsafe_cast", row[1], row[0], "", (row[2] or "")[:200]))

    # Dead methods
    for row in conn.execute("SELECT class_name, method_name, header_file FROM dead_methods"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("dead_method", f"{row[0]}::{row[1]}", row[2], "", ""))

    # Infinite loop risks
    for row in conn.execute("SELECT file, line, risk_type, severity, expression FROM infinite_loop_risks"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("loop_risk", f"{row[2]}:{row[4]}", row[0], row[3], f"L{row[1]} {row[2]}"))

    # Duplicate blocks
    for row in conn.execute("SELECT file_a, file_b, line_a, line_b, preview FROM duplicate_blocks"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("duplicate", row[4] or "", f"{row[0]}:{row[2]}", f"{row[1]}:{row[3]}", ""))

    # File summaries
    for row in conn.execute("SELECT file, project, summary, category FROM file_summaries"):
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("file_summary", row[0], row[0], row[3] or "", row[2] or ""))

    # Research assets
    for row in conn.execute("SELECT path, file_name, asset_type, title, summary, symbol_refs, protocol_refs, notes FROM research_assets"):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "research_asset",
                f"{row[1]} {row[3] or ''}",
                row[0],
                row[2] or "",
                _trim_text(f"{row[4] or ''} symbols={row[5] or ''} protocols={row[6] or ''} {row[7] or ''}", 220),
            ),
        )

    # Research mentions
    for row in conn.execute("SELECT asset_path, symbol, mention_type, context FROM research_mentions"):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            ("research_mention", row[1], row[0], row[2], row[3] or ""),
        )

    # Function index
    for row in conn.execute("SELECT file, function_name, class_name, start_line, end_line, signature, project FROM function_index"):
        label = f"{row[2]}::{row[1]}" if row[2] else row[1]
        conn.execute("INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
                     ("function", label, row[0], row[6] or "", f"L{row[3]}-{row[4]} {(row[5] or '')[:100]}"))

    # Semantic tags
    for row in conn.execute("SELECT entity_type, entity_name, file, tag, confidence, evidence FROM semantic_tags"):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            ("semantic_tag", f"{row[3]} {row[1]}", row[2], row[0], f"confidence={row[4]} {row[5] or ''}")
        )

    # Semantic profiles
    for row in conn.execute(
        "SELECT entity_type, entity_name, file, risk_score, gain_score, security_sensitive, "
        "hot_path_candidate, optimization_candidate, evidence FROM semantic_profiles"
    ):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "semantic_profile",
                f"{row[1]} risk={row[3]} gain={row[4]}",
                row[2],
                row[0],
                f"security={row[5]} hot={row[6]} opt={row[7]} {row[8] or ''}"
            )
        )

    # Audit coverage
    for row in conn.execute(
        "SELECT file, project, coverage_score, unit_test_refs, crash_risk_count, null_risk_count, notes FROM audit_coverage"
    ):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "audit_coverage",
                f"{row[0]} coverage={row[2]}",
                row[0],
                row[1] or "",
                f"tests={row[3]} crash={row[4]} null={row[5]} {row[6] or ''}"
            )
        )

    # History metrics
    for row in conn.execute(
        "SELECT file, commit_count, recent_commit_count, bugfix_commits, perf_commits, audit_commits, churn_score FROM history_metrics"
    ):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "history_metrics",
                f"{row[0]} churn={row[6]}",
                row[0],
                "history",
                f"commits={row[1]} recent={row[2]} fixes={row[3]} perf={row[4]} audit={row[5]}"
            )
        )

    # Review queue
    for row in conn.execute("SELECT file, queue_type, priority_score, rationale FROM review_queue"):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            ("review_queue", f"{row[1]} priority={row[2]}", row[0], row[1], row[3] or "")
        )

    # Ownership metrics
    for row in conn.execute(
        "SELECT file, primary_author, primary_author_share, author_count, bus_factor_risk FROM ownership_metrics"
    ):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "ownership_metrics",
                f"{row[0]} owner={row[1] or 'unknown'}",
                row[0],
                "ownership",
                f"share={row[2]} authors={row[3]} bus_factor_risk={row[4]}"
            )
        )

    # Test function map
    for row in conn.execute(
        "SELECT test_file, target_file, function_name, class_name, mapping_type FROM test_function_map"
    ):
        label = f"{row[3]}::{row[2]}" if row[3] else row[2]
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            ("test_function_map", label, row[1], row[4], f"test={row[0]}")
        )

    # Call graph
    for row in conn.execute(
        "SELECT caller_symbol, caller_file, callee_symbol, callee_file, confidence, call_count, evidence FROM call_edges"
    ):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "call_edge",
                f"{row[0]} -> {row[2]}",
                row[1],
                "call-graph",
                f"callee={row[3]} confidence={row[4]} count={row[5]} {row[6] or ''}"
            )
        )

    # Symbol metadata
    for row in conn.execute(
        "SELECT symbol_name, file_path, semantic_tags, hot_path, ct_sensitive, batchable, gpu_candidate, risk_level, review_priority, reasons FROM symbol_metadata"
    ):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "symbol_metadata",
                row[0],
                row[1],
                row[7],
                f"tags={row[2] or ''} hot={row[3]} ct={row[4]} batch={row[5]} gpu={row[6]} priority={row[8]} {row[9] or ''}"
            )
        )

    # Analysis scores
    for row in conn.execute(
        "SELECT symbol_name, file_path, hotness_score, complexity_score, fanin_score, gpu_score, ct_risk_score, audit_gap_score, overall_priority, reasons FROM analysis_scores"
    ):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "analysis_score",
                f"{row[0]} priority={row[8]}",
                row[1],
                "bottleneck",
                f"hot={row[2]} complexity={row[3]} fanin={row[4]} gpu={row[5]} ct={row[6]} gap={row[7]} {row[9] or ''}"
            )
        )

    # Symbol audit coverage
    for row in conn.execute(
        "SELECT symbol_name, file_path, coverage_score, last_status, test_count, mapping_types, audit_modules, evidence FROM symbol_audit_coverage"
    ):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "symbol_audit",
                f"{row[0]} coverage={row[2]}",
                row[1],
                row[3],
                f"tests={row[4]} mappings={row[5] or ''} modules={row[6] or ''} {row[7] or ''}"
            )
        )

    # AI tasks
    for row in conn.execute(
        "SELECT task_type, symbol_name, file_path, status, priority, rationale FROM ai_tasks"
    ):
        conn.execute(
            "INSERT INTO fts_index (entity_type, name, file, category, description) VALUES (?,?,?,?,?)",
            (
                "ai_task",
                f"{row[0]} {row[1]}",
                row[2],
                row[3],
                f"priority={row[4]} {row[5] or ''}"
            )
        )


# ============================================================
# BUILD COMMAND
# ============================================================

def _bprint(msg):
    """Print a build-progress message unless --quiet is set."""
    if not _QUIET:
        print(msg)

# ============================================================
# SCANNER: DATAFLOW GRAPH
# ============================================================

# secp256k1-specific type vocabulary (ordered from primitive → complex)
_SECP256K1_TYPES = [
    "uint8_t", "uint32_t", "uint64_t",
    "Scalar", "FieldElement", "FieldElem", "fe",
    "Point", "GroupElement", "ge", "gej",
    "ECDSASignature", "SchnorrSignature", "RecoverableSignature",
    "Scalar", "PrivateKey", "PublicKey", "SecretKey",
    "SchnorrKeypair", "SchnorrXonlyPubkey",
    "MuSig2State", "FROSTShare", "FROSTSignature",
    "SHA256", "SHA512", "HMACSHA256", "HMACSHA512",
    "ECIESMessage", "PedersenCommitment",
    "array", "vector", "span",
]

_TYPE_NORMALIZE_RE = re.compile(
    r'\b(fast::)?(secp256k1::(?:fast::)?)?'
    r'(Scalar|FieldElement|FieldElem|Point|ECDSASignature|SchnorrSignature|'
    r'RecoverableSignature|PrivateKey|PublicKey|SecretKey|SchnorrKeypair|'
    r'SchnorrXonlyPubkey|MuSig2State|FROSTShare|FROSTSignature|'
    r'SHA256|SHA512|HMAC|ECIESMessage|PedersenCommitment|GroupElement)\b'
)

def _extract_types_from_signature(sig_text):
    """Return list of normalized type names mentioned in a function signature."""
    found = []
    for m in _TYPE_NORMALIZE_RE.finditer(sig_text or ""):
        t = m.group(3)
        if t and t not in found:
            found.append(t)
    return found


def scan_dataflow(conn):
    """Build type-level dataflow graph: which types flow into and out of each function."""
    rows = conn.execute(
        "SELECT fi.file, fi.function_name, fi.class_name, fi.project, "
        "COALESCE(fs.params, fi.signature) AS params_text, "
        "COALESCE(fs.return_type, '') AS return_type "
        "FROM function_index fi "
        "LEFT JOIN function_summaries fs ON fs.file = fi.file AND fs.function_name = fi.function_name AND fs.start_line = fi.start_line "
        "ORDER BY fi.file, fi.start_line"
    ).fetchall()

    inserted = 0
    for row in rows:
        func = row["function_name"]
        ffile = row["file"]
        proj = row["project"] or ""
        params_text = row["params_text"] or ""
        return_type = row["return_type"] or ""

        param_types = _extract_types_from_signature(params_text)
        ret_types   = _extract_types_from_signature(return_type)

        # Build edges: param_type → return_type  via this function
        for pt in param_types:
            for rt in ret_types:
                if pt == rt:
                    kind = "alias"
                else:
                    kind = "transform"
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO dataflow_edges "
                        "(from_type, to_type, via_function, via_file, project, flow_kind, confidence) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (pt, rt, func, ffile, proj, kind, 70)
                    )
                    inserted += 1
                except Exception:
                    pass

        # Also record single-type "produce" (function produces this type from primitives)
        for rt in ret_types:
            if not param_types:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO dataflow_edges "
                        "(from_type, to_type, via_function, via_file, project, flow_kind, confidence) "
                        "VALUES (?,?,?,?,?,?,?)",
                        ("(input)", rt, func, ffile, proj, "produce", 50)
                    )
                    inserted += 1
                except Exception:
                    pass

    conn.commit()
    if not _QUIET:
        print(f"  [dataflow] {inserted} flow edges from {len(rows)} functions")


# ============================================================
# SCANNER: INVARIANTS
# ============================================================

_INVARIANT_PATTERNS = [
    # C assert / SECP256K1_ASSERT / custom ASSERT macros
    (re.compile(r'\b(?:assert|ASSERT|SECP256K1_ASSERT|VERIFY_CHECK|CHECK)\s*\(([^;]{1,200})\)'), "assert", "critical"),
    # static_assert
    (re.compile(r'\bstatic_assert\s*\(([^,;]{1,200})'), "static_assert", "info"),
    # REQUIRES / EXPECTS macros (GSL-style)
    (re.compile(r'\b(?:REQUIRES|EXPECTS|ENSURES|GUARANTEE)\s*\(([^;]{1,200})\)'), "precondition", "warning"),
    # CT_VERIFY / constant-time guards
    (re.compile(r'\b(?:CT_VERIFY|SECP256K1_DECLASSIFY|VALGRIND_MAKE_MEM_DEFINED)\s*\(([^;]{1,120})\)'), "ct_guard", "critical"),
]

# Comment invariant keywords: lines starting with // containing these trigger a "comment_constraint"
_COMMENT_INVARIANT_RE = re.compile(
    r'//.*\b(must|never|require[sd]?|precondition|postcondition|invariant|guaranteed|MUST|NEVER|assert)\b'
    r'(?::?\s*)([^/\n]{0,120})',
    re.IGNORECASE
)

# Types that comment invariants frequently constrain
_CONSTRAINT_TYPE_RE = re.compile(
    r'\b(Scalar|FieldElement|Point|ECDSASignature|SchnorrSignature|RecoverableSignature|'
    r'PrivateKey|PublicKey|nonce|hash|recid|r|s|k)\b'
)


def scan_invariants(conn):
    """Extract assert/precondition/comment constraints from function bodies."""
    rows = conn.execute(
        "SELECT fb.file, fb.function_name, fb.class_name, fb.start_line, fb.body, fb.project "
        "FROM function_bodies fb ORDER BY fb.file, fb.start_line"
    ).fetchall()

    inserted = 0
    for row in rows:
        fname = row["function_name"]
        ffile = row["file"]
        start = row["start_line"] or 0
        body = row["body"] or ""
        lines = body.splitlines()

        for lineno, line in enumerate(lines, start=start):
            # Structural invariant patterns
            for pat, inv_type, severity in _INVARIANT_PATTERNS:
                for m in pat.finditer(line):
                    text = m.group(1).strip()[:240]
                    # Detect which type this constrains
                    tm = _CONSTRAINT_TYPE_RE.search(text)
                    related = tm.group(1) if tm else None
                    try:
                        conn.execute(
                            "INSERT INTO invariants "
                            "(symbol_name, file, line, invariant_text, invariant_type, severity, confidence, related_type) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            (fname, ffile, lineno, text, inv_type, severity, 75, related)
                        )
                        inserted += 1
                    except Exception:
                        pass

            # Comment constraints
            cm = _COMMENT_INVARIANT_RE.search(line)
            if cm:
                text = (cm.group(1) + ": " + cm.group(2)).strip()[:240]
                tm = _CONSTRAINT_TYPE_RE.search(text)
                related = tm.group(1) if tm else None
                try:
                    conn.execute(
                        "INSERT INTO invariants "
                        "(symbol_name, file, line, invariant_text, invariant_type, severity, confidence, related_type) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (fname, ffile, lineno, text, "comment_constraint", "info", 50, related)
                    )
                    inserted += 1
                except Exception:
                    pass

    conn.commit()
    if not _QUIET:
        print(f"  [invariants] {inserted} invariant entries from {len(rows)} function bodies")


# ============================================================
# SCANNER: BACKEND MAP (CPU ↔ CUDA ↔ OpenCL ↔ Metal)
# ============================================================

# Backend directory patterns relative to project root
_BACKEND_DIRS = {
    "cpu":    ["cpu/src", "cpu/include"],
    "cuda":   ["gpu/src", "gpu/include", "gpu/kernels"],
    "opencl": ["opencl", "opencl/kernels", "opencl/src"],
    "metal":  ["metal", "metal/src", "metal/kernels", "shaders"],
}

# Direct project label → backend mapping (overrides path/extension heuristics)
_PROJECT_BACKEND = {"cpu": "cpu", "cuda": "cuda", "opencl": "opencl", "metal": "metal"}

# Normalize function names for cross-backend matching:
# strip backend prefixes, lower-case, remove _impl/_kernel/_cl/_cu
_BACKEND_NORM_RE = re.compile(
    r'^(?:ufsecp_gpu_|ufsecp_|secp256k1_|secp256k1::|ct::|ct_|fast::)?'
    r'(?:point_)?'
    r'(.*?)(?:_impl|_kernel|_cl|_cu|_metal|_host|_dispatch)?$',
    re.IGNORECASE
)

# Synonym map for cross-backend naming differences.
# Keys are normalized names (after _BACKEND_NORM_RE; already lower-cased).
# Values are the canonical operation name used for parity matching.
_OP_SYNONYMS = {
    # jacobian ↔ point naming (OpenCL uses "point_*", CUDA/Metal use "jacobian_*")
    "jacobian_double": "double",
    "jacobian_add": "add",
    "jacobian_add_mixed": "add_mixed",
    "jacobian_add_mixed_h": "add_mixed_h",
    "jacobian_to_affine": "to_affine",
    "dbl_inplace": "double",
    "double": "double",
    "add_mixed": "add_mixed",
    # _batch suffix ↔ batch_ prefix (Metal: ecdsa_verify_batch; CPU: ecdsa_batch_verify)
    "ecdsa_verify_batch": "ecdsa_batch_verify",
    "schnorr_verify_batch": "schnorr_batch_verify",
    "ecdsa_sign_batch": "ecdsa_batch_sign",
    "schnorr_sign_batch": "schnorr_batch_sign",
    # Metal batch kernels that drop the middle batch_ prefix
    "ecdsa_snark_witness_batch": "ecdsa_snark_witness",
    "frost_verify_partial_batch": "frost_verify_partial",
    # Metal BIP324 kernel naming (kernel_bip324_ prefix + _batch suffix)
    "kernel_bip324_chacha20_block_batch": "chacha20_block",
    "kernel_bip324_aead_encrypt": "bip324_encrypt",
    "kernel_bip324_aead_decrypt": "bip324_decrypt",
    # Metal batch_inverse vs CPU fe_batch_inverse
    "batch_inverse": "fe_batch_inverse",
    # Metal uses scalar_mul_batch for generator multiplication
    "scalar_mul_batch": "scalar_mul_generator",
}

def _normalize_op(name):
    m = _BACKEND_NORM_RE.match(name or "")
    base = m.group(1).lower() if m else (name or "").lower()
    return _OP_SYNONYMS.get(base, base)


def scan_backend_map(conn):
    """Match CPU/CUDA/OpenCL/Metal functions by normalized operation name."""
    # Gather all functions per backend bucket
    all_funcs = conn.execute(
        "SELECT file, function_name, project FROM function_index ORDER BY file"
    ).fetchall()

    backend_funcs = {"cpu": {}, "cuda": {}, "opencl": {}, "metal": {}}

    # Skip test/bench functions and builtins that are not real backend operations
    _SKIP_PREFIXES = ("test_", "bench_", "fuzz_", "run_", "main")
    _SKIP_NAMES = {
        "clobbermemory", "donotoptimize", "alignas", "block", "check",
        "printf", "print", "assert", "abort", "exit", "memset", "memcpy",
        "memmove", "malloc", "free", "calloc", "realloc", "strlen",
    }

    for row in all_funcs:
        f = row["file"].replace("\\", "/")
        fn = row["function_name"]
        project = row["project"]

        # Skip C ABI wrappers — they dispatch to backends, not implementations
        if fn.startswith("ufsecp_"):
            continue

        # Skip test/bench/fuzz functions and builtins
        fn_lower = fn.lower()
        if fn_lower in _SKIP_NAMES or any(fn_lower.startswith(p) for p in _SKIP_PREFIXES):
            continue

        # Skip non-implementation projects (tests, benchmarks, docs, tooling)
        if project in ("audit", "tests", "examples", "benchmarks", "docs",
                        "scripts", "tools", "bindings", "bindings_rust",
                        "github", "repo_root_meta"):
            continue

        backend = None

        # 1. Direct project → backend for known backend projects
        if project in _PROJECT_BACKEND:
            backend = _PROJECT_BACKEND[project]
        # 2. gpu project: classify by filename/extension
        elif project == "gpu":
            fl = f.lower()
            if "opencl" in fl:
                backend = "opencl"
            elif "metal" in fl or fl.endswith(".mm"):
                backend = "metal"
            elif "cuda" in fl or fl.endswith(".cu"):
                backend = "cuda"
        # 3. include project: classify by extension
        else:
            if f.endswith((".cu", ".cuh")):
                backend = "cuda"
            elif f.endswith(".cl"):
                backend = "opencl"
            elif f.endswith(".metal"):
                backend = "metal"
            elif project == "include" and f.endswith((".cpp", ".c", ".h", ".hpp", ".hh")):
                backend = "cpu"

        if backend:
            norm = _normalize_op(fn)
            if norm and len(norm) > 3:
                if norm not in backend_funcs[backend]:
                    backend_funcs[backend][norm] = (fn, f)

    # Union of all known operations
    all_ops = set()
    for bd in backend_funcs.values():
        all_ops.update(bd.keys())

    # Host-utility and CT-primitive inline operations — these exist as device inlines
    # inside GPU kernels, not as top-level dispatch targets.  Do not report parity
    # gaps for them; they are not independently dispatchable on GPU.
    _NON_GPU_OPS = frozenset({
        "add64", "mul64", "rotl32", "rotr32", "store32_le", "load32_le",
        "field_to_hex", "write_json", "print_row", "print_usage",
        "has_flag", "fe_eq", "selftest",
        "normalize", "make_key", "make_scalar", "random_scalar",
        "scalar_bitlen", "scalar_eq", "scalar_is_high", "scalar_is_zero",
        "x_bytes_and_parity", "x_only_bytes", "x_coord",
        "from_mont", "lookup_256", "from_hex", "to_hex",
        "next", "next_inplace", "finish",
        # CT primitive inlines (device inlines in all GPU kernels, not top-level)
        "select", "cmov256", "cmov64", "cswap256", "add256", "sub256",
        # BIP/encoding / short-predicate helpers
        "rfc6979_nonce_hedged", "schnorr_xonly_pubkey_parse",
        "has_even_y", "is_valid", "points_equal",
        "from_compressed", "to_uncompressed",
        "x_bytes", "negate", "zero", "infinity",
        # Hash/crypto primitives that exist as device inlines, not top-level kernels
        "sha256", "sha512", "inverse",
        # Field/point host-side helpers (CPU+CUDA+Metal, not separately in OpenCL)
        "from_bytes", "to_bytes", "is_zero", "generator", "from_uint64",
        "hex_to_bytes32", "fe_batch_inverse",
        # Signing ops that must stay CPU-only (private key must not be on GPU)
        "ecdsa_sign_hedged", "ecdsa_sign_hedged_verified",
        # Misc bench/test utility
        "range",
    })

    inserted = 0
    for op in sorted(all_ops):
        cpu    = backend_funcs["cpu"].get(op)
        cuda   = backend_funcs["cuda"].get(op)
        opencl = backend_funcs["opencl"].get(op)
        metal  = backend_funcs["metal"].get(op)

        present = sum(1 for x in [cpu, cuda, opencl, metal] if x)
        gpu_count = sum(1 for x in [cuda, opencl, metal] if x)

        if present == 4:
            parity = "full"
        elif cpu and gpu_count == 0:
            parity = "cpu-only"
        elif not cpu and gpu_count == 3:
            # Present on all 3 GPU backends but not CPU — full GPU parity
            parity = "gpu-full"
        elif not cpu and gpu_count >= 1:
            # GPU-only function present on 1-2 backends — internal kernel
            # helper that does not need cross-backend parity.
            parity = "gpu-internal"
        elif present == 1:
            parity = "single-backend"
        elif not metal and gpu_count >= 1:
            parity = "missing-metal"
        elif not opencl and gpu_count >= 1:
            parity = "missing-opencl"
        elif not cuda and gpu_count >= 1:
            parity = "missing-cuda"
        else:
            parity = "partial"

        # Do not report host-utility inlines as cross-backend parity gaps.
        if op in _NON_GPU_OPS and parity in (
            "missing-metal", "missing-opencl", "missing-cuda", "partial", "single-backend"
        ):
            continue

        try:
            conn.execute(
                "INSERT OR REPLACE INTO backend_map "
                "(operation_name, cpu_symbol, cpu_file, cuda_symbol, cuda_file, "
                "opencl_symbol, opencl_file, metal_symbol, metal_file, parity_status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    op,
                    cpu[0] if cpu else None,    cpu[1] if cpu else None,
                    cuda[0] if cuda else None,   cuda[1] if cuda else None,
                    opencl[0] if opencl else None, opencl[1] if opencl else None,
                    metal[0] if metal else None,  metal[1] if metal else None,
                    parity,
                )
            )
            inserted += 1
        except Exception:
            pass

    conn.commit()
    if not _QUIET:
        print(f"  [backend_map] {inserted} operations mapped across backends")


def build():
    """Build the complete source graph database."""
    _bprint("[*] Creating database...")
    conn = create_db()

    _bprint("[*] Recording graph metadata...")
    populate_graph_metadata(conn)

    if SEED_DATA:
        _bprint("[*] Populating seed data from config...")
        populate_seed_data(conn)
    else:
        _bprint("[*] Populating singletons (120+)...")
        populate_singletons(conn)
        _bprint("[*] Populating Player files (28)...")
        populate_player_files(conn)
        _bprint("[*] Populating events (35)...")
        populate_events(conn)
        _bprint("[*] Populating AI handlers (43)...")
        populate_ai_handlers(conn)
        _bprint("[*] Populating inventory scripts (12)...")
        populate_inventory_scripts(conn)
        _bprint("[*] Populating constants (35+)...")
        populate_constants(conn)
        _bprint("[*] Populating config files (20)...")
        populate_config_files(conn)
        _bprint("[*] Populating packet handlers (40+)...")
        populate_packet_handlers(conn)

    print("[*] Scanning source files...")
    scan_files(conn)

    print("[*] Scanning #include dependencies...")
    scan_includes(conn)

    print("[*] Scanning singleton usage patterns...")
    scan_singleton_usage(conn)

    print("[*] Scanning enums...")
    scan_enums(conn)

    print("[*] Scanning structs/packets...")
    scan_structs(conn)

    print("[*] Scanning class hierarchy...")
    scan_classes(conn)

    print("[*] Scanning methods...")
    scan_methods(conn)

    print("[*] Scanning TODO/FIXME/HACK comments...")
    scan_todos(conn)

    print("[*] Scanning database table usage...")
    scan_db_tables(conn)

    print("[*] Scanning prepared statements...")
    scan_prepared_statements(conn)

    print("[*] Scanning config keys...")
    scan_config_keys(conn)

    print("[*] Scanning #define macros...")
    scan_defines(conn)

    print("[*] Counting file lines...")
    scan_file_lines(conn)

    print("[*] Analyzing memory leak risks...")
    scan_leak_risks(conn)

    print("[*] Scanning NULL-check risks...")
    scan_null_risks(conn)

    print("[*] Scanning raw pointer members...")
    scan_raw_pointers(conn)

    print("[*] Scanning C-style casts...")
    scan_unsafe_casts(conn)

    print("[*] Scanning crash risk patterns...")
    scan_crash_risks(conn)

    print("[*] Scanning infinite loop risks...")
    scan_infinite_loop_risks(conn)

    print("[*] Detecting dead methods...")
    scan_dead_methods(conn)

    print("[*] Detecting duplicate code blocks...")
    scan_duplicate_blocks(conn)

    print("[*] Generating file summaries...")
    scan_file_summaries(conn)

    print("[*] Indexing ReversingResearch assets...")
    scan_research_assets(conn)

    print("[*] Scanning function definitions (line ranges)...")
    func_count = scan_function_index(conn)
    print(f"    Found {func_count} function definitions")

    print("[*] Scanning edges (tests, extends)...")
    scan_edges(conn)

    print("[*] Scanning heuristic call graph...")
    scan_call_edges(conn)

    print("[*] Deriving semantic tags...")
    scan_semantic_tags(conn)

    print("[*] Aggregating semantic profiles...")
    scan_semantic_profiles(conn)

    print("[*] Aggregating audit coverage...")
    scan_audit_coverage(conn)

    print("[*] Aggregating git history metrics...")
    scan_history_metrics(conn)

    print("[*] Aggregating ownership metrics...")
    scan_ownership_metrics(conn)

    print("[*] Mapping tests to functions...")
    scan_test_function_map(conn)

    print("[*] Aggregating symbol metadata...")
    scan_symbol_metadata(conn)

    print("[*] Computing analysis scores...")
    scan_analysis_scores(conn)

    print("[*] Building review queue...")
    scan_review_queue(conn)

    print("[*] Building symbol audit coverage...")
    scan_symbol_audit_coverage(conn)

    print("[*] Generating AI task queue...")
    scan_ai_tasks(conn)

    print("[*] Building full-text search index...")
    build_fts_index(conn)

    print("[*] Building type-level dataflow graph...")
    scan_dataflow(conn)

    print("[*] Extracting invariants (assert/precondition/comment)...")
    scan_invariants(conn)

    print("[*] Mapping CPU/CUDA/OpenCL/Metal backend parity...")
    scan_backend_map(conn)

    print("[*] Updating file hashes...")
    _update_file_hashes(conn)

    conn.commit()
    conn.close()

    db_size = DB_PATH.stat().st_size / 1024
    print(f"\n[+] Source graph built: {DB_PATH}")
    print(f"[+] Database size: {db_size:.0f} KB")
    stats_cmd(show_header=False)


def build_incremental():
    """Incremental build: skip if no files changed, full rebuild with preserve if changed."""
    if not DB_PATH.exists():
        print("[!] No existing database. Running full build...")
        build()
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Check if file_hashes table exists
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "file_hashes" not in tables:
        print("[!] No file_hashes table. Running full build with preserve...")
        conn.close()
        _full_build_with_preserve()
        return

    print("[*] Computing changed files...")
    changed_files = _compute_changed_files(conn)
    conn.close()

    if not changed_files:
        print("[+] No files changed. Nothing to do.")
        return

    print(f"[*] {len(changed_files)} files changed. Rebuilding with preserve...")
    _full_build_with_preserve()


def _full_build_with_preserve():
    """Full build that preserves persistent tables."""
    conn = create_db(preserve_persistent=True)
    populate_graph_metadata(conn)

    if SEED_DATA:
        populate_seed_data(conn)
    else:
        populate_singletons(conn)
        populate_player_files(conn)
        populate_events(conn)
        populate_ai_handlers(conn)
        populate_inventory_scripts(conn)
        populate_constants(conn)
        populate_config_files(conn)
        populate_packet_handlers(conn)

    # Run all scanners (same as build())
    scan_files(conn)
    scan_includes(conn)
    scan_singleton_usage(conn)
    scan_enums(conn)
    scan_structs(conn)
    scan_classes(conn)
    scan_methods(conn)
    scan_todos(conn)
    scan_db_tables(conn)
    scan_prepared_statements(conn)
    scan_config_keys(conn)
    scan_defines(conn)
    scan_file_lines(conn)
    scan_leak_risks(conn)
    scan_null_risks(conn)
    scan_raw_pointers(conn)
    scan_unsafe_casts(conn)
    scan_crash_risks(conn)
    scan_infinite_loop_risks(conn)
    scan_dead_methods(conn)
    scan_duplicate_blocks(conn)
    scan_file_summaries(conn)
    scan_research_assets(conn)
    scan_function_index(conn)
    scan_edges(conn)
    scan_call_edges(conn)
    scan_dataflow(conn)
    scan_invariants(conn)
    scan_backend_map(conn)

    _rebuild_aggregation_tables(conn)
    _update_file_hashes(conn)

    conn.commit()
    conn.close()

    db_size = DB_PATH.stat().st_size / 1024
    print(f"\n[+] Full build (with preserve) complete: {DB_PATH}")
    print(f"[+] Database size: {db_size:.0f} KB")


def _rebuild_aggregation_tables(conn):
    """Rebuild tables that aggregate across files (always safe to fully rebuild)."""
    # Clear aggregation tables
    agg_tables = [
        "dead_methods", "duplicate_blocks", "semantic_tags", "semantic_profiles",
        "audit_coverage", "history_metrics", "review_queue", "ownership_metrics",
        "test_function_map", "symbol_metadata", "analysis_scores",
        "symbol_audit_coverage", "ai_tasks", "edges",
        "dataflow_edges", "invariants", "backend_map",
    ]
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for t in agg_tables:
        if t in tables:
            conn.execute(f"DELETE FROM {t}")
    conn.commit()

    scan_dead_methods(conn)
    scan_duplicate_blocks(conn)
    scan_edges(conn)
    scan_semantic_tags(conn)
    scan_semantic_profiles(conn)
    scan_audit_coverage(conn)
    scan_history_metrics(conn)
    scan_ownership_metrics(conn)
    scan_test_function_map(conn)
    scan_symbol_metadata(conn)
    scan_analysis_scores(conn)
    scan_review_queue(conn)
    scan_symbol_audit_coverage(conn)
    scan_ai_tasks(conn)
    build_fts_index(conn)
    scan_dataflow(conn)
    scan_invariants(conn)
    scan_backend_map(conn)


def _compute_changed_files(conn):
    """Compare current files against stored hashes, return set of changed file names."""
    changed = set()

    # Load existing hashes (keyed by "label/filename")
    stored = {}
    try:
        for row in conn.execute("SELECT file_path, content_hash, mtime_ns, size_bytes FROM file_hashes"):
            stored[row["file_path"]] = (row["content_hash"], row["mtime_ns"], row["size_bytes"])
    except Exception:
        return changed

    # Check all source files
    current_keys = set()
    for label, base_dir, exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for ext in exts:
            for f in base_dir.rglob(ext):
                key = f"{label}/{_rel_name(f, base_dir)}"
                current_keys.add(key)
                try:
                    st = f.stat()
                    mtime_ns = int(st.st_mtime_ns)
                    size_bytes = st.st_size
                except OSError:
                    changed.add(_rel_name(f, base_dir))
                    continue

                if key in stored:
                    old_hash, old_mtime, old_size = stored[key]
                    if mtime_ns == old_mtime and size_bytes == old_size:
                        continue  # Fast path: unchanged
                    # Size or mtime changed, check content hash
                    try:
                        content_hash = hashlib.sha256(f.read_bytes()).hexdigest()
                    except OSError:
                        changed.add(_rel_name(f, base_dir))
                        continue
                    if content_hash != old_hash:
                        changed.add(_rel_name(f, base_dir))
                else:
                    changed.add(_rel_name(f, base_dir))  # New file

    # Check for deleted files
    for key in stored:
        if key not in current_keys:
            # Extract filename from "label/filename"
            fname = key.split("/", 1)[-1] if "/" in key else key
            changed.add(fname)

    return changed


def _update_file_hashes(conn):
    """Update file_hashes table with current file state."""
    conn.execute("DELETE FROM file_hashes")
    now = datetime.now(timezone.utc).isoformat()

    for label, base_dir, exts in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for ext in exts:
            for f in base_dir.rglob(ext):
                try:
                    st = f.stat()
                    content_hash = hashlib.sha256(f.read_bytes()).hexdigest()
                    key = f"{label}/{_rel_name(f, base_dir)}"
                    conn.execute(
                        "INSERT OR REPLACE INTO file_hashes (file_path, content_hash, mtime_ns, size_bytes, last_scan_at) VALUES (?,?,?,?,?)",
                        (key, content_hash, int(st.st_mtime_ns), st.st_size, now)
                    )
                except OSError:
                    pass
    conn.commit()


# ============================================================
# SYMBOL GRAPH VIEW — in-memory graph projection
# ============================================================

class SymbolId:
    """Canonical identifier for a symbol node."""
    __slots__ = ("name", "file", "start_line")

    def __init__(self, name: str, file: str, start_line: int = 0):
        self.name = name
        self.file = file
        self.start_line = start_line

    def __hash__(self):
        return hash((self.name, self.file))

    def __eq__(self, other):
        return isinstance(other, SymbolId) and self.name == other.name and self.file == other.file

    def __repr__(self):
        return f"{self.name}@{self.file}"

    def key(self):
        return (self.name, self.file)


class SymbolNode:
    """Aggregated metadata for a single symbol in the graph."""
    __slots__ = (
        "sid", "class_name", "project", "summary", "start_line", "end_line",
        "tags", "hot_path", "ct_sensitive", "batchable", "gpu_candidate",
        "risk_score", "gain_score", "audit_coverage_score",
        "review_priority", "risk_level", "change_frequency",
        "line_span", "loop_count", "branch_count",
        "caller_count", "callee_count", "reasons",
        # analysis_scores fields
        "hotness_score", "complexity_score", "optimization_score",
        "gpu_score", "ct_risk_score", "audit_gap_score",
        "perf_priority", "overall_priority",
    )

    def __init__(self, sid: SymbolId):
        self.sid = sid
        self.class_name = None
        self.project = None
        self.summary = None
        self.start_line = 0
        self.end_line = 0
        self.tags = ""
        self.hot_path = 0
        self.ct_sensitive = 0
        self.batchable = 0
        self.gpu_candidate = 0
        self.risk_score = 0
        self.gain_score = 0
        self.audit_coverage_score = 0
        self.review_priority = 0
        self.risk_level = "low"
        self.change_frequency = 0
        self.line_span = 0
        self.loop_count = 0
        self.branch_count = 0
        self.caller_count = 0
        self.callee_count = 0
        self.reasons = ""
        self.hotness_score = 0
        self.complexity_score = 0
        self.optimization_score = 0
        self.gpu_score = 0
        self.ct_risk_score = 0
        self.audit_gap_score = 0
        self.perf_priority = 0
        self.overall_priority = 0

    def composite_score(self, mode: str) -> float:
        """Compute a task-aware composite relevance score."""
        if mode == "bugfix":
            return (
                self.risk_score * 3.0
                + self.caller_count * 2.0
                + self.change_frequency * 1.5
                + self.branch_count * 1.0
                + (10 if self.ct_sensitive else 0)
                - self.audit_coverage_score * 0.5
            )
        elif mode == "optimize":
            return (
                self.gain_score * 3.0
                + self.hotness_score * 2.5
                + self.optimization_score * 2.0
                + self.loop_count * 2.0
                + self.caller_count * 1.5
                + (8 if self.gpu_candidate else 0)
                + (6 if self.batchable else 0)
            )
        elif mode == "audit":
            return (
                self.ct_risk_score * 3.0
                + self.risk_score * 2.5
                + self.audit_gap_score * 2.0
                + (12 if self.ct_sensitive else 0)
                - self.audit_coverage_score * 0.8
                + self.change_frequency * 1.0
            )
        else:  # explore / generic
            return (
                self.review_priority * 2.0
                + self.gain_score * 1.5
                + self.risk_score * 1.0
                + self.caller_count * 1.0
            )

    def to_manifest_dict(self):
        """Return a machine-oriented dict for JSON output."""
        return {
            "symbol": self.sid.name,
            "file": self.sid.file,
            "line_range": [self.start_line, self.end_line],
            "class": self.class_name,
            "flags": {
                "hot_path": bool(self.hot_path),
                "ct_sensitive": bool(self.ct_sensitive),
                "batchable": bool(self.batchable),
                "gpu_candidate": bool(self.gpu_candidate),
            },
            "scores": {
                "risk": self.risk_score,
                "gain": self.gain_score,
                "review_priority": self.review_priority,
                "audit_coverage": self.audit_coverage_score,
                "hotness": self.hotness_score,
                "optimization": self.optimization_score,
            },
            "summary": self.summary or "",
        }


class SymbolEdge:
    """A typed, weighted edge between two symbol nodes."""
    __slots__ = ("source", "target", "edge_type", "confidence", "call_count", "weight")

    def __init__(self, source: SymbolId, target: SymbolId, edge_type: str = "calls",
                 confidence: int = 0, call_count: int = 1):
        self.source = source
        self.target = target
        self.edge_type = edge_type
        self.confidence = confidence
        self.call_count = call_count
        self.weight = confidence + call_count * 2

    def to_dict(self):
        return {
            "source": repr(self.source),
            "target": repr(self.target),
            "type": self.edge_type,
            "confidence": self.confidence,
            "call_count": self.call_count,
            "weight": self.weight,
        }


class SymbolGraphView:
    """In-memory graph projection built from the SQLite relational tables.

    Provides graph-oriented traversal (neighbors, shortest path, impact set)
    while the underlying DB stays relational.  Cheap to build — all heavy
    data stays in SQLite until explicitly pulled.
    """

    def __init__(self, conn):
        self._conn = conn
        self._nodes: dict[tuple, SymbolNode] = {}  # (name, file) -> SymbolNode
        self._outgoing: dict[tuple, list[SymbolEdge]] = defaultdict(list)
        self._incoming: dict[tuple, list[SymbolEdge]] = defaultdict(list)
        self._loaded = False

    # ------ lazy load ------

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._load_nodes()
        self._load_edges()
        self._loaded = True

    def _load_nodes(self):
        for row in self._conn.execute(
            "SELECT sm.symbol_name, sm.file_path, sm.project, sm.class_name, sm.summary,"
            " sm.semantic_tags, sm.hot_path, sm.ct_sensitive, sm.batchable, sm.gpu_candidate,"
            " sm.risk_level, sm.review_priority, sm.risk_score, sm.gain_score,"
            " sm.audit_coverage_score, sm.change_frequency, sm.line_span, sm.loop_count,"
            " sm.branch_count, sm.caller_count, sm.callee_count, sm.reasons,"
            " COALESCE(a.hotness_score,0) AS hotness_score,"
            " COALESCE(a.complexity_score,0) AS complexity_score,"
            " COALESCE(a.optimization_score,0) AS optimization_score,"
            " COALESCE(a.gpu_score,0) AS gpu_score,"
            " COALESCE(a.ct_risk_score,0) AS ct_risk_score,"
            " COALESCE(a.audit_gap_score,0) AS audit_gap_score,"
            " COALESCE(a.perf_priority,0) AS perf_priority,"
            " COALESCE(a.overall_priority,0) AS overall_priority"
            " FROM symbol_metadata sm"
            " LEFT JOIN analysis_scores a ON a.symbol_name = sm.symbol_name AND a.file_path = sm.file_path"
        ).fetchall():
            sid = SymbolId(row["symbol_name"], row["file_path"])
            n = SymbolNode(sid)
            n.class_name = row["class_name"]
            n.project = row["project"]
            n.summary = row["summary"]
            n.tags = row["semantic_tags"] or ""
            n.hot_path = row["hot_path"]
            n.ct_sensitive = row["ct_sensitive"]
            n.batchable = row["batchable"]
            n.gpu_candidate = row["gpu_candidate"]
            n.risk_level = row["risk_level"]
            n.review_priority = row["review_priority"]
            n.risk_score = row["risk_score"]
            n.gain_score = row["gain_score"]
            n.audit_coverage_score = row["audit_coverage_score"]
            n.change_frequency = row["change_frequency"]
            n.line_span = row["line_span"]
            n.loop_count = row["loop_count"]
            n.branch_count = row["branch_count"]
            n.caller_count = row["caller_count"]
            n.callee_count = row["callee_count"]
            n.reasons = row["reasons"] or ""
            n.hotness_score = row["hotness_score"]
            n.complexity_score = row["complexity_score"]
            n.optimization_score = row["optimization_score"]
            n.gpu_score = row["gpu_score"]
            n.ct_risk_score = row["ct_risk_score"]
            n.audit_gap_score = row["audit_gap_score"]
            n.perf_priority = row["perf_priority"]
            n.overall_priority = row["overall_priority"]
            # fill line range from function_index if present
            fi = self._conn.execute(
                "SELECT start_line, end_line FROM function_index WHERE function_name = ? AND file = ? LIMIT 1",
                (row["symbol_name"], row["file_path"])
            ).fetchone()
            if fi:
                n.start_line = fi["start_line"]
                n.end_line = fi["end_line"]
                sid.start_line = fi["start_line"]
            self._nodes[sid.key()] = n

    def _load_edges(self):
        for row in self._conn.execute(
            "SELECT caller_symbol, caller_file, callee_symbol, callee_file, confidence, call_count FROM call_edges"
        ).fetchall():
            src = SymbolId(row["caller_symbol"], row["caller_file"])
            tgt = SymbolId(row["callee_symbol"], row["callee_file"])
            edge = SymbolEdge(src, tgt, "calls", row["confidence"], row["call_count"])
            self._outgoing[src.key()].append(edge)
            self._incoming[tgt.key()].append(edge)
        # Also load generic edges table
        for row in self._conn.execute(
            "SELECT source, target, edge_type, detail FROM edges"
        ).fetchall():
            src = SymbolId(row["source"], "")
            tgt = SymbolId(row["target"], "")
            edge = SymbolEdge(src, tgt, row["edge_type"] or "semantic")
            self._outgoing[src.key()].append(edge)
            self._incoming[tgt.key()].append(edge)

    # ------ core graph API ------

    def get_node(self, name: str, file: str) -> SymbolNode | None:
        self._ensure_loaded()
        return self._nodes.get((name, file))

    def all_nodes(self):
        self._ensure_loaded()
        return self._nodes.values()

    def get_outgoing(self, sid: SymbolId, edge_type: str | None = None) -> list[SymbolEdge]:
        self._ensure_loaded()
        edges = self._outgoing.get(sid.key(), [])
        if edge_type:
            return [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_incoming(self, sid: SymbolId, edge_type: str | None = None) -> list[SymbolEdge]:
        self._ensure_loaded()
        edges = self._incoming.get(sid.key(), [])
        if edge_type:
            return [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_neighbors(self, sid: SymbolId, depth: int = 1) -> set[tuple]:
        """Return all neighbor keys within `depth` hops."""
        self._ensure_loaded()
        visited = set()
        frontier = {sid.key()}
        for _ in range(depth):
            next_frontier = set()
            for key in frontier:
                if key in visited:
                    continue
                visited.add(key)
                for e in self._outgoing.get(key, []):
                    next_frontier.add(e.target.key())
                for e in self._incoming.get(key, []):
                    next_frontier.add(e.source.key())
            frontier = next_frontier - visited
        visited.update(frontier)
        visited.discard(sid.key())
        return visited

    def rank_neighbors(self, sid: SymbolId, mode: str = "explore", depth: int = 1, limit: int = 12) -> list[SymbolNode]:
        """Return neighbors ranked by task-aware composite score."""
        neighbor_keys = self.get_neighbors(sid, depth)
        nodes = [self._nodes[k] for k in neighbor_keys if k in self._nodes]
        nodes.sort(key=lambda n: n.composite_score(mode), reverse=True)
        return nodes[:limit]

    def minimal_slice(self, seed_keys: list[tuple], budget: int = 40, mode: str = "explore") -> dict:
        """Extract a minimal subgraph around seed symbols.

        Returns a dict with nodes, edges, entrypoints, risky_paths, summaries.
        """
        self._ensure_loaded()
        included_keys = set(seed_keys)
        # 1-hop expansion from seeds
        for key in list(seed_keys):
            for e in self._outgoing.get(key, []):
                included_keys.add(e.target.key())
            for e in self._incoming.get(key, []):
                included_keys.add(e.source.key())

        # Score and trim to budget
        scored = []
        for key in included_keys:
            node = self._nodes.get(key)
            if node:
                is_seed = key in seed_keys
                score = node.composite_score(mode) + (100 if is_seed else 0)
                scored.append((score, key, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        kept_keys = set()
        for _, key, _ in scored[:budget]:
            kept_keys.add(key)
        # Always keep seeds
        kept_keys.update(k for k in seed_keys if k in self._nodes)

        # Collect edges between kept nodes
        result_edges = []
        for key in kept_keys:
            for e in self._outgoing.get(key, []):
                if e.target.key() in kept_keys:
                    result_edges.append(e)

        # Identify risky paths (nodes with high risk in the slice)
        risky = [self._nodes[k] for k in kept_keys if k in self._nodes and self._nodes[k].risk_score >= 6]
        risky.sort(key=lambda n: n.risk_score, reverse=True)

        result_nodes = [self._nodes[k] for k in kept_keys if k in self._nodes]
        result_nodes.sort(key=lambda n: n.composite_score(mode), reverse=True)

        return {
            "nodes": [n.to_manifest_dict() for n in result_nodes],
            "edges": [e.to_dict() for e in result_edges],
            "entrypoints": [repr(self._nodes[k].sid) for k in seed_keys if k in self._nodes],
            "risky_paths": [{"symbol": n.sid.name, "file": n.sid.file, "risk": n.risk_score} for n in risky[:8]],
            "summaries": {n.sid.name: n.summary or "" for n in result_nodes if n.summary},
        }

    def impact_set(self, changed_keys: list[tuple], radius: int = 2) -> list[SymbolNode]:
        """Return all symbols transitively affected by changes to the given symbols."""
        self._ensure_loaded()
        affected = set()
        frontier = set(changed_keys)
        for _ in range(radius):
            next_frontier = set()
            for key in frontier:
                if key in affected:
                    continue
                affected.add(key)
                for e in self._incoming.get(key, []):
                    next_frontier.add(e.source.key())
            frontier = next_frontier - affected
        affected.update(frontier)
        for k in changed_keys:
            affected.discard(k)
        nodes = [self._nodes[k] for k in affected if k in self._nodes]
        nodes.sort(key=lambda n: n.caller_count + n.risk_score, reverse=True)
        return nodes

    def find_seeds(self, term: str, limit: int = 8, core_only: bool = False) -> list[SymbolNode]:
        """Find seed nodes matching a term, ranked by composite relevance."""
        self._ensure_loaded()
        term_lower = term.lower()
        candidates = []
        for key, node in self._nodes.items():
            name_lower = node.sid.name.lower()
            file_lower = node.sid.file.lower()
            if term_lower in name_lower or term_lower in file_lower:
                if core_only and not _is_core_file(node.sid.file):
                    continue
                score = _symbol_match_score(node.sid.name, node.sid.file, term) + node.composite_score("explore")
                score -= _file_focus_penalty(node.sid.file) * 8
                candidates.append((score, node))
        if not candidates:
            candidate_files = _select_candidate_files(self._conn, term, limit=max(limit, 6), core_only=core_only)
            for node in self._nodes.values():
                if node.sid.file not in candidate_files:
                    continue
                if core_only and not _is_core_file(node.sid.file):
                    continue
                score = _symbol_match_score(node.sid.name, node.sid.file, term) + node.composite_score("explore") + 60
                score -= _file_focus_penalty(node.sid.file) * 8
                candidates.append((score, node))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in candidates[:limit]]

    def focus_manifest(self, term: str, budget: int = 24, core_only: bool = False, mode: str = "explore") -> dict:
        """Generate a machine-oriented ranked manifest for an agent.

        Pipeline: find seeds -> 1-hop expand -> rank -> assemble manifest.
        """
        seeds = self.find_seeds(term, limit=6, core_only=core_only)
        if not seeds:
            fallback_files = _select_candidate_files(self._conn, term, limit=6, core_only=core_only)
            return {"target": term, "suspects": [], "hot_functions": [],
                "risks": [], "constraints": [],
                "recommended_next_steps": ([f"context {fallback_files[0]}"] if fallback_files else []),
                "relevant_files": fallback_files}

        seed_keys = [s.sid.key() for s in seeds]
        # 1-hop neighbors
        all_neighbor_keys = set()
        for key in seed_keys:
            all_neighbor_keys.update(self.get_neighbors(SymbolId(*key), depth=1))
        # Merge seeds + neighbors, score them
        all_keys = set(seed_keys) | all_neighbor_keys
        all_nodes_scored = []
        for key in all_keys:
            node = self._nodes.get(key)
            if node:
                is_seed = key in seed_keys
                score = node.composite_score(mode) + (80 if is_seed else 0)
                all_nodes_scored.append((score, node, is_seed))
        all_nodes_scored.sort(key=lambda x: x[0], reverse=True)
        top_nodes = all_nodes_scored[:budget]

        suspects = [n.to_manifest_dict() for _, n, is_seed in top_nodes if is_seed]
        hot_functions = [
            n.to_manifest_dict() for _, n, _ in top_nodes
            if n.hot_path or n.hotness_score >= 3
        ][:8]

        relevant_files = list(dict.fromkeys(n.sid.file for _, n, _ in top_nodes))[:12]

        risks = []
        for _, n, _ in top_nodes:
            if n.risk_score >= 6:
                risks.append({"symbol": n.sid.name, "file": n.sid.file, "risk_score": n.risk_score,
                              "risk_level": n.risk_level, "ct_sensitive": bool(n.ct_sensitive)})
        risks = risks[:8]

        constraints = []
        for _, n, _ in top_nodes:
            if n.ct_sensitive:
                constraints.append(f"Do not break constant-time behavior in {n.sid.name}")
        # Deduplicate constraints
        constraints = list(dict.fromkeys(constraints))[:6]

        next_steps = []
        if seeds:
            top = seeds[0]
            next_steps.append(f"slice {top.sid.name} {budget * 2}")
            if mode == "bugfix":
                next_steps.append(f"bundle bugfix {top.sid.name}")
            elif mode == "optimize":
                next_steps.append(f"bundle optimize {top.sid.name}")
            elif mode == "audit":
                next_steps.append(f"bundle audit {top.sid.name}")

        return {
            "target": term,
            "mode": mode,
            "suspects": suspects,
            "hot_functions": hot_functions,
            "relevant_files": relevant_files,
            "risks": risks,
            "constraints": constraints,
            "recommended_next_steps": next_steps,
        }

    def slice_manifest(self, term: str, budget: int = 32, core_only: bool = False, mode: str = "explore") -> dict:
        """Generate a strict minimal dependency cut for an agent.

        Returns nodes, edges, entrypoints, risky_paths, summaries — all JSON-ready.
        """
        seeds = self.find_seeds(term, limit=4, core_only=core_only)
        if not seeds:
            return {"target": term, "nodes": [], "edges": [], "entrypoints": [],
                    "risky_paths": [], "summaries": {}}
        seed_keys = [s.sid.key() for s in seeds]
        result = self.minimal_slice(seed_keys, budget=budget, mode=mode)
        result["target"] = term
        result["mode"] = mode

        # Add function signatures for top nodes
        sigs = []
        for nd in result["nodes"][:12]:
            row = self._conn.execute(
                "SELECT signature FROM function_index WHERE function_name = ? AND file = ? LIMIT 1",
                (nd["symbol"], nd["file"])
            ).fetchone()
            if row and row["signature"]:
                sigs.append({"symbol": nd["symbol"], "file": nd["file"], "signature": row["signature"]})
        result["signatures"] = sigs

        # Add direct callers/callees for seeds
        caller_callee = {"callers": [], "callees": []}
        for key in seed_keys:
            sid = SymbolId(*key)
            for e in self.get_incoming(sid, "calls")[:6]:
                caller_callee["callers"].append(e.to_dict())
            for e in self.get_outgoing(sid, "calls")[:6]:
                caller_callee["callees"].append(e.to_dict())
        result["call_context"] = caller_callee

        return result

    def bundle_manifest(self, task_type: str, term: str, budget: int = 40, core_only: bool = False) -> dict:
        """Assemble a task-shaped context pack as a structured manifest."""
        seeds = self.find_seeds(term, limit=6, core_only=core_only)
        if not seeds:
            return {"task": task_type, "target": term, "sections": []}

        seed_keys = [s.sid.key() for s in seeds]
        sections = []

        def add(title, items, max_items=20):
            if items:
                sections.append({"title": title, "items": items[:max_items]})

        # Common: target functions with bodies
        bodies = self._conn.execute(
            "SELECT file, function_name, class_name, start_line, end_line, body, line_count "
            "FROM function_bodies WHERE function_name LIKE ? OR class_name LIKE ? "
            "ORDER BY line_count ASC LIMIT 8",
            (f"%{term}%", f"%{term}%")
        ).fetchall()
        if bodies:
            add("target_functions", [
                {"symbol": _function_label(b["class_name"], b["function_name"]),
                 "file": b["file"], "lines": [b["start_line"], b["end_line"]],
                 "line_count": b["line_count"]}
                for b in bodies
            ])

        # Common: function signatures
        add("signatures", [
            {"symbol": s.sid.name, "file": s.sid.file,
             "signature": (self._conn.execute(
                 "SELECT signature FROM function_index WHERE function_name = ? AND file = ? LIMIT 1",
                 (s.sid.name, s.sid.file)).fetchone() or {"signature": None})["signature"]}
            for s in seeds if (self._conn.execute(
                "SELECT 1 FROM function_index WHERE function_name = ? AND file = ? LIMIT 1",
                (s.sid.name, s.sid.file)).fetchone())
        ])

        # Common: relevant decisions
        decs = self._conn.execute(
            "SELECT decision, rationale, file, function_name FROM decisions "
            "WHERE status = 'active' AND (decision LIKE ? OR file LIKE ? OR function_name LIKE ?) "
            "ORDER BY created_at DESC LIMIT 5",
            (f"%{term}%", f"%{term}%", f"%{term}%")
        ).fetchall()
        if decs:
            add("decisions", [{"decision": d["decision"], "rationale": d["rationale"]} for d in decs])

        # Callers/callees (always useful)
        callers = []
        callees = []
        for key in seed_keys:
            sid = SymbolId(*key)
            callers.extend(e.to_dict() for e in self.get_incoming(sid, "calls")[:8])
            callees.extend(e.to_dict() for e in self.get_outgoing(sid, "calls")[:8])
        add("callers", callers)
        add("callees", callees)

        # Task-specific sections
        if task_type == "bugfix":
            # Top risks
            risks = self._conn.execute(
                "SELECT file, line, risk_type, severity, expression FROM crash_risks "
                "WHERE file LIKE ? ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END LIMIT 10",
                (f"%{term}%",)
            ).fetchall()
            null_risks = self._conn.execute(
                "SELECT file, line, function_call, risk_type FROM null_risks WHERE file LIKE ? LIMIT 8",
                (f"%{term}%",)
            ).fetchall()
            add("risks", [
                {"file": r["file"], "line": r["line"], "type": r["risk_type"],
                 "severity": r["severity"], "expr": r["expression"]} for r in risks
            ] + [
                {"file": n["file"], "line": n["line"], "type": "null",
                 "severity": "medium", "expr": n["function_call"]} for n in null_risks
            ])
            # Recent changes
            churn = self._conn.execute(
                "SELECT file, recent_commit_count, churn_score, bugfix_commits FROM history_metrics "
                "WHERE file LIKE ? ORDER BY churn_score DESC LIMIT 8",
                (f"%{term}%",)
            ).fetchall()
            add("recent_churn", [
                {"file": c["file"], "churn": c["churn_score"],
                 "recent_commits": c["recent_commit_count"], "bugfixes": c["bugfix_commits"]}
                for c in churn
            ])

        elif task_type == "optimize":
            # Bottleneck queue
            bots = self._conn.execute(
                "SELECT symbol_name, file_path, overall_priority, perf_priority, hotness_score, gpu_score, summary "
                "FROM v_bottleneck_queue WHERE symbol_name LIKE ? OR file_path LIKE ? "
                "ORDER BY overall_priority DESC LIMIT 12",
                (f"%{term}%", f"%{term}%")
            ).fetchall()
            add("bottlenecks", [
                {"symbol": b["symbol_name"], "file": b["file_path"],
                 "perf_priority": b["perf_priority"], "hotness": b["hotness_score"],
                 "gpu_score": b["gpu_score"], "summary": b["summary"]}
                for b in bots
            ])
            # GPU candidates in slice
            gpu_nodes = [n for n in seeds if n.gpu_candidate or n.batchable]
            add("gpu_candidates", [n.to_manifest_dict() for n in gpu_nodes])

        elif task_type == "audit":
            # Coverage gaps
            coverage = self._conn.execute(
                "SELECT file, coverage_score, unit_test_refs, crash_risk_count, null_risk_count "
                "FROM audit_coverage WHERE file LIKE ? ORDER BY coverage_score ASC LIMIT 10",
                (f"%{term}%",)
            ).fetchall()
            add("coverage_gaps", [
                {"file": c["file"], "coverage": c["coverage_score"],
                 "tests": c["unit_test_refs"], "crashes": c["crash_risk_count"],
                 "nulls": c["null_risk_count"]}
                for c in coverage
            ])
            # CT-sensitive symbols
            ct = [n for n in seeds if n.ct_sensitive]
            add("ct_sensitive_symbols", [n.to_manifest_dict() for n in ct])
            # Unsafe patterns
            casts = self._conn.execute(
                "SELECT file, line, cast_expr FROM unsafe_casts WHERE file LIKE ? LIMIT 8",
                (f"%{term}%",)
            ).fetchall()
            raw = self._conn.execute(
                "SELECT file, line, member_type, member_name FROM raw_pointers WHERE file LIKE ? LIMIT 8",
                (f"%{term}%",)
            ).fetchall()
            add("unsafe_patterns", [
                {"file": c["file"], "line": c["line"], "type": "cast", "expr": c["cast_expr"]} for c in casts
            ] + [
                {"file": r["file"], "line": r["line"], "type": "raw_ptr",
                 "expr": f"{r['member_type']} {r['member_name']}"} for r in raw
            ])

        # Negative context (what NOT to touch)
        neg = []
        for s in seeds:
            if s.ct_sensitive:
                neg.append(f"Preserve constant-time behavior in {s.sid.name}")
        # Add global constraints from decisions table
        global_constraints = self._conn.execute(
            "SELECT decision FROM decisions WHERE status = 'active' AND tags LIKE '%constraint%' LIMIT 5"
        ).fetchall()
        for gc in global_constraints:
            neg.append(gc["decision"])
        if neg:
            add("negative_context", neg[:8])

        # Tests
        tests = self._conn.execute(
            "SELECT test_file, function_name, mapping_type FROM test_function_map "
            "WHERE target_file LIKE ? OR function_name LIKE ? LIMIT 10",
            (f"%{term}%", f"%{term}%")
        ).fetchall()
        add("related_tests", [
            {"test": t["test_file"], "function": t["function_name"], "type": t["mapping_type"]}
            for t in tests
        ])

        return {
            "task": task_type,
            "target": term,
            "seed_count": len(seeds),
            "sections": sections,
        }

    def pipeline_manifest(self, task_type: str, term: str, budget: int = 48,
                          core_only: bool = False) -> dict:
        """Single compound query: focus -> slice -> bundle in one call.

        Chains progressively: broad discovery -> minimal dependency cut ->
        task-shaped context pack.  Returns ONE combined JSON response so the
        calling agent needs exactly one CLI invocation instead of three,
        saving both tokens and rate-limit capacity.
        """
        mode = {"bugfix": "bugfix", "optimize": "optimize",
                "audit": "audit"}.get(task_type, "explore")

        # Phase 1: Focus — discover seeds and ranked suspects
        focus = self.focus_manifest(term, budget=budget, core_only=core_only, mode=mode)
        if not focus.get("suspects"):
            return {"task": task_type, "target": term, "phases": {
                "focus": focus, "slice": {}, "bundle": {}},
                "summary": {"seed_count": 0, "files": [], "risks": []}}

        # Phase 2: Slice — narrow to minimal dependency subgraph
        # Use top suspect names as more precise slice targets
        top_symbols = [s["symbol"] for s in focus["suspects"][:4]]
        slice_term = top_symbols[0] if top_symbols else term
        slice_budget = max(budget, 32)
        slc = self.slice_manifest(slice_term, budget=slice_budget,
                                  core_only=core_only, mode=mode)

        # Phase 3: Bundle — task-shaped context with function bodies
        bundle_budget = max(budget // 2, 24)
        bndl = self.bundle_manifest(task_type, term, budget=bundle_budget,
                                    core_only=core_only)

        # Merge unique files across all phases
        all_files = list(dict.fromkeys(
            focus.get("relevant_files", []) +
            [n["file"] for n in slc.get("nodes", [])] +
            [item["file"] for sec in bndl.get("sections", [])
             for item in (sec.get("items") or []) if isinstance(item, dict) and "file" in item]
        ))

        # Merge unique risks
        seen_risk_keys = set()
        merged_risks = []
        for r in focus.get("risks", []) + slc.get("risky_paths", []):
            key = (r.get("symbol", r.get("source", "")), r.get("file", ""))
            if key not in seen_risk_keys:
                seen_risk_keys.add(key)
                merged_risks.append(r)

        # Collect constraints and negative context from all phases
        constraints = focus.get("constraints", [])
        for sec in bndl.get("sections", []):
            if sec.get("title") == "negative_context":
                constraints.extend(sec.get("items", []))
        constraints = list(dict.fromkeys(constraints))[:8]

        # Compact summary for quick agent orientation
        summary = {
            "seed_count": len(focus.get("suspects", [])),
            "slice_nodes": len(slc.get("nodes", [])),
            "slice_edges": len(slc.get("edges", [])),
            "bundle_sections": len(bndl.get("sections", [])),
            "files": all_files[:16],
            "risks": merged_risks[:10],
            "constraints": constraints,
            "top_suspects": [s["symbol"] for s in focus.get("suspects", [])[:6]],
        }

        return {
            "task": task_type,
            "target": term,
            "mode": mode,
            "phases": {
                "focus": focus,
                "slice": slc,
                "bundle": bndl,
            },
            "summary": summary,
        }


# Cache for the graph view so repeated commands don't rebuild it
_graph_view_cache = None

def get_graph_view() -> SymbolGraphView:
    """Return (and cache) the SymbolGraphView singleton."""
    global _graph_view_cache
    if _graph_view_cache is None:
        _graph_view_cache = SymbolGraphView(get_conn())
    return _graph_view_cache


# ============================================================
# QUERY COMMANDS
# ============================================================

def get_conn():
    if not DB_PATH.exists():
        print("[!] Database not found. Run: python source_graph.py build")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def print_table(rows, columns):
    """Simple formatted table output."""
    if not rows:
        print("  (no results)")
        return
    widths = [len(c) for c in columns]
    for row in rows:
        for i, col in enumerate(columns):
            val = str(row[col] or "")
            widths[i] = max(widths[i], len(val))

    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    sep = "-+-".join("-" * widths[i] for i in range(len(columns)))
    print(f"  {header}")
    print(f"  {sep}")
    for row in rows:
        line = " | ".join(str(row[col] or "").ljust(widths[i]) for i, col in enumerate(columns))
        print(f"  {line}")
    print(f"\n  ({len(rows)} results)")


def _compact_text(value, limit=96):
    text = str(value or "").strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _print_compact_section(title, items):
    if not items:
        return
    if not _QUIET:
        print(f"\n  === {title} ===")
    for item in items:
        print(f"  - {item}")


def _parse_budget_arg(raw_value, default_value):
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return default_value
    return max(20, min(parsed, 400))


def _file_snapshot_row(conn, file_name):
    coverage = conn.execute(
        "SELECT coverage_score, crash_risk_count, null_risk_count FROM audit_coverage WHERE file = ?",
        (file_name,),
    ).fetchone()
    history = conn.execute(
        "SELECT churn_score, recent_commit_count FROM history_metrics WHERE file = ?",
        (file_name,),
    ).fetchone()
    tested = conn.execute(
        "SELECT COUNT(DISTINCT test_file) FROM test_function_map WHERE target_file = ?",
        (file_name,),
    ).fetchone()[0]
    callers = conn.execute(
        "SELECT COUNT(*) FROM call_edges WHERE callee_file = ?",
        (file_name,),
    ).fetchone()[0]
    symbols = conn.execute(
        "SELECT COUNT(*) FROM symbol_metadata WHERE file_path = ?",
        (file_name,),
    ).fetchone()[0]
    return {
        "file": file_name,
        "coverage_score": coverage["coverage_score"] if coverage else 0,
        "crash_risks": coverage["crash_risk_count"] if coverage else 0,
        "null_risks": coverage["null_risk_count"] if coverage else 0,
        "churn": history["churn_score"] if history else 0,
        "recent_commits": history["recent_commit_count"] if history else 0,
        "tests": tested,
        "callers": callers,
        "symbols": symbols,
    }


def _compact_file_snapshot(conn, file_name):
    row = _file_snapshot_row(conn, file_name)
    return (
        f"{row['file']} | cov={row['coverage_score']} callers={row['callers']} "
        f"tests={row['tests']} churn={row['churn']} risks={row['crash_risks'] + row['null_risks']} "
        f"symbols={row['symbols']}"
    )


def _iter_candidate_symbols(conn, term, candidate_files, limit=16):
    pattern = f"%{term}%"
    params = [pattern, pattern, pattern, pattern]
    file_clause = ""
    if candidate_files:
        placeholders = _sql_in(candidate_files)
        file_clause = f" OR file_path IN ({placeholders})"
        params.extend(candidate_files)
    return conn.execute(
        "SELECT symbol_name, file_path, semantic_tags, hot_path, ct_sensitive, batchable, gpu_candidate, "
        "review_priority, risk_score, gain_score, audit_coverage_score "
        "FROM symbol_metadata WHERE symbol_name LIKE ? OR file_path LIKE ? OR semantic_tags LIKE ? OR summary LIKE ?"
        f"{file_clause} ORDER BY "
        "CASE "
        "WHEN symbol_name LIKE ? THEN 0 "
        "WHEN file_path LIKE ? THEN 1 "
        "WHEN semantic_tags LIKE ? THEN 2 "
        "WHEN summary LIKE ? THEN 3 "
        "ELSE 4 END, "
        "review_priority DESC, gain_score DESC, symbol_name "
        f"LIMIT {limit}",
        params + [pattern, pattern, pattern, pattern],
    ).fetchall()


def _build_neighbor_file_scores(conn, candidate_files, limit=12):
    if not candidate_files:
        return []
    placeholders = _sql_in(candidate_files)
    dep_rows = conn.execute(
        f"SELECT source_file, target_file FROM dependencies WHERE source_file IN ({placeholders}) OR target_file IN ({placeholders})",
        candidate_files + candidate_files,
    ).fetchall()
    call_rows = conn.execute(
        f"SELECT caller_file, callee_file, call_count FROM call_edges WHERE caller_file IN ({placeholders}) OR callee_file IN ({placeholders})",
        candidate_files + candidate_files,
    ).fetchall()

    scores = defaultdict(lambda: {"dep_edges": 0, "call_edges": 0, "call_weight": 0})
    candidate_set = set(candidate_files)

    for row in dep_rows:
        for name in (row["source_file"], row["target_file"]):
            if name and name not in candidate_set:
                scores[name]["dep_edges"] += 1

    for row in call_rows:
        for name in (row["caller_file"], row["callee_file"]):
            if name and name not in candidate_set:
                scores[name]["call_edges"] += 1
                scores[name]["call_weight"] += row["call_count"] or 0

    ranked = sorted(
        scores.items(),
        key=lambda item: (item[1]["call_weight"], item[1]["call_edges"], item[1]["dep_edges"], item[0]),
        reverse=True,
    )
    return ranked[:limit]


def _focus_term_variants(term):
    raw = (term or "").strip().replace("\\", "/")
    lowered = raw.lower()
    basename = lowered.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    path_parts = [part for part in lowered.split("/") if part]
    token_pool = set()
    for value in (lowered, basename, stem):
        for token in re.split(r"[^a-z0-9]+", value):
            if token:
                token_pool.add(token)
    return {
        "raw": raw,
        "lowered": lowered,
        "basename": basename,
        "stem": stem,
        "path_parts": path_parts,
        "tokens": sorted(token_pool),
    }


def _file_match_rank(file_name, term):
    if not file_name:
        return 100
    variants = _focus_term_variants(term)
    file_lower = file_name.lower()
    basename = os.path.basename(file_lower)
    stem = basename.rsplit(".", 1)[0]
    path_parts = [part for part in file_lower.split("/") if part]
    file_tokens = set(token for token in re.split(r"[^a-z0-9]+", file_lower) if token)

    if variants["lowered"] and file_lower == variants["lowered"]:
        return 0
    if variants["basename"] and basename == variants["basename"]:
        return 1
    if variants["stem"] and stem == variants["stem"]:
        return 2
    if variants["lowered"] and (file_lower.endswith("/" + variants["lowered"]) or file_lower.startswith(variants["lowered"] + "/")):
        return 3
    if variants["basename"] and variants["basename"] in path_parts:
        return 4
    if variants["path_parts"] and len(variants["path_parts"]) > 1:
        matched_parts = sum(1 for part in variants["path_parts"] if part in path_parts)
        if matched_parts == len(variants["path_parts"]):
            return 5
        if matched_parts:
            return 6
    if variants["tokens"]:
        matched_tokens = sum(1 for token in variants["tokens"] if token in file_tokens)
        if matched_tokens == len(variants["tokens"]):
            return 7
        if matched_tokens:
            return 8
    if variants["lowered"] and variants["lowered"] in file_lower:
        return 9
    return 10


def _symbol_match_score(symbol_name, file_name, term):
    variants = _focus_term_variants(term)
    symbol_lower = (symbol_name or "").lower()
    file_lower = (file_name or "").lower()
    basename = os.path.basename(file_lower)
    score = 0

    if variants["lowered"]:
        if symbol_lower == variants["lowered"]:
            score += 160
        elif symbol_lower.startswith(variants["lowered"]) or symbol_lower.endswith(variants["lowered"]):
            score += 90
        elif variants["lowered"] in symbol_lower:
            score += 35

        if file_lower == variants["lowered"]:
            score += 180
        elif basename == variants["basename"]:
            score += 140
        elif file_lower.endswith("/" + variants["lowered"]):
            score += 120
        elif variants["lowered"] in file_lower:
            score += 40

    symbol_tokens = set(token for token in re.split(r"[^a-z0-9]+", symbol_lower) if token)
    file_tokens = set(token for token in re.split(r"[^a-z0-9]+", file_lower) if token)
    shared_symbol = sum(1 for token in variants["tokens"] if token in symbol_tokens)
    shared_file = sum(1 for token in variants["tokens"] if token in file_tokens)
    score += shared_symbol * 20
    score += shared_file * 12
    return score


def _file_focus_penalty(file_name):
    if not file_name:
        return 100
    lowered = file_name.lower()
    if lowered.startswith(("build/", "build_", "build-", "cmakefiles/", "testing/")):
        return 40
    if "/cmakefiles/" in lowered or lowered.endswith(("/compile_commands.json", ".pb.cc", ".pb.h", ".generated.h", ".generated.cpp")):
        return 35
    if lowered.startswith(("dist/", "out/", "node_modules/", ".git/")):
        return 30
    if file_name.startswith(("src/", "include/", "ufsecp/", "cpu/", "gpu/", "opencl/", "compat/")):
        return 0
    if file_name.startswith(("audit/", "scripts/", "tools/", ".github/", "cmake/")):
        return 1
    if file_name.startswith(("apps/", "examples/", "benchmarks/")):
        return 1
    if file_name.startswith(("tests/", "test_", "docs/")):
        return 2
    if file_name.startswith(("shaders/", "esp32_test/")):
        return 4
    return 2


def _is_core_file(file_name):
    if not file_name:
        return False
    return file_name.startswith((
        "src/", "include/", "ufsecp/", "cpu/", "gpu/", "opencl/", "compat/",
        "audit/", "tests/", "docs/", "scripts/", "tools/", ".github/", "cmake/"
    ))


def _select_candidate_files(conn, term, limit=12, core_only=False):
    candidate_files = _collect_candidate_files(conn, term, limit=max(limit * 2, 12 if core_only else limit))
    if any(_file_focus_penalty(file_name) < 30 for file_name in candidate_files):
        candidate_files = [file_name for file_name in candidate_files if _file_focus_penalty(file_name) < 30]
    if not core_only:
        return candidate_files[:limit]
    core_files = [file_name for file_name in candidate_files if _is_core_file(file_name)]
    return (core_files or candidate_files)[:limit]


def _parse_compact_args(argv, default_budget):
    args = [arg for arg in argv if arg]
    core_only = False
    json_mode = False
    mode = "explore"
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "--core":
            core_only = True
        elif args[i] == "--json":
            json_mode = True
        elif args[i] == "--mode" and i + 1 < len(args):
            i += 1
            mode = args[i]
        else:
            filtered.append(args[i])
        i += 1
    budget = default_budget
    if filtered and filtered[-1].isdigit():
        budget = filtered.pop()
    return " ".join(filtered), budget, core_only, json_mode, mode


def _sql_in(values):
    return ",".join("?" for _ in values)


def _collect_candidate_files(conn, term, limit=12):
    pattern = f"%{term}%"
    queries = [
        ("SELECT DISTINCT path AS file FROM files WHERE path LIKE ? LIMIT 12", (pattern,)),
        ("SELECT DISTINCT file FROM file_summaries WHERE file LIKE ? OR summary LIKE ? OR category LIKE ? LIMIT 12", (pattern, pattern, pattern)),
        ("SELECT DISTINCT file FROM semantic_tags WHERE tag LIKE ? OR entity_name LIKE ? OR evidence LIKE ? OR file LIKE ? LIMIT 12", (pattern, pattern, pattern, pattern)),
        ("SELECT DISTINCT file_path AS file FROM symbol_metadata WHERE symbol_name LIKE ? OR file_path LIKE ? OR semantic_tags LIKE ? OR summary LIKE ? LIMIT 12", (pattern, pattern, pattern, pattern)),
        ("SELECT DISTINCT file FROM function_index WHERE function_name LIKE ? OR class_name LIKE ? OR file LIKE ? LIMIT 12", (pattern, pattern, pattern)),
        ("SELECT DISTINCT cpp_file AS file FROM classes WHERE name LIKE ? OR cpp_file LIKE ? LIMIT 8", (pattern, pattern)),
        ("SELECT DISTINCT header AS file FROM classes WHERE name LIKE ? OR header LIKE ? LIMIT 8", (pattern, pattern)),
        ("SELECT DISTINCT source_file AS file FROM packet_handlers WHERE headcode_name LIKE ? OR handler_method LIKE ? OR description LIKE ? LIMIT 8", (pattern, pattern, pattern)),
        ("SELECT DISTINCT source_file AS file FROM config_keys WHERE key_name LIKE ? OR source_file LIKE ? LIMIT 8", (pattern, pattern)),
        ("SELECT DISTINCT target_file AS file FROM test_function_map WHERE target_file LIKE ? OR function_name LIKE ? OR class_name LIKE ? LIMIT 8", (pattern, pattern, pattern)),
        ("SELECT DISTINCT caller_file AS file FROM call_edges WHERE caller_symbol LIKE ? OR callee_symbol LIKE ? OR caller_file LIKE ? OR callee_file LIKE ? LIMIT 8", (pattern, pattern, pattern, pattern)),
        ("SELECT DISTINCT callee_file AS file FROM call_edges WHERE caller_symbol LIKE ? OR callee_symbol LIKE ? OR caller_file LIKE ? OR callee_file LIKE ? LIMIT 8", (pattern, pattern, pattern, pattern)),
    ]
    results = []
    seen = set()
    for query_rank, (query, params) in enumerate(queries):
        for row in conn.execute(query, params).fetchall():
            file_name = row["file"]
            if not file_name or file_name in seen:
                continue
            seen.add(file_name)
            match_rank = _file_match_rank(file_name, term)
            has_directory = 0 if "/" in file_name else 1
            results.append((match_rank, query_rank, _file_focus_penalty(file_name), has_directory, len(file_name), file_name))
    ranked = sorted(results, key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]))
    ranked_files = [file_name for _, _, _, _, _, file_name in ranked]
    indexed_paths = set()
    if ranked_files:
        placeholders = _sql_in(ranked_files)
        indexed_rows = conn.execute(
            f"SELECT path FROM files WHERE path IN ({placeholders})",
            ranked_files,
        ).fetchall()
        indexed_paths = {row["path"] for row in indexed_rows}
    directory_basenames = {
        os.path.basename(file_name)
        for file_name in ranked_files
        if "/" in file_name
    }
    selected = []
    for file_name in ranked_files:
        basename = os.path.basename(file_name)
        if "/" not in file_name and basename in directory_basenames and file_name not in indexed_paths:
            continue
        selected.append(file_name)
        if len(selected) >= limit:
            break
    return selected


def where_cmd(name):
    """Ultra-compact symbol definition lookup. Returns file+line in JSON.
    Usage: where <name>  (also available as: def <name>)
    Output: {"sym":"name","file":"path","line":N,"end":N} or list of matches.
    """
    import json as _json
    conn = get_conn()
    rows = conn.execute(
        "SELECT function_name, file, start_line, end_line "
        "FROM function_index WHERE function_name LIKE ? ORDER BY function_name LIMIT 20",
        (f"%{name}%",)
    ).fetchall()
    if not rows:
        rows2 = conn.execute(
            "SELECT path FROM files WHERE path LIKE ? LIMIT 10",
            (f"%{name}%",)
        ).fetchall()
        if not rows2:
            print(_json.dumps({"error": f"not found: {name}"}, separators=(",",":")))
            return
        results = [{"file": r["path"]} for r in rows2]
    else:
        results = [{"sym": r["function_name"], "file": r["file"],
                    "line": r["start_line"], "end": r["end_line"]} for r in rows]
    if len(results) == 1:
        print(_json.dumps(results[0], separators=(",",":")))
    else:
        print(_json.dumps(results, separators=(",",":")))


def _research_matches(conn, term, limit=20):
    pattern = f"%{term}%"
    return conn.execute(
        "SELECT asset_path, symbol, mention_type, context FROM research_mentions "
        "WHERE symbol LIKE ? OR context LIKE ? OR asset_path LIKE ? "
        "ORDER BY CASE mention_type WHEN 'protocol' THEN 0 WHEN 'symbol' THEN 1 ELSE 2 END, symbol "
        "LIMIT ?",
        (pattern, pattern, pattern, limit),
    ).fetchall()


def _fts_safe_query(term):
    """Convert a user term into a conservative FTS5 query.

    Bare FTS input treats punctuation like '.', '/', '-', and ':' as syntax in
    some positions, which breaks filename and path searches such as
    'SECURITY.md' or 'docs/ROADMAP.md'. Quote any token containing non-word
    characters so literal filename/path lookups stay inside the graph path.
    """
    tokens = (term or "").split()
    if not tokens:
        return '""'

    safe_tokens = []
    for token in tokens:
        if re.fullmatch(r"[A-Za-z0-9_]+", token):
            safe_tokens.append(token)
            continue
        safe_tokens.append('"' + token.replace('"', '""') + '"')
    return " ".join(safe_tokens)


def _ai_file_matches(conn, term, limit=30):
    """Prioritize exact/basename/path-hint file matches before generic FTS.

    The AI commonly searches by dotted filenames or source-dir-qualified paths
    like 'docs/README.md'. The graph stores paths relative to each source dir,
    so use the files table plus a small scorer to recover the intended repo file
    before falling back to noisier FTS results.
    """
    raw = (term or "").strip()
    if not raw:
        return []

    needle = raw.lower()
    basename = needle.rsplit("/", 1)[-1]
    project_hint = needle.split("/", 1)[0] if "/" in needle else None

    like_terms = [
        f"%{needle}%",
        f"%{basename}%",
    ]

    rows = conn.execute(
        "SELECT path, project, category, description FROM files "
        "WHERE lower(path) LIKE ? OR lower(path) LIKE ?",
        tuple(like_terms),
    ).fetchall()

    scored = []
    for path, project, category, description in rows:
        path_l = (path or "").lower()
        project_l = (project or "").lower()
        score = 100

        if path_l == needle:
            score = 0
        elif project_hint and project_l == project_hint and path_l == basename:
            score = 1
        elif path_l == basename:
            score = 2
        elif path_l.endswith("/" + basename):
            score = 3
        elif needle in path_l:
            score = 4
        elif basename in path_l:
            score = 5

        noise_penalty = 10 if "node_modules/" in path_l or path_l.startswith("build") else 0
        scored.append((score + noise_penalty, len(path_l), path, project, category, description))

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    results = []
    seen = set()
    for _score, _path_len, path, _project, category, description in scored:
        if path in seen:
            continue
        seen.add(path)
        results.append(("file", path, path, category or "", description or ""))
        if len(results) >= limit:
            break
    return results


def find_cmd(term):
    """Search everything using FTS."""
    # FTS5 needs simple conn without row_factory for proper fetch
    conn2 = sqlite3.connect(str(DB_PATH))
    rows = _ai_file_matches(conn2, term)
    query = _fts_safe_query(term)
    if len(rows) < 30:
        try:
            fts_rows = conn2.execute(
                "SELECT entity_type, name, file, category, description FROM fts_index WHERE fts_index MATCH ? ORDER BY rank LIMIT 30",
                (query,)
            ).fetchall()
        except sqlite3.OperationalError:
            fts_rows = []
        seen = {(row[0], row[1], row[2]) for row in rows}
        for row in fts_rows:
            key = (row[0], row[1], row[2])
            if key in seen:
                continue
            rows.append(row)
            seen.add(key)
            if len(rows) >= 30:
                break
    if not rows:
        # Fallback to LIKE search
        pattern = f"%{term}%"
        rows = conn2.execute(
            "SELECT entity_type, name, file, category, description FROM fts_index WHERE name LIKE ? OR description LIKE ? OR file LIKE ? LIMIT 30",
            (pattern, pattern, pattern)
        ).fetchall()
    # Fallback: also search function bodies for string literals
    if len(rows) < 30:
        body_pattern = f"%{term}%"
        try:
            body_rows = conn2.execute(
                "SELECT 'body' AS entity_type, function_name AS name, file, project AS category, "
                "'function body match' AS description "
                "FROM function_bodies WHERE body LIKE ? "
                "ORDER BY file, start_line LIMIT 15",
                (body_pattern,)
            ).fetchall()
            seen_bodies = set()
            for br in body_rows:
                key = (br[0], br[1], br[2])
                if key not in seen_bodies:
                    rows.append(br)
                    seen_bodies.add(key)
                    if len(rows) >= 30:
                        break
        except sqlite3.OperationalError:
            pass
    conn2.close()
    cols = ["entity_type", "name", "file", "category", "description"]
    print(f"\n  Search: '{term}'")
    if not rows:
        print("  (no results)")
        return
    # Manual table print for tuple rows
    widths = [len(c) for c in cols]
    for row in rows:
        for i in range(len(cols)):
            widths[i] = max(widths[i], len(str(row[i] or "")))
    header = " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols)))
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
    print(f"  {header}")
    print(f"  {sep}")
    for row in rows:
        line = " | ".join(str(row[i] or "").ljust(widths[i]) for i in range(len(cols)))
        print(f"  {line}")
    print(f"\n  ({len(rows)} results)")


def singleton_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT macro, class_name, header, project, category, description FROM singletons WHERE macro LIKE ? OR class_name LIKE ?",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  Singletons matching '{name}':")
    print_table(rows, ["macro", "class_name", "header", "project", "category", "description"])


def file_cmd(pattern):
    conn = get_conn()
    like = f"%{pattern}%"
    rows = conn.execute(
        "SELECT path, project, category FROM files WHERE path LIKE ?",
        (like,)
    ).fetchall()
    print(f"\n  Files matching '{pattern}':")
    print_table(rows, ["path", "project", "category"])


def handler_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT headcode_name, headcode_value, headcode_hex, handler_method, source_file, handler_type, description FROM packet_handlers WHERE headcode_name LIKE ? OR handler_method LIKE ? OR description LIKE ?",
        (pattern, pattern, pattern)
    ).fetchall()
    print(f"\n  Handlers matching '{name}':")
    print_table(rows, ["headcode_name", "headcode_hex", "handler_method", "source_file", "handler_type", "description"])


def event_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT name, singleton_macro, cpp_file, header_file, def_file, ai_file, description FROM events WHERE name LIKE ? OR description LIKE ?",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  Events matching '{name}':")
    print_table(rows, ["name", "singleton_macro", "cpp_file", "header_file", "def_file", "ai_file", "description"])


def method_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT class_name, method_name, file, category, description FROM methods WHERE method_name LIKE ? OR description LIKE ?",
        (pattern, pattern)
    ).fetchall()
    if not rows:
        # Try player_files key_methods
        rows = conn.execute(
            "SELECT 'Player' as class_name, domain as method_name, file, 'player' as category, key_methods as description FROM player_files WHERE key_methods LIKE ? OR domain LIKE ?",
            (pattern, pattern)
        ).fetchall()
    print(f"\n  Methods matching '{name}':")
    print_table(rows, ["class_name", "method_name", "file", "category", "description"])


def player_cmd(domain):
    conn = get_conn()
    pattern = f"%{domain}%"
    rows = conn.execute(
        "SELECT file, domain, key_methods, description FROM player_files WHERE domain LIKE ? OR description LIKE ? OR key_methods LIKE ?",
        (pattern, pattern, pattern)
    ).fetchall()
    print(f"\n  Player files matching '{domain}':")
    print_table(rows, ["file", "domain", "key_methods", "description"])


def ai_cmd(target):
    conn = get_conn()
    pattern = f"%{target}%"
    rows = conn.execute(
        "SELECT file, target, description FROM ai_handlers WHERE file LIKE ? OR target LIKE ? OR description LIKE ?",
        (pattern, pattern, pattern)
    ).fetchall()
    print(f"\n  AI handlers matching '{target}':")
    print_table(rows, ["file", "target", "description"])


def const_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT name, value, header, category, description FROM constants WHERE name LIKE ? OR description LIKE ?",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  Constants matching '{name}':")
    print_table(rows, ["name", "value", "header", "category", "description"])


def config_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT path, description FROM config_files WHERE path LIKE ? OR description LIKE ?",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  Config files matching '{name}':")
    print_table(rows, ["path", "description"])


def deps_cmd(cls):
    conn = get_conn()
    pattern = f"%{cls}%"
    print(f"\n  === Files that USE '{cls}' ===")
    rows = conn.execute(
        "SELECT source_file, target_file, dep_type FROM dependencies WHERE target_file LIKE ? ORDER BY dep_type, source_file",
        (pattern,)
    ).fetchall()
    print_table(rows, ["source_file", "target_file", "dep_type"])

    print(f"\n  === Files that '{cls}' DEPENDS ON ===")
    rows2 = conn.execute(
        "SELECT source_file, target_file, dep_type FROM dependencies WHERE source_file LIKE ? ORDER BY dep_type, target_file",
        (pattern,)
    ).fetchall()
    print_table(rows2, ["source_file", "target_file", "dep_type"])


def enum_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT name, file, line, value_count, values_preview, category FROM enums WHERE name LIKE ? OR values_preview LIKE ?",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  Enums matching '{name}':")
    print_table(rows, ["name", "file", "line", "value_count", "values_preview", "category"])


def struct_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT name, file, line, field_count, is_packed, fields_preview, category FROM structs WHERE name LIKE ? OR fields_preview LIKE ?",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  Structs matching '{name}':")
    print_table(rows, ["name", "file", "line", "field_count", "is_packed", "fields_preview", "category"])


def class_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT name, header, cpp_file, parent_class, project FROM classes WHERE name LIKE ? OR parent_class LIKE ?",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  Classes matching '{name}':")
    print_table(rows, ["name", "header", "cpp_file", "parent_class", "project"])
    for row in rows:
        class_name = row["name"]
        children = conn.execute(
            "SELECT name, header FROM classes WHERE parent_class = ?",
            (class_name,)
        ).fetchall()
        if children:
            print(f"\n  Children of {class_name}:")
            print_table(children, ["name", "header"])


def todo_cmd(filter_text=None):
    conn = get_conn()
    if filter_text:
        pattern = f"%{filter_text}%"
        rows = conn.execute(
            "SELECT file, line, todo_type, text, project FROM todos WHERE text LIKE ? OR file LIKE ? OR todo_type LIKE ? ORDER BY todo_type, file, line",
            (pattern, pattern, pattern)
        ).fetchall()
        print(f"\n  TODOs matching '{filter_text}':")
        print_table(rows, ["file", "line", "todo_type", "text", "project"])
    else:
        print("\n  === TODO/FIXME Summary ===")
        summary = conn.execute(
            "SELECT todo_type, COUNT(*) as cnt FROM todos GROUP BY todo_type ORDER BY cnt DESC"
        ).fetchall()
        print_table(summary, ["todo_type", "cnt"])
        print("\n  === Top 20 files with most TODOs ===")
        rows = conn.execute(
            "SELECT file, COUNT(*) as cnt, GROUP_CONCAT(DISTINCT todo_type) as types FROM todos GROUP BY file ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        print_table(rows, ["file", "cnt", "types"])


def dbtable_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT table_name, source_file, line, query_type, context FROM db_tables WHERE table_name LIKE ? ORDER BY table_name, source_file",
        (pattern,)
    ).fetchall()
    print(f"\n  Database table refs matching '{name}':")
    print_table(rows, ["table_name", "source_file", "line", "query_type", "context"])


def leaks_cmd():
    conn = get_conn()
    print("\n  === Memory Leak Risk Analysis (Top 30) ===")
    rows = conn.execute(
        "SELECT file, project, new_count, delete_count, make_unique_count, make_shared_count, risk_score FROM leak_risks ORDER BY risk_score DESC LIMIT 30"
    ).fetchall()
    print_table(rows, ["file", "project", "new_count", "delete_count", "make_unique_count", "make_shared_count", "risk_score"])
    total_risk = conn.execute("SELECT SUM(risk_score), COUNT(*) FROM leak_risks WHERE risk_score > 0").fetchone()
    if total_risk[0]:
        print(f"\n  Total risk score: {total_risk[0]} across {total_risk[1]} files")
        print("  (risk = new - delete - smart_ptr; higher = more likely leak)")


def nullrisks_cmd(file_filter=None):
    conn = get_conn()
    if file_filter:
        pattern = f"%{file_filter}%"
        rows = conn.execute(
            "SELECT file, line, function_call, pointer_var, risk_type, context FROM null_risks WHERE file LIKE ? ORDER BY file, line",
            (pattern,)
        ).fetchall()
        print(f"\n  NULL-check risks in '{file_filter}':")
    else:
        print("\n  === NULL-Check Risk Summary ===")
        summary = conn.execute(
            "SELECT function_call, risk_type, COUNT(*) as cnt FROM null_risks GROUP BY function_call, risk_type ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        print_table(summary, ["function_call", "risk_type", "cnt"])
        print("\n  === Top 20 Files with Most NULL Risks ===")
        rows = conn.execute(
            "SELECT file, COUNT(*) as cnt FROM null_risks GROUP BY file ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        print_table(rows, ["file", "cnt"])
        return
    print_table(rows, ["file", "line", "function_call", "pointer_var", "risk_type", "context"])


def rawptrs_cmd(file_filter=None):
    conn = get_conn()
    if file_filter:
        pattern = f"%{file_filter}%"
        rows = conn.execute(
            "SELECT file, line, class_name, member_type, member_name FROM raw_pointers WHERE file LIKE ? OR class_name LIKE ? ORDER BY file, line",
            (pattern, pattern)
        ).fetchall()
        print(f"\n  Raw pointer members in '{file_filter}':")
    else:
        print("\n  === Raw Pointer Members Summary ===")
        rows = conn.execute(
            "SELECT class_name, COUNT(*) as cnt FROM raw_pointers GROUP BY class_name ORDER BY cnt DESC LIMIT 25"
        ).fetchall()
        print_table(rows, ["class_name", "cnt"])
        total = conn.execute("SELECT COUNT(*) FROM raw_pointers").fetchone()[0]
        print(f"\n  Total: {total} raw pointer members (RAII migration candidates)")
        return
    print_table(rows, ["file", "line", "class_name", "member_type", "member_name"])


def casts_cmd(file_filter=None):
    conn = get_conn()
    if file_filter:
        pattern = f"%{file_filter}%"
        rows = conn.execute(
            "SELECT file, line, cast_expr, context FROM unsafe_casts WHERE file LIKE ? ORDER BY file, line",
            (pattern,)
        ).fetchall()
        print(f"\n  C-style casts in '{file_filter}':")
    else:
        print("\n  === C-Style Cast Summary ===")
        rows = conn.execute(
            "SELECT file, COUNT(*) as cnt FROM unsafe_casts GROUP BY file ORDER BY cnt DESC LIMIT 25"
        ).fetchall()
        print_table(rows, ["file", "cnt"])
        total = conn.execute("SELECT COUNT(*) FROM unsafe_casts").fetchone()[0]
        print(f"\n  Total: {total} C-style casts (static_cast migration candidates)")
        return
    print_table(rows, ["file", "line", "cast_expr", "context"])


def crashes_cmd(file_filter=None):
    conn = get_conn()
    if file_filter:
        pattern = f"%{file_filter}%"
        rows = conn.execute(
            "SELECT file, line, risk_type, severity, expression, context FROM crash_risks WHERE file LIKE ? ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END, file, line",
            (pattern,)
        ).fetchall()
        print(f"\n  Crash risks in '{file_filter}':")
    else:
        print("\n  === Crash Risk Pattern Summary ===")
        summary = conn.execute(
            "SELECT risk_type, severity, COUNT(*) as cnt FROM crash_risks GROUP BY risk_type, severity ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END, cnt DESC"
        ).fetchall()
        print_table(summary, ["risk_type", "severity", "cnt"])
        print("\n  === Top 20 Files with Most Crash Risks ===")
        by_file = conn.execute(
            "SELECT file, COUNT(*) as cnt, SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) as critical FROM crash_risks GROUP BY file ORDER BY critical DESC, cnt DESC LIMIT 20"
        ).fetchall()
        print_table(by_file, ["file", "cnt", "critical"])
        total = conn.execute("SELECT COUNT(*) FROM crash_risks").fetchone()[0]
        crit = conn.execute("SELECT COUNT(*) FROM crash_risks WHERE severity='critical'").fetchone()[0]
        print(f"\n  Total: {total} crash risks ({crit} critical)")
        return
    print_table(rows, ["file", "line", "risk_type", "severity", "expression", "context"])


def deadmethods_cmd():
    conn = get_conn()
    print("\n  === Dead Methods (declared in .h, never referenced in .cpp) ===")
    rows = conn.execute(
        "SELECT class_name, method_name, header_file, header_line, ref_count FROM dead_methods ORDER BY class_name, method_name LIMIT 50"
    ).fetchall()
    print_table(rows, ["class_name", "method_name", "header_file", "header_line", "ref_count"])
    total = conn.execute("SELECT COUNT(*) FROM dead_methods").fetchone()[0]
    if total > 50:
        print(f"\n  ... and {total - 50} more. Use: py tools/source_graph.py sql \"SELECT * FROM dead_methods\"")


def looprisks_cmd(file_filter=None):
    conn = get_conn()
    if file_filter:
        pattern = f"%{file_filter}%"
        rows = conn.execute(
            "SELECT file, line, risk_type, severity, expression, context FROM infinite_loop_risks "
            "WHERE file LIKE ? ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END, file, line",
            (pattern,)
        ).fetchall()
        print(f"\n  Infinite loop risks in '{file_filter}':")
    else:
        print("\n  === Infinite Loop Risk Summary ===")
        summary = conn.execute(
            "SELECT risk_type, severity, COUNT(*) as cnt FROM infinite_loop_risks "
            "GROUP BY risk_type, severity ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END, cnt DESC"
        ).fetchall()
        print_table(summary, ["risk_type", "severity", "cnt"])
        print("\n  === Top 20 Files with Loop Risks ===")
        by_file = conn.execute(
            "SELECT file, COUNT(*) as cnt, SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) as critical "
            "FROM infinite_loop_risks GROUP BY file ORDER BY critical DESC, cnt DESC LIMIT 20"
        ).fetchall()
        print_table(by_file, ["file", "cnt", "critical"])
        total = conn.execute("SELECT COUNT(*) FROM infinite_loop_risks").fetchone()[0]
        crit = conn.execute("SELECT COUNT(*) FROM infinite_loop_risks WHERE severity='critical'").fetchone()[0]
        print(f"\n  Total: {total} loop risks ({crit} critical)")
        return
    print_table(rows, ["file", "line", "risk_type", "severity", "expression", "context"])


def duplicates_cmd():
    conn = get_conn()
    print("\n  === Duplicate Code Blocks (5+ identical lines across files) ===")
    rows = conn.execute(
        "SELECT file_a, line_a, file_b, line_b, line_count, preview FROM duplicate_blocks ORDER BY file_a, line_a LIMIT 40"
    ).fetchall()
    print_table(rows, ["file_a", "line_a", "file_b", "line_b", "line_count", "preview"])
    total = conn.execute("SELECT COUNT(DISTINCT block_hash) FROM duplicate_blocks").fetchone()[0]
    total_instances = conn.execute("SELECT COUNT(*) FROM duplicate_blocks").fetchone()[0]
    print(f"\n  {total} unique duplicate blocks, {total_instances} cross-file pairs")
    print("  (candidates for extraction into shared functions/utilities)")


def context_cmd(name):
    """Unified context query: summary + functions + deps + rdeps + tests + edges for a file."""
    conn = get_conn()
    pattern = f"%{name}%"

    # 1. File summary
    summary = conn.execute(
        "SELECT file, project, summary, category FROM file_summaries WHERE file LIKE ?",
        (pattern,)
    ).fetchall()
    if summary:
        print(f"\n  === File Summary ===")
        print_table(summary, ["file", "project", "summary", "category"])
    else:
        print(f"\n  (no summary for '{name}')")

    # 2. Functions with line ranges
    funcs = conn.execute(
        "SELECT function_name, class_name, start_line, end_line, signature FROM function_index WHERE file LIKE ? ORDER BY start_line",
        (pattern,)
    ).fetchall()
    if funcs:
        print(f"\n  === Functions ({len(funcs)}) ===")
        print_table(funcs, ["function_name", "class_name", "start_line", "end_line", "signature"])

    symbols = conn.execute(
        "SELECT symbol_name, hot_path, ct_sensitive, batchable, gpu_candidate, risk_level, review_priority "
        "FROM symbol_metadata WHERE file_path LIKE ? ORDER BY review_priority DESC, symbol_name LIMIT 25",
        (pattern,)
    ).fetchall()
    if symbols:
        print(f"\n  === Symbol Metadata ===")
        print_table(symbols, ["symbol_name", "hot_path", "ct_sensitive", "batchable", "gpu_candidate", "risk_level", "review_priority"])

    calls = conn.execute(
        "SELECT caller_symbol, callee_symbol, confidence, call_count FROM call_edges "
        "WHERE caller_file LIKE ? OR callee_file LIKE ? ORDER BY confidence DESC, call_count DESC LIMIT 25",
        (pattern, pattern)
    ).fetchall()
    if calls:
        print(f"\n  === Call Edges ===")
        print_table(calls, ["caller_symbol", "callee_symbol", "confidence", "call_count"])

    # 3. Dependencies (what this file includes/uses)
    deps = conn.execute(
        "SELECT target_file, dep_type FROM dependencies WHERE source_file LIKE ? ORDER BY dep_type, target_file",
        (pattern,)
    ).fetchall()
    if deps:
        print(f"\n  === Depends On ({len(deps)}) ===")
        print_table(deps, ["target_file", "dep_type"])

    # 4. Reverse dependencies (who includes/uses this file)
    rdeps = conn.execute(
        "SELECT source_file, dep_type FROM dependencies WHERE target_file LIKE ? ORDER BY dep_type, source_file",
        (pattern,)
    ).fetchall()
    if rdeps:
        print(f"\n  === Used By ({len(rdeps)}) ===")
        print_table(rdeps, ["source_file", "dep_type"])

    # 5. Edges (tests, extends, etc.)
    edges_out = conn.execute(
        "SELECT target, edge_type, detail FROM edges WHERE source LIKE ? ORDER BY edge_type",
        (pattern,)
    ).fetchall()
    edges_in = conn.execute(
        "SELECT source, edge_type, detail FROM edges WHERE target LIKE ? ORDER BY edge_type",
        (pattern,)
    ).fetchall()
    if edges_out:
        print(f"\n  === Outgoing Edges ===")
        print_table(edges_out, ["target", "edge_type", "detail"])
    if edges_in:
        print(f"\n  === Incoming Edges ===")
        print_table(edges_in, ["source", "edge_type", "detail"])

    # 6. Singletons defined in this file
    singletons = conn.execute(
        "SELECT macro, class_name, description FROM singletons WHERE header LIKE ?",
        (pattern,)
    ).fetchall()
    if singletons:
        print(f"\n  === Singletons ===")
        print_table(singletons, ["macro", "class_name", "description"])

    # 7. Enums in this file
    enums = conn.execute(
        "SELECT name, value_count, values_preview FROM enums WHERE file LIKE ?",
        (pattern,)
    ).fetchall()
    if enums:
        print(f"\n  === Enums ===")
        print_table(enums, ["name", "value_count", "values_preview"])

    # 8. Structs in this file
    structs = conn.execute(
        "SELECT name, field_count, is_packed, fields_preview FROM structs WHERE file LIKE ?",
        (pattern,)
    ).fetchall()
    if structs:
        print(f"\n  === Structs ===")
        print_table(structs, ["name", "field_count", "is_packed", "fields_preview"])

    if not summary and not funcs:
        print(f"\n  No context found for '{name}'. Try a more specific filename.")


def func_cmd(name):
    """Find function by name with exact line ranges."""
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT file, function_name, class_name, start_line, end_line, signature, project FROM function_index "
        "WHERE function_name LIKE ? OR class_name LIKE ? OR signature LIKE ? ORDER BY file, start_line",
        (pattern, pattern, pattern)
    ).fetchall()
    print(f"\n  Functions matching '{name}':")
    print_table(rows, ["file", "function_name", "class_name", "start_line", "end_line", "signature", "project"])

    meta = conn.execute(
        "SELECT symbol_name, file_path, hot_path, ct_sensitive, batchable, gpu_candidate, risk_level, review_priority, risk_score, gain_score "
        "FROM symbol_metadata WHERE symbol_name LIKE ? OR file_path LIKE ? ORDER BY review_priority DESC LIMIT 40",
        (pattern, pattern)
    ).fetchall()
    if meta:
        print(f"\n  === Symbol Metadata ===")
        print_table(meta, ["symbol_name", "file_path", "hot_path", "ct_sensitive", "batchable", "gpu_candidate", "risk_level", "review_priority", "risk_score", "gain_score"])


def api_cmd(name, limit=20):
    """AI-oriented: show CPU public API signatures for a function/type.

    Focuses on the public API surface (src/ implementations, exclude CT/test).
    Shows function signatures + related structs in compact form.
    Designed for minimal token usage when AI agents need API context.

    Usage: source_graph.py api <name> [limit]
    """
    conn = get_conn()
    pattern = f"%{name}%"

    # 1. Function signatures from CPU project (headers + src, exclude CT wrappers)
    rows = conn.execute(
        "SELECT function_name, file, signature, start_line "
        "FROM function_index "
        "WHERE function_name LIKE ? "
        "  AND project = 'cpu' "
        "  AND (file LIKE 'include/secp256k1/%' OR file LIKE 'src/%') "
        "  AND file NOT LIKE '%ct/%' "
        "  AND file NOT LIKE '%ct_sign%' "
        "  AND file NOT LIKE '%ct_point%' "
        "  AND function_name NOT LIKE 'test_%' "
        "ORDER BY file, start_line",
        (pattern,)
    ).fetchall()
    # Deduplicate by function name: prefer earliest match (usually .hpp or main .cpp)
    seen = set()
    deduped = []
    for r in rows:
        key = r['function_name']
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    rows = deduped[:limit]

    if rows:
        print(f"\n  === Public API: {name} ===")
        for r in rows:
            sig = r['signature'] or ''
            # Take only first line of signature (before '{')
            sig_clean = sig.split('{')[0].strip().split('\n')[0].strip()
            if len(sig_clean) > 120:
                sig_clean = sig_clean[:117] + '...'
            print(f"  {r['function_name']}  [{r['file']}:{r['start_line']}]")
            print(f"    {sig_clean}")
    else:
        print(f"\n  No public API matches for '{name}'")

    # 2. Related structs from public headers
    structs = conn.execute(
        "SELECT name, file, line FROM structs "
        "WHERE name LIKE ? AND file LIKE 'include/secp256k1/%' "
        "ORDER BY file, line",
        (pattern,)
    ).fetchall()
    if structs:
        print(f"\n  === Related Structs ===")
        for s in structs:
            print(f"  {s['name']}  [{s['file']}:{s['line']}]")

    # 3. Also search for the term in struct definitions when function matches
    # are sparse (e.g. searching for "ECDSASignature" finds the struct)
    if not rows and not structs:
        # Fallback: search ALL cpu functions for the term in signature
        body_rows = conn.execute(
            "SELECT function_name, file, start_line, signature FROM function_index "
            "WHERE signature LIKE ? AND project = 'cpu' "
            "AND function_name NOT LIKE 'test_%' LIMIT 10",
            (f'%{name}%',)
        ).fetchall()
        if body_rows:
            print(f"\n  === Signatures containing '{name}' ===")
            for r in body_rows:
                sig = (r['signature'] or '').split('{')[0].strip().split('\n')[0].strip()
                print(f"  {r['function_name']}  [{r['file']}:{r['start_line']}]")
                if sig:
                    print(f"    {sig[:120]}")


def gaps_cmd():
    """Find documentation and coverage gaps in the codebase."""
    conn = get_conn()

    # 1. Files without summaries
    print("\n  === Files Without Summaries ===")
    rows = conn.execute(
        "SELECT f.path, f.project FROM files f "
        "LEFT JOIN file_summaries fs ON f.path = fs.file "
        "WHERE fs.id IS NULL AND (f.path LIKE '%.cpp' OR f.path LIKE '%.h') "
        "ORDER BY f.project, f.path LIMIT 30"
    ).fetchall()
    print_table(rows, ["path", "project"])

    # 2. Large files without function index entries
    print("\n  === Large Files Missing Function Index (500+ lines) ===")
    rows = conn.execute(
        "SELECT fl.file, fl.project, fl.line_count FROM file_lines fl "
        "LEFT JOIN function_index fi ON fl.file = fi.file "
        "WHERE fi.id IS NULL AND fl.line_count >= 500 "
        "ORDER BY fl.line_count DESC LIMIT 20"
    ).fetchall()
    print_table(rows, ["file", "project", "line_count"])

    # 3. Files with no tests
    print("\n  === Source Files Without Test Coverage ===")
    rows = conn.execute(
        "SELECT f.path, f.project FROM files f "
        "LEFT JOIN edges e ON f.path = e.target AND e.edge_type = 'tests' "
        "WHERE e.id IS NULL AND f.path LIKE '%.cpp' AND f.project = 'Game' "
        "ORDER BY f.path LIMIT 30"
    ).fetchall()
    print_table(rows, ["path", "project"])

    # 4. Stats
    total_files = conn.execute("SELECT COUNT(*) FROM files WHERE path LIKE '%.cpp' OR path LIKE '%.h'").fetchone()[0]
    summarized = conn.execute("SELECT COUNT(*) FROM file_summaries").fetchone()[0]
    indexed_files = conn.execute("SELECT COUNT(DISTINCT file) FROM function_index").fetchone()[0]
    tested_files = conn.execute("SELECT COUNT(DISTINCT target) FROM edges WHERE edge_type = 'tests'").fetchone()[0]
    print(f"\n  Coverage: {summarized}/{total_files} summarized, {indexed_files}/{total_files} function-indexed, {tested_files}/{total_files} tested")


def summary_cmd():
    """Show project-level overview with key statistics."""
    conn = get_conn()
    print("\n  === Project Summary ===")

    meta = conn.execute(
        "SELECT schema_version, extractor_version, graph_build_revision, built_at FROM graph_metadata ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if meta:
        print(f"\n  Graph Build: schema={meta['schema_version']} extractor={meta['extractor_version']} built_at={meta['built_at']}")
        if meta["graph_build_revision"]:
            print(f"  Revision: {meta['graph_build_revision']}")

    # Total code
    totals = conn.execute(
        "SELECT project, SUM(line_count) as total_lines, COUNT(*) as files FROM file_lines GROUP BY project ORDER BY total_lines DESC"
    ).fetchall()
    print_table(totals, ["project", "total_lines", "files"])

    total_lines = conn.execute("SELECT SUM(line_count) FROM file_lines").fetchone()[0] or 0
    print(f"\n  Total: {total_lines:,} lines of code")

    # Key counts
    counts = {
        "Singletons": conn.execute("SELECT COUNT(*) FROM singletons").fetchone()[0],
        "Classes": conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0],
        "Methods (headers)": conn.execute("SELECT COUNT(*) FROM methods").fetchone()[0],
        "Functions (indexed)": conn.execute("SELECT COUNT(*) FROM function_index").fetchone()[0],
        "Semantic Tags": conn.execute("SELECT COUNT(*) FROM semantic_tags").fetchone()[0],
        "Semantic Profiles": conn.execute("SELECT COUNT(*) FROM semantic_profiles").fetchone()[0],
        "Audit Coverage Rows": conn.execute("SELECT COUNT(*) FROM audit_coverage").fetchone()[0],
        "History Metrics": conn.execute("SELECT COUNT(*) FROM history_metrics").fetchone()[0],
        "Review Queue Items": conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0],
        "Ownership Metrics": conn.execute("SELECT COUNT(*) FROM ownership_metrics").fetchone()[0],
        "Test Mappings": conn.execute("SELECT COUNT(*) FROM test_function_map").fetchone()[0],
        "Call Edges": conn.execute("SELECT COUNT(*) FROM call_edges").fetchone()[0],
        "Symbol Metadata": conn.execute("SELECT COUNT(*) FROM symbol_metadata").fetchone()[0],
        "Analysis Scores": conn.execute("SELECT COUNT(*) FROM analysis_scores").fetchone()[0],
        "Symbol Audit Rows": conn.execute("SELECT COUNT(*) FROM symbol_audit_coverage").fetchone()[0],
        "AI Tasks": conn.execute("SELECT COUNT(*) FROM ai_tasks").fetchone()[0],
        "Research Assets": conn.execute("SELECT COUNT(*) FROM research_assets").fetchone()[0],
        "Research Mentions": conn.execute("SELECT COUNT(*) FROM research_mentions").fetchone()[0],
        "Packet Handlers": conn.execute("SELECT COUNT(*) FROM packet_handlers").fetchone()[0],
        "Events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        "Enums": conn.execute("SELECT COUNT(*) FROM enums").fetchone()[0],
        "Structs": conn.execute("SELECT COUNT(*) FROM structs").fetchone()[0],
        "DB Tables Referenced": conn.execute("SELECT COUNT(DISTINCT table_name) FROM db_tables").fetchone()[0],
        "Prepared Statements": conn.execute("SELECT COUNT(*) FROM prepared_statements").fetchone()[0],
        "#define Macros": conn.execute("SELECT COUNT(*) FROM defines").fetchone()[0],
    }
    print("\n  === Key Metrics ===")
    for label, count in counts.items():
        print(f"  {label:.<30} {count}")

    # Quality
    print("\n  === Code Quality ===")
    quality = {
        "Crash Risks (critical)": conn.execute("SELECT COUNT(*) FROM crash_risks WHERE severity='critical'").fetchone()[0],
        "NULL-Check Risks": conn.execute("SELECT COUNT(*) FROM null_risks").fetchone()[0],
        "Memory Leak Risk Files": conn.execute("SELECT COUNT(*) FROM leak_risks WHERE risk_score > 0").fetchone()[0],
        "Dead Methods": conn.execute("SELECT COUNT(*) FROM dead_methods").fetchone()[0],
        "Duplicate Blocks": conn.execute("SELECT COUNT(DISTINCT block_hash) FROM duplicate_blocks").fetchone()[0],
        "TODO/FIXME Comments": conn.execute("SELECT COUNT(*) FROM todos").fetchone()[0],
        "Low Coverage Files (<40)": conn.execute("SELECT COUNT(*) FROM audit_coverage WHERE coverage_score < 40").fetchone()[0],
        "High Churn Files (>=20)": conn.execute("SELECT COUNT(*) FROM history_metrics WHERE churn_score >= 20").fetchone()[0],
        "Review Queue Items": conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0],
        "Ownership Risk Files (>=5)": conn.execute("SELECT COUNT(*) FROM ownership_metrics WHERE bus_factor_risk >= 5").fetchone()[0],
        "Untested Function Maps": conn.execute("SELECT COUNT(*) FROM review_queue WHERE queue_type='untested_hotspot'").fetchone()[0],
    }
    for label, count in quality.items():
        print(f"  {label:.<30} {count}")


def query_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT query_name, sql_text, connection_type, source_file, line FROM prepared_statements WHERE query_name LIKE ? OR sql_text LIKE ?",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  Prepared statements matching '{name}':")
    print_table(rows, ["query_name", "sql_text", "connection_type", "source_file", "line"])


def tags_cmd(term):
    conn = get_conn()
    pattern = f"%{term}%"
    rows = conn.execute(
        "SELECT tag, entity_type, entity_name, file, line, confidence, evidence "
        "FROM semantic_tags "
        "WHERE tag LIKE ? OR entity_name LIKE ? OR file LIKE ? OR evidence LIKE ? "
        "ORDER BY confidence DESC, tag, entity_type, file, line "
        "LIMIT 80",
        (pattern, pattern, pattern, pattern)
    ).fetchall()
    print(f"\n  Semantic tags matching '{term}':")
    print_table(rows, ["tag", "entity_type", "entity_name", "file", "line", "confidence", "evidence"])


def hotspots_cmd(term=None):
    conn = get_conn()
    params = []
    query = (
        "SELECT entity_type, entity_name, file, line, risk_score, gain_score, "
        "security_sensitive, hot_path_candidate, optimization_candidate, network_surface, fragile_surface, test_surface "
        "FROM semantic_profiles"
    )
    if term:
        pattern = f"%{term}%"
        query += (
            " WHERE entity_name LIKE ? OR file LIKE ? OR evidence LIKE ? "
            "OR EXISTS (SELECT 1 FROM semantic_tags st WHERE st.file = semantic_profiles.file "
            "AND st.entity_name = semantic_profiles.entity_name AND st.tag LIKE ?)"
        )
        params = [pattern, pattern, pattern, pattern]
    query += " ORDER BY gain_score DESC, risk_score DESC, entity_type, file LIMIT 80"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  Semantic hotspots for {title}:")
    print_table(
        rows,
        [
            "entity_type", "entity_name", "file", "line", "risk_score", "gain_score",
            "security_sensitive", "hot_path_candidate", "optimization_candidate",
            "network_surface", "fragile_surface", "test_surface"
        ]
    )


def coverage_cmd(term=None):
    conn = get_conn()
    params = []
    query = (
        "SELECT file, project, coverage_score, unit_test_refs, has_summary, has_function_index, "
        "crash_risk_count, null_risk_count, duplicate_pair_count, semantic_risk_max, semantic_gain_max, notes "
        "FROM audit_coverage"
    )
    if term:
        pattern = f"%{term}%"
        query += " WHERE file LIKE ? OR project LIKE ? OR notes LIKE ?"
        params = [pattern, pattern, pattern]
    query += " ORDER BY coverage_score ASC, semantic_risk_max DESC, file LIMIT 80"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  Coverage signals for {title}:")
    print_table(
        rows,
        [
            "file", "project", "coverage_score", "unit_test_refs", "has_summary", "has_function_index",
            "crash_risk_count", "null_risk_count", "duplicate_pair_count", "semantic_risk_max",
            "semantic_gain_max", "notes"
        ]
    )


def churn_cmd(term=None):
    conn = get_conn()
    params = []
    query = (
        "SELECT file, commit_count, recent_commit_count, unique_authors, bugfix_commits, perf_commits, "
        "audit_commits, last_commit_date, churn_score FROM history_metrics"
    )
    if term:
        pattern = f"%{term}%"
        query += " WHERE file LIKE ?"
        params = [pattern]
    query += " ORDER BY churn_score DESC, recent_commit_count DESC, file LIMIT 80"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  Churn/history signals for {title}:")
    print_table(
        rows,
        ["file", "commit_count", "recent_commit_count", "unique_authors", "bugfix_commits", "perf_commits", "audit_commits", "last_commit_date", "churn_score"]
    )


def reviewqueue_cmd(term=None):
    conn = get_conn()
    params = []
    query = "SELECT file, queue_type, priority_score, rationale FROM review_queue"
    if term:
        pattern = f"%{term}%"
        query += " WHERE file LIKE ? OR queue_type LIKE ? OR rationale LIKE ?"
        params = [pattern, pattern, pattern]
    query += " ORDER BY priority_score DESC, queue_type, file LIMIT 120"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  Review queue for {title}:")
    print_table(rows, ["file", "queue_type", "priority_score", "rationale"])


def ownership_cmd(term=None):
    conn = get_conn()
    params = []
    query = (
        "SELECT file, primary_author, primary_author_share, author_count, blamed_lines, "
        "most_recent_line_date, bus_factor_risk FROM ownership_metrics"
    )
    if term:
        pattern = f"%{term}%"
        query += " WHERE file LIKE ? OR primary_author LIKE ?"
        params = [pattern, pattern]
    query += " ORDER BY bus_factor_risk DESC, primary_author_share DESC, file LIMIT 80"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  Ownership signals for {title}:")
    print_table(
        rows,
        ["file", "primary_author", "primary_author_share", "author_count", "blamed_lines", "most_recent_line_date", "bus_factor_risk"]
    )


def testmap_cmd(term=None):
    conn = get_conn()
    params = []
    query = (
        "SELECT test_file, target_file, function_name, class_name, mapping_type FROM test_function_map"
    )
    if term:
        pattern = f"%{term}%"
        query += " WHERE test_file LIKE ? OR target_file LIKE ? OR function_name LIKE ? OR class_name LIKE ?"
        params = [pattern, pattern, pattern, pattern]
    query += " ORDER BY target_file, mapping_type, class_name, function_name LIMIT 120"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  Test-function mapping for {title}:")
    print_table(rows, ["test_file", "target_file", "function_name", "class_name", "mapping_type"])


def calls_cmd(term):
    conn = get_conn()
    pattern = f"%{term}%"
    rows = conn.execute(
        "SELECT caller_symbol, caller_file, callee_symbol, callee_file, confidence, call_count, evidence "
        "FROM call_edges WHERE caller_symbol LIKE ? OR caller_file LIKE ? OR callee_symbol LIKE ? OR callee_file LIKE ? "
        "ORDER BY confidence DESC, call_count DESC, caller_symbol, callee_symbol LIMIT 120",
        (pattern, pattern, pattern, pattern)
    ).fetchall()
    print(f"\n  Call graph edges matching '{term}':")
    print_table(rows, ["caller_symbol", "caller_file", "callee_symbol", "callee_file", "confidence", "call_count", "evidence"])


def focus_cmd(term, budget=80, core_only=False, json_mode=False, mode="explore"):
    """Compact ranked snapshot optimized for low-token agent context."""
    if json_mode:
        gv = get_graph_view()
        manifest = gv.focus_manifest(term, budget=int(budget), core_only=core_only, mode=mode)
        import json as _json
        print(_json.dumps(manifest, indent=2))
        return
    conn = get_conn()
    budget = _parse_budget_arg(budget, 80)
    candidate_files = _select_candidate_files(conn, term, limit=6, core_only=core_only)
    pattern = f"%{term}%"
    mode_suffix = ", core-only" if core_only else ""
    if not _QUIET:
        print(f"\n  === Focus for '{term}' (budget={budget}{mode_suffix}) ===")

    sections = []
    line_budget = budget

    def add(title, items, max_items):
        nonlocal line_budget
        if not items or line_budget <= 0:
            return
        trimmed = list(items[: min(max_items, line_budget)])
        if trimmed:
            sections.append((title, trimmed))
            line_budget -= len(trimmed)

    if candidate_files:
        file_items = [_compact_file_snapshot(conn, file_name) for file_name in candidate_files]
        add("Primary Files", file_items, 6)

    symbol_rows = _iter_candidate_symbols(conn, term, candidate_files, limit=10)
    if core_only:
        filtered_symbol_rows = [row for row in symbol_rows if _is_core_file(row["file_path"])]
        if filtered_symbol_rows:
            symbol_rows = filtered_symbol_rows
    symbol_items = []
    for row in symbol_rows:
        tags = _compact_text(row["semantic_tags"], 36)
        flags = []
        if row["hot_path"]:
            flags.append("hot")
        if row["ct_sensitive"]:
            flags.append("ct")
        if row["batchable"]:
            flags.append("batch")
        if row["gpu_candidate"]:
            flags.append("gpu")
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        symbol_items.append(
            f"{row['symbol_name']} @ {row['file_path']} | prio={row['review_priority']} gain={row['gain_score']} risk={row['risk_score']} cov={row['audit_coverage_score']}{flag_text} tags={tags}"
        )
    add("Top Symbols", symbol_items, 8)

    call_rows = conn.execute(
        "SELECT caller_symbol, caller_file, callee_symbol, callee_file, confidence, call_count "
        "FROM call_edges WHERE caller_symbol LIKE ? OR callee_symbol LIKE ? OR caller_file LIKE ? OR callee_file LIKE ? "
        "ORDER BY confidence DESC, call_count DESC LIMIT 8",
        (pattern, pattern, pattern, pattern),
    ).fetchall()
    if not call_rows and candidate_files:
        placeholders = _sql_in(candidate_files)
        call_rows = conn.execute(
            f"SELECT caller_symbol, caller_file, callee_symbol, callee_file, confidence, call_count FROM call_edges "
            f"WHERE caller_file IN ({placeholders}) OR callee_file IN ({placeholders}) "
            f"ORDER BY confidence DESC, call_count DESC LIMIT 8",
            candidate_files + candidate_files,
        ).fetchall()
    call_items = [
        f"{row['caller_symbol']} -> {row['callee_symbol']} | {row['caller_file']} -> {row['callee_file']} | conf={row['confidence']} calls={row['call_count']}"
        for row in call_rows
    ]
    add("Call Frontier", call_items, 8)

    risk_items = []
    if candidate_files:
        placeholders = _sql_in(candidate_files)
        crash_rows = conn.execute(
            f"SELECT file, line, risk_type, severity, expression FROM crash_risks WHERE file IN ({placeholders}) "
            "ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END, line LIMIT 6",
            candidate_files,
        ).fetchall()
        null_rows = conn.execute(
            f"SELECT file, line, function_call, risk_type FROM null_risks WHERE file IN ({placeholders}) LIMIT 4",
            candidate_files,
        ).fetchall()
        for row in crash_rows:
            risk_items.append(
                f"[{row['severity']}] {row['file']}:{row['line']} {row['risk_type']} :: {_compact_text(row['expression'], 64)}"
            )
        for row in null_rows:
            risk_items.append(
                f"[null] {row['file']}:{row['line']} {row['function_call']} ({row['risk_type']})"
            )
    add("Top Risks", risk_items, 8)

    test_items = []
    if candidate_files:
        placeholders = _sql_in(candidate_files)
        tests = conn.execute(
            f"SELECT DISTINCT test_file, target_file, function_name, mapping_type FROM test_function_map WHERE target_file IN ({placeholders}) "
            "ORDER BY target_file, test_file LIMIT 8",
            candidate_files,
        ).fetchall()
        for row in tests:
            test_items.append(
                f"{row['test_file']} -> {row['target_file']} :: {row['function_name']} ({row['mapping_type']})"
            )
    add("Tests", test_items, 8)

    research_items = [
        f"{row['asset_path']} :: {row['symbol']} [{row['mention_type']}] {_compact_text(row['context'], 60)}"
        for row in _research_matches(conn, term, limit=5)
    ]
    add("Research", research_items, 5)

    if not sections:
        print("  (no focus results)")
        return

    for title, items in sections:
        _print_compact_section(title, items)


def slice_cmd(term, budget=120, core_only=False, json_mode=False, mode="explore"):
    """Minimal implementation slice: entry files, hot symbols, neighbors, and tests."""
    if json_mode:
        gv = get_graph_view()
        manifest = gv.slice_manifest(term, budget=int(budget), core_only=core_only, mode=mode)
        import json as _json
        print(_json.dumps(manifest, indent=2))
        return
    conn = get_conn()
    budget = _parse_budget_arg(budget, 120)
    pattern = f"%{term}%"
    candidate_files = _select_candidate_files(conn, term, limit=4, core_only=core_only)
    if not candidate_files:
        print(f"\n  No slice found for '{term}'.")
        return

    mode_suffix = ", core-only" if core_only else ""
    if not _QUIET:
        print(f"\n  === Slice for '{term}' (budget={budget}{mode_suffix}) ===")
    sections = []
    line_budget = budget

    def add(title, items, max_items):
        nonlocal line_budget
        if not items or line_budget <= 0:
            return
        trimmed = list(items[: min(max_items, line_budget)])
        if trimmed:
            sections.append((title, trimmed))
            line_budget -= len(trimmed)

    add("Entry Files", [_compact_file_snapshot(conn, file_name) for file_name in candidate_files], 4)

    func_items = []
    placeholders = _sql_in(candidate_files)
    func_rows = conn.execute(
        f"SELECT fi.file, fi.function_name, fi.class_name, fi.start_line, fi.end_line, sm.review_priority, sm.risk_score, sm.gain_score, sm.ct_sensitive, sm.hot_path "
        f"FROM function_index fi LEFT JOIN symbol_metadata sm ON sm.symbol_name = fi.function_name AND sm.file_path = fi.file "
        f"WHERE fi.file IN ({placeholders}) ORDER BY "
        "CASE WHEN fi.function_name LIKE ? THEN 0 WHEN fi.class_name LIKE ? THEN 1 ELSE 2 END, "
        "COALESCE(sm.review_priority, 0) DESC, fi.start_line LIMIT 14",
        candidate_files + [pattern, pattern],
    ).fetchall()
    for row in func_rows:
        label = _function_label(row["class_name"], row["function_name"])
        flags = []
        if row["hot_path"]:
            flags.append("hot")
        if row["ct_sensitive"]:
            flags.append("ct")
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        func_items.append(
            f"{label} @ {row['file']}:{row['start_line']}-{row['end_line']} | prio={row['review_priority'] or 0} gain={row['gain_score'] or 0} risk={row['risk_score'] or 0}{flag_text}"
        )
    add("Entry Functions", func_items, 12)

    symbol_items = []
    symbol_rows = _iter_candidate_symbols(conn, term, candidate_files, limit=10)
    if core_only:
        filtered_symbol_rows = [row for row in symbol_rows if _is_core_file(row["file_path"])]
        if filtered_symbol_rows:
            symbol_rows = filtered_symbol_rows
    for row in symbol_rows:
        symbol_items.append(
            f"{row['symbol_name']} @ {row['file_path']} | prio={row['review_priority']} gain={row['gain_score']} risk={row['risk_score']} cov={row['audit_coverage_score']}"
        )
    add("Entry Symbols", symbol_items, 10)

    neighbor_items = []
    for file_name, score in _build_neighbor_file_scores(conn, candidate_files, limit=10):
        neighbor_items.append(
            f"{file_name} | call_edges={score['call_edges']} dep_edges={score['dep_edges']} call_weight={score['call_weight']}"
        )
    add("Neighbor Files", neighbor_items, 10)

    edge_rows = conn.execute(
        f"SELECT source_file, target_file, dep_type FROM dependencies WHERE source_file IN ({placeholders}) OR target_file IN ({placeholders}) "
        "ORDER BY dep_type, source_file, target_file LIMIT 10",
        candidate_files + candidate_files,
    ).fetchall()
    edge_items = [f"{row['source_file']} -> {row['target_file']} ({row['dep_type']})" for row in edge_rows]
    add("Dependency Edges", edge_items, 10)

    call_rows = conn.execute(
        f"SELECT caller_symbol, caller_file, callee_symbol, callee_file, confidence, call_count FROM call_edges "
        f"WHERE caller_file IN ({placeholders}) OR callee_file IN ({placeholders}) "
        f"ORDER BY confidence DESC, call_count DESC LIMIT 10",
        candidate_files + candidate_files,
    ).fetchall()
    call_items = [
        f"{row['caller_symbol']} -> {row['callee_symbol']} | {row['caller_file']} -> {row['callee_file']} | conf={row['confidence']} calls={row['call_count']}"
        for row in call_rows
    ]
    add("Call Edges", call_items, 10)

    test_rows = conn.execute(
        f"SELECT DISTINCT test_file, target_file, function_name, mapping_type FROM test_function_map WHERE target_file IN ({placeholders}) "
        "ORDER BY target_file, mapping_type, test_file LIMIT 8",
        candidate_files,
    ).fetchall()
    test_items = [
        f"{row['test_file']} -> {row['target_file']} :: {row['function_name']} ({row['mapping_type']})"
        for row in test_rows
    ]
    add("Tests", test_items, 8)

    if not sections:
        print("  (no slice results)")
        return

    for title, items in sections:
        _print_compact_section(title, items)


def trace_cmd(term):
    """Build a compact trace: direct matches, call edges, configs, tests, and research refs."""
    conn = get_conn()
    pattern = f"%{term}%"
    print(f"\n  === Trace for '{term}' ===")

    direct_functions = conn.execute(
        "SELECT file, function_name, class_name, start_line, end_line "
        "FROM function_index WHERE function_name LIKE ? OR class_name LIKE ? OR file LIKE ? "
        "ORDER BY file, start_line LIMIT 20",
        (pattern, pattern, pattern),
    ).fetchall()
    if direct_functions:
        print("\n  Direct function matches:")
        print_table(direct_functions, ["file", "function_name", "class_name", "start_line", "end_line"])

    direct_handlers = conn.execute(
        "SELECT headcode_name, handler_method, source_file, handler_type "
        "FROM packet_handlers WHERE headcode_name LIKE ? OR handler_method LIKE ? OR description LIKE ? "
        "ORDER BY source_file, headcode_name LIMIT 16",
        (pattern, pattern, pattern),
    ).fetchall()
    if direct_handlers:
        print("\n  Direct packet handlers:")
        print_table(direct_handlers, ["headcode_name", "handler_method", "source_file", "handler_type"])

    direct_configs = conn.execute(
        "SELECT path, description FROM config_files WHERE path LIKE ? OR description LIKE ? ORDER BY path LIMIT 12",
        (pattern, pattern),
    ).fetchall()
    if direct_configs:
        print("\n  Direct config/runtime assets:")
        print_table(direct_configs, ["path", "description"])

    candidate_files = _collect_candidate_files(conn, term, limit=10)
    if candidate_files:
        placeholders = _sql_in(candidate_files)
        print("\n  Candidate files:")
        file_rows = [{"file": file_name} for file_name in candidate_files]
        print_table(file_rows, ["file"])

        outgoing = conn.execute(
            f"SELECT caller_symbol, caller_file, callee_symbol, callee_file, confidence, call_count "
            f"FROM call_edges WHERE caller_file IN ({placeholders}) "
            f"ORDER BY confidence DESC, call_count DESC LIMIT 24",
            candidate_files,
        ).fetchall()
        if outgoing:
            print("\n  Outgoing calls from candidate files:")
            print_table(outgoing, ["caller_symbol", "caller_file", "callee_symbol", "callee_file", "confidence", "call_count"])

        incoming = conn.execute(
            f"SELECT caller_symbol, caller_file, callee_symbol, callee_file, confidence, call_count "
            f"FROM call_edges WHERE callee_file IN ({placeholders}) "
            f"ORDER BY confidence DESC, call_count DESC LIMIT 24",
            candidate_files,
        ).fetchall()
        if incoming:
            print("\n  Incoming calls into candidate files:")
            print_table(incoming, ["caller_symbol", "caller_file", "callee_symbol", "callee_file", "confidence", "call_count"])

        deps = conn.execute(
            f"SELECT source_file, target_file, dep_type FROM dependencies "
            f"WHERE source_file IN ({placeholders}) OR target_file IN ({placeholders}) "
            f"ORDER BY dep_type, source_file, target_file LIMIT 30",
            candidate_files + candidate_files,
        ).fetchall()
        if deps:
            print("\n  File dependencies around the trace:")
            print_table(deps, ["source_file", "target_file", "dep_type"])

        tests = conn.execute(
            f"SELECT test_file, target_file, function_name, class_name, mapping_type FROM test_function_map "
            f"WHERE target_file IN ({placeholders}) "
            f"ORDER BY target_file, mapping_type, class_name, function_name LIMIT 30",
            candidate_files,
        ).fetchall()
        if tests:
            print("\n  Related unit tests:")
            print_table(tests, ["test_file", "target_file", "function_name", "class_name", "mapping_type"])

    research = _research_matches(conn, term, limit=18)
    if research:
        print("\n  ReversingResearch cross-reference:")
        print_table(research, ["asset_path", "symbol", "mention_type", "context"])

    if not direct_functions and not direct_handlers and not direct_configs and not candidate_files and not research:
        print("  (no trace results)")


def impact_cmd(term):
    """Aggregate change-impact signals for likely affected files."""
    conn = get_conn()
    candidate_files = _collect_candidate_files(conn, term, limit=12)
    if not candidate_files:
        print(f"\n  No impacted files found for '{term}'.")
        return

    print(f"\n  === Impact for '{term}' ===")
    rows = []
    for file_name in candidate_files:
        coverage = conn.execute(
            "SELECT coverage_score, unit_test_refs, crash_risk_count, null_risk_count FROM audit_coverage WHERE file = ?",
            (file_name,),
        ).fetchone()
        history = conn.execute(
            "SELECT churn_score, recent_commit_count, bugfix_commits FROM history_metrics WHERE file = ?",
            (file_name,),
        ).fetchone()
        ownership = conn.execute(
            "SELECT primary_author, primary_author_share, bus_factor_risk FROM ownership_metrics WHERE file = ?",
            (file_name,),
        ).fetchone()
        tested = conn.execute(
            "SELECT COUNT(DISTINCT test_file) FROM test_function_map WHERE target_file = ?",
            (file_name,),
        ).fetchone()[0]
        callers = conn.execute(
            "SELECT COUNT(*) FROM call_edges WHERE callee_file = ?",
            (file_name,),
        ).fetchone()[0]
        callees = conn.execute(
            "SELECT COUNT(*) FROM call_edges WHERE caller_file = ?",
            (file_name,),
        ).fetchone()[0]
        review_items = conn.execute(
            "SELECT COUNT(*) FROM review_queue WHERE file = ?",
            (file_name,),
        ).fetchone()[0]
        rows.append(
            {
                "file": file_name,
                "coverage_score": coverage["coverage_score"] if coverage else 0,
                "tests": tested,
                "callers": callers,
                "callees": callees,
                "crash_risks": coverage["crash_risk_count"] if coverage else 0,
                "null_risks": coverage["null_risk_count"] if coverage else 0,
                "churn": history["churn_score"] if history else 0,
                "recent_commits": history["recent_commit_count"] if history else 0,
                "bus_factor_risk": ownership["bus_factor_risk"] if ownership else 0,
                "owner": ownership["primary_author"] if ownership else "",
                "review_items": review_items,
            }
        )

    rows.sort(
        key=lambda row: (
            -row["callers"],
            row["coverage_score"],
            -row["crash_risks"],
            -row["null_risks"],
            -row["churn"],
        )
    )
    print_table(
        rows,
        [
            "file",
            "coverage_score",
            "tests",
            "callers",
            "callees",
            "crash_risks",
            "null_risks",
            "churn",
            "recent_commits",
            "bus_factor_risk",
            "owner",
            "review_items",
        ],
    )

    placeholders = _sql_in(candidate_files)
    tests = conn.execute(
        f"SELECT DISTINCT test_file, target_file FROM test_function_map "
        f"WHERE target_file IN ({placeholders}) ORDER BY target_file, test_file LIMIT 24",
        candidate_files,
    ).fetchall()
    if tests:
        print("\n  Impacted unit tests:")
        print_table(tests, ["test_file", "target_file"])

    callers = conn.execute(
        f"SELECT caller_symbol, caller_file, callee_symbol, callee_file, confidence "
        f"FROM call_edges WHERE callee_file IN ({placeholders}) "
        f"ORDER BY confidence DESC, call_count DESC LIMIT 24",
        candidate_files,
    ).fetchall()
    if callers:
        print("\n  High-confidence inbound callers:")
        print_table(callers, ["caller_symbol", "caller_file", "callee_symbol", "callee_file", "confidence"])

    research = _research_matches(conn, term, limit=12)
    if research:
        print("\n  Related research notes:")
        print_table(research, ["asset_path", "symbol", "mention_type", "context"])


def research_cmd(term=None):
    """Search research assets and extracted mentions from ReversingResearch."""
    conn = get_conn()
    if not term:
        print("\n  === ReversingResearch Summary ===")
        assets = conn.execute(
            "SELECT asset_type, COUNT(*) AS count, SUM(size_bytes) AS total_bytes "
            "FROM research_assets GROUP BY asset_type ORDER BY count DESC, asset_type"
        ).fetchall()
        print_table(assets, ["asset_type", "count", "total_bytes"])
        top_assets = conn.execute(
            "SELECT path, asset_type, size_bytes, title, summary FROM research_assets "
            "ORDER BY size_bytes DESC, path LIMIT 12"
        ).fetchall()
        print("\n  Largest assets:")
        print_table(top_assets, ["path", "asset_type", "size_bytes", "title", "summary"])
        return

    pattern = f"%{term}%"
    assets = conn.execute(
        "SELECT path, asset_type, size_bytes, title, summary, symbol_refs, protocol_refs "
        "FROM research_assets WHERE path LIKE ? OR file_name LIKE ? OR title LIKE ? OR summary LIKE ? "
        "OR symbol_refs LIKE ? OR protocol_refs LIKE ? "
        "ORDER BY CASE asset_type WHEN 'markdown' THEN 0 WHEN 'asm' THEN 1 ELSE 2 END, path LIMIT 20",
        (pattern, pattern, pattern, pattern, pattern, pattern),
    ).fetchall()
    mentions = _research_matches(conn, term, limit=30)
    print(f"\n  Research assets for '{term}':")
    if assets:
        print_table(assets, ["path", "asset_type", "size_bytes", "title", "summary", "symbol_refs", "protocol_refs"])
    else:
        print("  (no matching assets)")
    if mentions:
        print("\n  Extracted research mentions:")
        print_table(mentions, ["asset_path", "symbol", "mention_type", "context"])


def symbols_cmd(term=None):
    conn = get_conn()
    params = []
    query = (
        "SELECT symbol_name, file_path, project, semantic_tags, hot_path, ct_sensitive, batchable, gpu_candidate, risk_level, "
        "review_priority, risk_score, gain_score, audit_coverage_score, change_frequency, caller_count, callee_count "
        "FROM symbol_metadata"
    )
    if term:
        pattern = f"%{term}%"
        query += " WHERE symbol_name LIKE ? OR file_path LIKE ? OR semantic_tags LIKE ? OR summary LIKE ?"
        params = [pattern, pattern, pattern, pattern]
    query += " ORDER BY review_priority DESC, gain_score DESC, symbol_name LIMIT 120"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  Symbol metadata for {title}:")
    print_table(
        rows,
        [
            "symbol_name", "file_path", "project", "semantic_tags", "hot_path", "ct_sensitive", "batchable",
            "gpu_candidate", "risk_level", "review_priority", "risk_score", "gain_score",
            "audit_coverage_score", "change_frequency", "caller_count", "callee_count"
        ]
    )


def bottlenecks_cmd(term=None):
    conn = get_conn()
    params = []
    query = (
        "SELECT symbol_name, file_path, overall_priority, perf_priority, safe_priority, hotness_score, complexity_score, fanin_score, fanout_score, "
        "gpu_score, ct_risk_score, audit_gap_score, risk_level, semantic_tags, summary "
        "FROM v_bottleneck_queue"
    )
    if term:
        pattern = f"%{term}%"
        query += " WHERE symbol_name LIKE ? OR file_path LIKE ? OR semantic_tags LIKE ? OR summary LIKE ?"
        params = [pattern, pattern, pattern, pattern]
    query += " ORDER BY overall_priority DESC, hotness_score DESC, symbol_name LIMIT 120"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  Bottleneck queue for {title}:")
    print_table(
        rows,
        [
            "symbol_name", "file_path", "overall_priority", "perf_priority", "safe_priority", "hotness_score", "complexity_score",
            "fanin_score", "fanout_score", "gpu_score", "ct_risk_score", "audit_gap_score", "risk_level", "semantic_tags", "summary"
        ]
    )


# ============================================================
# COMMAND: dataflow  — Type-level data flow graph
# ============================================================

def dataflow_cmd(term=None):
    """Show how data types transform through the codebase.
    Usage: dataflow [type]  — e.g. dataflow Scalar, dataflow ECDSASignature
    Shows: which functions transform FROM or TO this type, and full type chain."""
    conn = get_conn()
    if term:
        pattern = f"%{term}%"
        # Show edges where this type is source OR destination
        rows = conn.execute(
            "SELECT from_type, to_type, via_function, via_file, flow_kind, confidence "
            "FROM dataflow_edges "
            "WHERE from_type LIKE ? OR to_type LIKE ? "
            "ORDER BY confidence DESC, from_type, to_type LIMIT 80",
            (pattern, pattern)
        ).fetchall()
        print(f"\n  Dataflow edges for type '{term}':")
        print_table(rows, ["from_type", "to_type", "via_function", "via_file", "flow_kind", "confidence"])

        # Show downstream chain: what does 'term' eventually produce?
        visited = set()
        chain = [(term, 0)]
        frontier = [term]
        print(f"\n  Downstream type chain from '{term}' (depth ≤ 6):")
        for _depth in range(6):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            nexts = conn.execute(
                f"SELECT DISTINCT to_type, via_function FROM dataflow_edges "
                f"WHERE from_type IN ({placeholders}) AND to_type != from_type",
                frontier
            ).fetchall()
            new_frontier = []
            for r in nexts:
                if r["to_type"] not in visited:
                    visited.add(r["to_type"])
                    new_frontier.append(r["to_type"])
                    chain.append((r["to_type"], _depth + 1))
            frontier = new_frontier
        for t, depth in chain[1:]:
            print(f"    {'  ' * depth}→ {t}")
    else:
        # Show type-flow summary: most-connected types
        rows = conn.execute(
            "SELECT from_type, COUNT(*) as out_degree FROM dataflow_edges GROUP BY from_type "
            "UNION ALL "
            "SELECT to_type, COUNT(*) as in_degree FROM dataflow_edges GROUP BY to_type "
            "ORDER BY 2 DESC LIMIT 40"
        ).fetchall()
        print("\n  Type connectivity (top 40):")
        print_table(rows, ["from_type", "out_degree"])


# ============================================================
# COMMAND: invariants  — Show invariants for a symbol / type
# ============================================================

def invariants_cmd(term=None):
    """Show invariants (assert/precondition/comment) for a function or type.
    Usage: invariants [term]  — term can be function name or type name"""
    conn = get_conn()
    if term:
        pattern = f"%{term}%"
        rows = conn.execute(
            "SELECT symbol_name, file, line, invariant_type, severity, invariant_text, related_type, confidence "
            "FROM invariants "
            "WHERE symbol_name LIKE ? OR invariant_text LIKE ? OR related_type LIKE ? "
            "ORDER BY severity DESC, confidence DESC, file, line LIMIT 100",
            (pattern, pattern, pattern)
        ).fetchall()
        title = f"'{term}'"
    else:
        rows = conn.execute(
            "SELECT symbol_name, file, line, invariant_type, severity, invariant_text, related_type, confidence "
            "FROM invariants ORDER BY severity DESC, confidence DESC, file LIMIT 80"
        ).fetchall()
        title = "all (top 80 by severity)"
    print(f"\n  Invariants for {title}:")
    print_table(rows, ["symbol_name", "file", "line", "invariant_type", "severity", "invariant_text", "related_type", "confidence"])


# ============================================================
# COMMAND: invcheck  — Find functions with invariant-sensitive types but no checks
# ============================================================

def invcheck_cmd(term=None):
    """Find functions that handle invariant-sensitive types but have no corresponding assert.
    Usage: invcheck [file_pattern]
    Threat model: missing normalization / precondition check → potential exploit path."""
    conn = get_conn()

    # Which types have known invariants?
    typed_invariants = conn.execute(
        "SELECT DISTINCT related_type FROM invariants WHERE related_type IS NOT NULL"
    ).fetchall()
    sensitive_types = {r["related_type"] for r in typed_invariants}

    if not sensitive_types:
        print("  [invcheck] No typed invariants indexed yet. Run 'build -i' first.")
        return

    # Functions that handle sensitive types (from dataflow) but have NO assert in their body
    has_assert = {
        r["symbol_name"]
        for r in conn.execute(
            "SELECT DISTINCT symbol_name FROM invariants WHERE invariant_type IN ('assert','precondition','ct_guard')"
        ).fetchall()
    }

    # Functions that consume sensitive types (from dataflow_edges)
    placeholders = ",".join("?" * len(sensitive_types))
    sensitive_consumers = conn.execute(
        f"SELECT DISTINCT via_function, via_file FROM dataflow_edges "
        f"WHERE from_type IN ({placeholders}) ORDER BY via_file, via_function",
        list(sensitive_types)
    ).fetchall()

    risky = []
    for r in sensitive_consumers:
        fname = r["via_function"]
        ffile = r["via_file"]
        if term and term.lower() not in ffile.lower() and term.lower() not in fname.lower():
            continue
        if fname not in has_assert:
            risky.append({"function": fname, "file": ffile, "risk": "no-assert-for-sensitive-type"})

    print(f"\n  [invcheck] Functions handling sensitive types ({', '.join(sorted(sensitive_types)[:6])}...) with NO assert:")
    print(f"  Found {len(risky)} potential gaps:\n")
    for r in risky[:60]:
        print(f"    {r['function']:<40}  {r['file']}")


# ============================================================
# COMMAND: backendmap  — CPU/CUDA/OpenCL/Metal parity view
# ============================================================

def backendmap_cmd(term=None):
    """Show cross-backend operation mapping.
    Usage: backendmap [operation]  — show one op; no arg = show all gaps
    Flags partial parity: missing CUDA, OpenCL, or Metal implementations."""
    conn = get_conn()
    if term:
        pattern = f"%{term}%"
        rows = conn.execute(
            "SELECT operation_name, cpu_symbol, cpu_file, cuda_symbol, cuda_file, "
            "opencl_symbol, opencl_file, metal_symbol, metal_file, parity_status, notes "
            "FROM backend_map WHERE operation_name LIKE ? OR cpu_symbol LIKE ? "
            "ORDER BY parity_status, operation_name LIMIT 40",
            (pattern, pattern)
        ).fetchall()
        title = f"'{term}'"
    else:
        # Show gaps: anything not 'full' or 'cpu-only'
        rows = conn.execute(
            "SELECT operation_name, cpu_symbol, cpu_file, cuda_symbol, cuda_file, "
            "opencl_symbol, opencl_file, metal_symbol, metal_file, parity_status "
            "FROM backend_map WHERE parity_status NOT IN ('full','cpu-only','single-backend') "
            "ORDER BY parity_status, operation_name LIMIT 80"
        ).fetchall()
        title = "parity gaps"
        # Also print summary counts
        counts = conn.execute(
            "SELECT parity_status, COUNT(*) as n FROM backend_map GROUP BY parity_status ORDER BY n DESC"
        ).fetchall()
        print("\n  Backend parity summary:")
        for c in counts:
            print(f"    {c['parity_status']:<20} {c['n']:>5}")

    print(f"\n  Backend map for {title}:")
    print_table(rows, [
        "operation_name", "cpu_symbol", "cuda_symbol", "opencl_symbol", "metal_symbol", "parity_status"
    ])


# ============================================================
# COMMAND: exploitpath  — Traverse call graph for exploit opportunities
# ============================================================

def exploitpath_cmd(term=None):
    """Traverse the call graph from an entry point looking for exploit opportunities.
    Usage: exploitpath [entry_function_or_file]
    Heuristics:
      - Parser boundary (input from untrusted source)
      - Missing normalization (handles Scalar/Point but no assert)
      - CT violation candidates (high-risk CT path without CT guard)
      - Invariant-constrained types passed to unchecked callees
    """
    conn = get_conn()

    # Step 1: identify entry points
    if term:
        pattern = f"%{term}%"
        entries = conn.execute(
            "SELECT DISTINCT caller_symbol, caller_file FROM call_edges "
            "WHERE caller_symbol LIKE ? OR caller_file LIKE ? "
            "UNION "
            "SELECT DISTINCT symbol_name, file_path FROM symbol_metadata "
            "WHERE symbol_name LIKE ? OR file_path LIKE ? "
            "LIMIT 20",
            (pattern, pattern, pattern, pattern)
        ).fetchall()
    else:
        # Default: start from parser/boundary functions
        entries = conn.execute(
            "SELECT DISTINCT caller_symbol, caller_file FROM call_edges "
            "WHERE caller_file LIKE '%parse%' OR caller_file LIKE '%abi%' "
            "   OR caller_symbol LIKE '%parse%' OR caller_symbol LIKE '%verify%' "
            "LIMIT 20"
        ).fetchall()

    if not entries:
        print("  [exploitpath] No entry points found. Try: exploitpath ecdsa_sign")
        return

    # Step 2: load risk signals
    no_assert_set = set()
    for r in conn.execute(
        "SELECT DISTINCT via_function FROM dataflow_edges de "
        "WHERE NOT EXISTS ("
        "  SELECT 1 FROM invariants inv WHERE inv.symbol_name = de.via_function "
        "  AND inv.invariant_type IN ('assert','precondition','ct_guard')"
        ")"
    ).fetchall():
        no_assert_set.add(r["via_function"])

    ct_sensitive_set = set()
    for r in conn.execute(
        "SELECT DISTINCT symbol_name FROM symbol_metadata WHERE ct_sensitive = 1"
    ).fetchall():
        ct_sensitive_set.add(r["symbol_name"])

    # Step 3: BFS along call edges, flag risky nodes
    visited = set()
    queue = [(e["caller_symbol"], e["caller_file"], 0) for e in entries[:5]]
    findings = []

    while queue:
        sym, ffile, depth = queue.pop(0)
        if sym in visited or depth > 8:
            continue
        visited.add(sym)

        risks = []
        if sym in no_assert_set:
            risks.append("no-invariant-check")
        if sym in ct_sensitive_set:
            risks.append("ct-sensitive")
        if "parse" in sym.lower() or "from_bytes" in sym.lower():
            risks.append("parser-boundary")
        if "verify" not in sym.lower() and ("sign" in sym.lower() or "scalar_mul" in sym.lower()):
            risks.append("signing-path")

        if risks:
            findings.append({
                "depth": depth,
                "symbol": sym,
                "file": ffile,
                "risks": "|".join(risks)
            })

        # Expand callees
        callees = conn.execute(
            "SELECT callee_symbol, callee_file FROM call_edges "
            "WHERE caller_symbol = ? AND caller_file = ? "
            "ORDER BY confidence DESC, call_count DESC LIMIT 8",
            (sym, ffile)
        ).fetchall()
        for c in callees:
            if c["callee_symbol"] not in visited:
                queue.append((c["callee_symbol"], c["callee_file"], depth + 1))

    print(f"\n  [exploitpath] Entry: '{term or 'parser/abi boundary'}' — {len(findings)} risky nodes found:\n")
    findings.sort(key=lambda x: (x["depth"], x["file"]))
    for f in findings[:60]:
        depth_indent = "  " * f["depth"]
        print(f"    [{f['depth']}]{depth_indent} {f['symbol']:<45} {f['risks']}")
        print(f"         {' ' * (f['depth'] * 2)} → {f['file']}")


# ============================================================
# COMMAND: hotpath  — Full hot call chain from entry to leaf
# ============================================================

def hotpath_cmd(term=None):
    """Show the hot call chain rooted at a function or symbol.
    Usage: hotpath [function_or_type]
    Combines: call_graph depth + hot_path flag + call_count to show critical paths."""
    conn = get_conn()

    if not term:
        # Default: top hot symbols as roots
        roots = conn.execute(
            "SELECT symbol_name, file_path, hot_path, caller_count, callee_count "
            "FROM symbol_metadata WHERE hot_path = 1 "
            "ORDER BY caller_count DESC LIMIT 10"
        ).fetchall()
        print("\n  Top hot path roots (use 'hotpath <symbol>' to expand):")
        print_table(roots, ["symbol_name", "file_path", "hot_path", "caller_count", "callee_count"])
        return

    pattern = f"%{term}%"
    # Find matching symbols
    start_rows = conn.execute(
        "SELECT DISTINCT caller_symbol, caller_file FROM call_edges "
        "WHERE caller_symbol LIKE ? OR caller_file LIKE ? LIMIT 5",
        (pattern, pattern)
    ).fetchall()

    if not start_rows:
        print(f"  [hotpath] No call edges found for '{term}'")
        return

    # BFS down call graph, tracking cumulative call_count product
    visited = set()
    paths = []

    def _dfs(sym, ffile, depth, path_so_far, cum_calls):
        if sym in visited or depth > 10:
            return
        visited.add(sym)
        hot = conn.execute(
            "SELECT hot_path, caller_count FROM symbol_metadata WHERE symbol_name = ? AND file_path = ? LIMIT 1",
            (sym, ffile)
        ).fetchone()
        is_hot = hot["hot_path"] if hot else 0
        path_so_far = path_so_far + [(sym, ffile, depth, cum_calls, is_hot)]

        callees = conn.execute(
            "SELECT callee_symbol, callee_file, call_count FROM call_edges "
            "WHERE caller_symbol = ? AND caller_file = ? ORDER BY call_count DESC LIMIT 6",
            (sym, ffile)
        ).fetchall()
        if not callees:
            paths.append(path_so_far)
        else:
            for c in callees:
                _dfs(c["callee_symbol"], c["callee_file"], depth + 1, path_so_far, cum_calls * (c["call_count"] or 1))

    for r in start_rows[:3]:
        visited.clear()
        _dfs(r["caller_symbol"], r["caller_file"], 0, [], 1)

    if not paths:
        print(f"  [hotpath] No paths found from '{term}'")
        return

    # Sort paths by cumulative call count descending
    paths.sort(key=lambda p: -max(node[3] for node in p))

    print(f"\n  Hot call chains from '{term}' (top {min(5, len(paths))}):\n")
    for path in paths[:5]:
        print("  — Chain:")
        for sym, ffile, depth, cum, is_hot in path:
            hot_marker = " ★HOT" if is_hot else ""
            print(f"    {'  ' * depth}{sym}{hot_marker}  (cum_calls={cum})  {ffile}")
        print()


def auditmap_cmd(term=None):
    conn = get_conn()
    params = []
    query = (
        "SELECT symbol_name, file_path, covered_by_tests, test_count, mapping_types, review_queue_types, audit_modules, coverage_score, last_status, historical_failures, evidence "
        "FROM symbol_audit_coverage"
    )
    if term:
        pattern = f"%{term}%"
        query += " WHERE symbol_name LIKE ? OR file_path LIKE ? OR review_queue_types LIKE ? OR audit_modules LIKE ? OR evidence LIKE ?"
        params = [pattern, pattern, pattern, pattern, pattern]
    query += " ORDER BY coverage_score ASC, historical_failures DESC, symbol_name LIMIT 120"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  Symbol audit coverage for {title}:")
    print_table(rows, ["symbol_name", "file_path", "covered_by_tests", "test_count", "mapping_types", "review_queue_types", "audit_modules", "coverage_score", "last_status", "historical_failures", "evidence"])


def aitasks_cmd(term=None):
    conn = get_conn()
    params = []
    query = "SELECT task_type, symbol_name, file_path, status, priority, rationale, prompt FROM ai_tasks"
    if term:
        pattern = f"%{term}%"
        query += " WHERE task_type LIKE ? OR symbol_name LIKE ? OR file_path LIKE ? OR rationale LIKE ? OR prompt LIKE ?"
        params = [pattern, pattern, pattern, pattern, pattern]
    query += " ORDER BY priority DESC, task_type, symbol_name LIMIT 120"
    rows = conn.execute(query, params).fetchall()
    title = f"'{term}'" if term else "all"
    print(f"\n  AI task queue for {title}:")
    print_table(rows, ["task_type", "symbol_name", "file_path", "status", "priority", "rationale", "prompt"])


def cfgkey_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT key_name, getter_type, source_file, line, context FROM config_keys WHERE key_name LIKE ? OR context LIKE ? ORDER BY key_name, source_file",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  Config keys matching '{name}':")
    print_table(rows, ["key_name", "getter_type", "source_file", "line", "context"])


def define_cmd(name):
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT name, value, file, line, category FROM defines WHERE name LIKE ? OR value LIKE ? ORDER BY category, name",
        (pattern, pattern)
    ).fetchall()
    print(f"\n  #defines matching '{name}':")
    print_table(rows, ["name", "value", "file", "line", "category"])


def complexity_cmd():
    conn = get_conn()
    print("\n  === Top 30 Largest Files (by line count) ===")
    rows = conn.execute(
        "SELECT file, project, line_count, category FROM file_lines ORDER BY line_count DESC LIMIT 30"
    ).fetchall()
    print_table(rows, ["file", "project", "line_count", "category"])
    # Summary
    totals = conn.execute(
        "SELECT project, SUM(line_count) as total, COUNT(*) as files, MAX(line_count) as biggest FROM file_lines GROUP BY project"
    ).fetchall()
    print("\n  === Summary by Project ===")
    print_table(totals, ["project", "total", "files", "biggest"])


def stats_cmd(show_header=True):
    conn = get_conn()
    if show_header:
        print()
    tables = [
        ("graph_metadata", "Graph Metadata"),
        ("singletons", "Singletons"),
        ("events", "Events/Dungeons"),
        ("ai_handlers", "AI Handlers"),
        ("player_files", "Player Files"),
        ("packet_handlers", "Packet Handlers"),
        ("inventory_scripts", "Inventory Scripts"),
        ("constants", "Constants"),
        ("config_files", "Config Files"),
        ("enums", "Enums"),
        ("structs", "Structs/Packets"),
        ("classes", "Classes"),
        ("methods", "Methods"),
        ("todos", "TODOs/FIXMEs"),
        ("db_tables", "DB Table Refs"),
        ("prepared_statements", "Prepared Stmts"),
        ("config_keys", "Config Keys"),
        ("defines", "#define Macros"),
        ("file_lines", "File Line Counts"),
        ("leak_risks", "Leak Risk Files"),
        ("null_risks", "NULL-Check Risks"),
        ("raw_pointers", "Raw Ptr Members"),
        ("unsafe_casts", "C-Style Casts"),
        ("crash_risks", "Crash Risk Patterns"),
        ("infinite_loop_risks", "Infinite Loop Risks"),
        ("dead_methods", "Dead Methods"),
        ("duplicate_blocks", "Duplicate Blocks"),
        ("file_summaries", "File Summaries"),
        ("function_index", "Function Index"),
        ("semantic_tags", "Semantic Tags"),
        ("semantic_profiles", "Semantic Profiles"),
        ("audit_coverage", "Audit Coverage"),
        ("history_metrics", "History Metrics"),
        ("review_queue", "Review Queue"),
        ("ownership_metrics", "Ownership Metrics"),
        ("test_function_map", "Test Function Map"),
        ("call_edges", "Call Graph Edges"),
        ("symbol_metadata", "Symbol Metadata"),
        ("analysis_scores", "Analysis Scores"),
        ("symbol_audit_coverage", "Symbol Audit Coverage"),
        ("ai_tasks", "AI Tasks"),
        ("research_assets", "Research Assets"),
        ("research_mentions", "Research Mentions"),
        ("edges", "Edges/Relations"),
        ("files", "Source Files"),
        ("dependencies", "Dependencies"),
        ("fts_index", "FTS Index Entries"),
    ]
    print("  === Source Graph Statistics ===")
    for table, label in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {label:.<30} {count}")


def validateai_cmd(base_path=None):
    if base_path:
        ai_dir = Path(base_path)
    else:
        ai_dir = RUNTIME_DATA_ROOT / "Monster" / "AI"

    ai_dir = ai_dir.resolve()
    automata_path = ai_dir / "MonsterAiAutomata.xml"
    element_path = ai_dir / "MonsterAiElement.xml"
    unit_path = ai_dir / "MonsterAiUnit.xml"

    print(f"\n  Validating Monster AI config in: {ai_dir}")

    required = [automata_path, element_path, unit_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("\n  Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return

    supported_transition_types = {0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12}
    supported_element_classes = {
        1, 11, 12, 13, 14,
        21, 22, 23,
        31, 32,
        41,
        51, 52, 53,
        61, 62, 64, 65, 66, 67, 68, 71
    }
    def parse_xml(path):
        return ET.parse(path).getroot()

    def line_number(path, needle):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if needle in line:
                        return idx
        except Exception:
            return None
        return None

    root_automata = parse_xml(automata_path)
    root_element = parse_xml(element_path)
    root_unit = parse_xml(unit_path)

    elements = {}
    units = {}
    issues = []

    for data in root_element.findall("Data"):
        elem_id = int(data.attrib.get("ID", "-1"))
        elem_class = int(data.attrib.get("Class", "-1"))
        elem_state = int(data.attrib.get("State", "-1"))
        elements[elem_id] = {"class": elem_class, "state": elem_state, "name": data.attrib.get("Name", "")}

        if elem_class not in supported_element_classes:
            issues.append(("unsupported_element_class", line_number(element_path, f'ID="{elem_id}"'),
                           f"Element ID {elem_id} uses unsupported Class={elem_class}"))

    for data in root_unit.findall("Data"):
        unit_id = int(data.attrib.get("ID", "-1"))
        units[unit_id] = dict(data.attrib)

        for attr_name in ("Normal", "Move", "Attack", "Heal", "Avoid", "Help", "Special", "Event"):
            raw = data.attrib.get(attr_name, "-1")
            try:
                ref = int(raw)
            except ValueError:
                issues.append(("invalid_unit_ref", line_number(unit_path, f'ID="{unit_id}"'),
                               f"Unit ID {unit_id} has non-integer {attr_name}='{raw}'"))
                continue

            if ref == -1:
                continue

            if ref not in elements:
                issues.append(("orphan_element_ref", line_number(unit_path, f'ID="{unit_id}"'),
                               f"Unit ID {unit_id} references missing element {attr_name}={ref}"))

    for data in root_automata.findall("Data"):
        raw_id = data.attrib.get("ID", "-1")
        current_state = data.attrib.get("CurrentState", "?")
        next_state = data.attrib.get("NextState", "?")
        priority = data.attrib.get("Priority", "?")
        line = line_number(automata_path, f'ID="{raw_id}"')

        if "TransitionType" not in data.attrib:
            issues.append(("missing_transition_type", line,
                           f"Automata ID {raw_id} state {current_state}->{next_state} priority {priority} is missing TransitionType"))
            continue

        transition_type = int(data.attrib["TransitionType"])
        if transition_type not in supported_transition_types:
            issues.append(("unsupported_transition_type", line,
                           f"Automata ID {raw_id} state {current_state}->{next_state} priority {priority} uses unsupported TransitionType={transition_type}"))

        if data.attrib.get("TransitionRate", "").count(" ") > 0:
            issues.append(("suspicious_transition_rate", line,
                           f"Automata ID {raw_id} has suspicious TransitionRate formatting"))

    print("\n  === AI Validation Summary ===")
    print(f"  Elements ................. {len(elements)}")
    print(f"  Units .................... {len(units)}")
    print(f"  Issues ................... {len(issues)}")

    if not issues:
        print("\n  No AI config issues found.")
        return

    grouped = defaultdict(list)
    for issue_type, line, message in issues:
        grouped[issue_type].append((line, message))

    for issue_type in sorted(grouped.keys()):
        print(f"\n  [{issue_type}]")
        for line, message in grouped[issue_type][:20]:
            if line:
                print(f"  - line {line}: {message}")
            else:
                print(f"  - {message}")

    if len(issues) > 20:
        print(f"\n  Showing first {min(len(issues), 20)} issues per category.")


def sql_cmd(query):
    conn = get_conn()
    try:
        rows = conn.execute(query).fetchall()
        if rows:
            columns = [desc[0] for desc in conn.execute(query).description]
            # Convert to dicts for print_table
            class DictRow:
                def __init__(self, row, cols):
                    self._data = {c: row[i] for i, c in enumerate(cols)}
                def __getitem__(self, key):
                    return self._data.get(key)
            dict_rows = [DictRow(r, columns) for r in rows]
            print_table(dict_rows, columns)
        else:
            print("  (no results)")
    except Exception as e:
        print(f"  SQL Error: {e}")


# ============================================================
# NEW COMMANDS: body, summarize, decide, decisions, bundle, claudemd, init, export-config
# ============================================================

def body_cmd(name):
    """Print function body from snippet cache."""
    conn = get_conn()
    pattern = f"%{name}%"
    rows = conn.execute(
        "SELECT file, function_name, class_name, start_line, end_line, body, line_count, project "
        "FROM function_bodies WHERE function_name LIKE ? OR class_name LIKE ? "
        "ORDER BY file, start_line LIMIT 10",
        (pattern, pattern)
    ).fetchall()
    if not rows:
        print(f"  No function bodies matching '{name}'")
        return
    for row in rows:
        cls = f"{row['class_name']}::" if row['class_name'] else ""
        print(f"\n  === {cls}{row['function_name']} ({row['file']}:{row['start_line']}-{row['end_line']}, {row['line_count']} lines) ===")
        print(row['body'])


def bodygrep_cmd(term):
    """Search for a string literal inside function bodies (SQL LIKE).

    This replaces grep/rg for finding error messages, throw strings,
    constant literals, or any text inside indexed function bodies.
    """
    conn = get_conn()
    pattern = f"%{term}%"
    rows = conn.execute(
        "SELECT function_name, class_name, file, start_line, end_line, line_count, project "
        "FROM function_bodies WHERE body LIKE ? "
        "ORDER BY file, start_line LIMIT 30",
        (pattern,)
    ).fetchall()
    print(f"\n  Body grep: '{term}'")
    if not rows:
        print("  (no results)")
        return
    print_table(rows, ["function_name", "class_name", "file", "start_line", "end_line", "line_count", "project"])
    print(f"\n  ({len(rows)} results)")


def summarize_cmd(args):
    """Generate or show function summaries."""
    conn = get_conn()

    # Parse args
    generate = "--generate" in args
    pattern = None
    for a in args:
        if not a.startswith("--"):
            pattern = a
            break

    if generate:
        _generate_heuristic_summaries(conn, pattern)
        return

    if pattern:
        like = f"%{pattern}%"
        rows = conn.execute(
            "SELECT file, function_name, class_name, start_line, end_line, summary, params, "
            "return_type, side_effects, stale, generator FROM function_summaries "
            "WHERE function_name LIKE ? OR class_name LIKE ? OR file LIKE ? "
            "ORDER BY file, start_line LIMIT 30",
            (like, like, like)
        ).fetchall()
        if rows:
            print(f"\n  Function summaries matching '{pattern}':")
            for row in rows:
                cls = f"{row['class_name']}::" if row['class_name'] else ""
                stale_mark = " [STALE]" if row['stale'] else ""
                print(f"  {cls}{row['function_name']} ({row['file']}:{row['start_line']}){stale_mark}")
                if row['summary']:
                    print(f"    {row['summary']}")
                if row['params']:
                    print(f"    params: {row['params']}")
                if row['side_effects']:
                    print(f"    side_effects: {row['side_effects']}")
            print(f"\n  ({len(rows)} results)")
        else:
            print(f"  No summaries matching '{pattern}'. Run 'summarize --generate {pattern}' to create them.")
        return

    # Show summary statistics
    total = conn.execute("SELECT COUNT(*) FROM function_summaries").fetchone()[0]
    stale = conn.execute("SELECT COUNT(*) FROM function_summaries WHERE stale = 1").fetchone()[0]
    bodies = conn.execute("SELECT COUNT(*) FROM function_bodies").fetchone()[0]
    unsummarized = bodies - total
    print(f"\n  Function Summary Statistics:")
    print(f"    Total functions:    {bodies}")
    print(f"    With summaries:     {total}")
    print(f"    Stale summaries:    {stale}")
    print(f"    Unsummarized:       {unsummarized}")


def _generate_heuristic_summaries(conn, pattern=None):
    """Generate heuristic summaries from function signatures."""
    query = """
        SELECT fb.file, fb.function_name, fb.class_name, fb.start_line, fb.end_line,
               fb.body_hash, fb.line_count, fi.signature
        FROM function_bodies fb
        JOIN function_index fi ON fb.file = fi.file AND fb.function_name = fi.function_name
            AND fb.start_line = fi.start_line
        LEFT JOIN function_summaries fs ON fb.file = fs.file AND fb.function_name = fs.function_name
            AND fb.start_line = fs.start_line
        WHERE fs.id IS NULL
    """
    params = []
    if pattern:
        query += " AND (fb.function_name LIKE ? OR fb.class_name LIKE ?)"
        params = [f"%{pattern}%", f"%{pattern}%"]
    query += " ORDER BY fb.file, fb.start_line"
    rows = conn.execute(query, params).fetchall()

    count = 0
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        sig = row["signature"] or ""
        # Extract return type and params from signature
        return_type = ""
        params_str = ""
        paren_idx = sig.find("(")
        if paren_idx > 0:
            prefix = sig[:paren_idx].strip()
            parts = prefix.rsplit(None, 1)
            if len(parts) > 1:
                return_type = parts[0].strip()
            # Extract params
            close = sig.rfind(")")
            if close > paren_idx:
                params_str = sig[paren_idx + 1:close].strip()

        # Build heuristic summary
        cls = f"{row['class_name']}::" if row['class_name'] else ""
        summary = f"{cls}{row['function_name']}"
        if return_type:
            summary = f"Returns {return_type}."
        else:
            summary = f"{row['function_name']} ({row['line_count']} lines)"

        try:
            conn.execute(
                "INSERT OR IGNORE INTO function_summaries "
                "(file, function_name, class_name, start_line, end_line, summary, params, "
                "return_type, side_effects, body_hash, stale, generated_at, generator) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?)",
                (row["file"], row["function_name"], row["class_name"],
                 row["start_line"], row["end_line"], summary, params_str,
                 return_type, None, row["body_hash"], now, "heuristic")
            )
            count += 1
        except Exception:
            pass

    conn.commit()
    print(f"  Generated {count} heuristic summaries")


def decide_cmd(args):
    """Log an architecture decision."""
    conn = get_conn()

    # Parse args
    if "--supersede" in args:
        # Supersede an existing decision
        idx = args.index("--supersede")
        if idx + 1 < len(args):
            old_id = int(args[idx + 1])
            new_text = " ".join(args[idx + 2:]) if idx + 2 < len(args) else ""
            conn.execute("UPDATE decisions SET status = 'superseded' WHERE id = ?", (old_id,))
            if new_text:
                conn.execute(
                    "INSERT INTO decisions (created_at, decision, rationale, status) VALUES (?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), new_text, f"Supersedes #{old_id}", "active")
                )
            conn.commit()
            print(f"  Decision #{old_id} superseded.")
        return

    decision_text = ""
    file_name = None
    func_name = None
    rationale = ""
    tags = ""
    author = ""

    i = 0
    text_parts = []
    while i < len(args):
        if args[i] == "--file" and i + 1 < len(args):
            file_name = args[i + 1]
            i += 2
        elif args[i] == "--func" and i + 1 < len(args):
            func_name = args[i + 1]
            i += 2
        elif args[i] == "--why" and i + 1 < len(args):
            rationale = args[i + 1]
            i += 2
        elif args[i] == "--tags" and i + 1 < len(args):
            tags = args[i + 1]
            i += 2
        elif args[i] == "--author" and i + 1 < len(args):
            author = args[i + 1]
            i += 2
        else:
            text_parts.append(args[i])
            i += 1

    decision_text = " ".join(text_parts)
    if not decision_text:
        print("  Usage: decide \"<decision text>\" --why \"<rationale>\" [--file X] [--func Y] [--tags Z]")
        return

    conn.execute(
        "INSERT INTO decisions (created_at, file, function_name, decision, rationale, alternatives, author, tags, status) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), file_name, func_name,
         decision_text, rationale, None, author, tags, "active")
    )
    conn.commit()
    print(f"  Decision logged: {decision_text[:80]}")


def decisions_cmd(term=None):
    """Query decision log."""
    conn = get_conn()
    if term:
        pattern = f"%{term}%"
        rows = conn.execute(
            "SELECT id, created_at, file, function_name, decision, rationale, tags, status "
            "FROM decisions WHERE decision LIKE ? OR rationale LIKE ? OR file LIKE ? "
            "OR function_name LIKE ? OR tags LIKE ? ORDER BY created_at DESC LIMIT 30",
            (pattern, pattern, pattern, pattern, pattern)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, created_at, file, function_name, decision, rationale, tags, status "
            "FROM decisions WHERE status = 'active' ORDER BY created_at DESC LIMIT 30"
        ).fetchall()

    if not rows:
        print("  No decisions found.")
        return

    print(f"\n  Architecture Decisions:")
    for row in rows:
        status = f" [{row['status']}]" if row['status'] != 'active' else ""
        file_info = f" ({row['file']}" if row['file'] else ""
        if row['function_name']:
            file_info += f"::{row['function_name']}"
        if file_info:
            file_info += ")"
        print(f"  #{row['id']} [{row['created_at'][:10]}]{status}{file_info}")
        print(f"    {row['decision']}")
        if row['rationale']:
            print(f"    Why: {row['rationale']}")
        if row['tags']:
            print(f"    Tags: {row['tags']}")
        print()
    print(f"  ({len(rows)} decisions)")


def bundle_cmd(task_type, term, max_lines=2000, json_mode=False):
    """Assemble minimal context bundle for a task."""
    if json_mode:
        gv = get_graph_view()
        manifest = gv.bundle_manifest(task_type, term, budget=max(max_lines // 50, 40))
        import json as _json
        print(_json.dumps(manifest, indent=2))
        return
    conn = get_conn()
    sections = []
    line_budget = max_lines
    pattern = f"%{term}%"

    def add_section(title, content, priority=5, budget=500):
        nonlocal line_budget
        if not content:
            return
        lines = content.strip().split("\n")
        if len(lines) > budget:
            lines = lines[:budget] + [f"... ({len(lines) - budget} more lines)"]
        used = len(lines) + 2
        if line_budget <= 0:
            return
        sections.append((priority, title, "\n".join(lines)))
        line_budget -= used

    # 1. File summary (always)
    rows = conn.execute(
        "SELECT file, summary FROM file_summaries WHERE file LIKE ? LIMIT 5", (pattern,)
    ).fetchall()
    if rows:
        content = "\n".join(f"{r['file']}: {r['summary']}" for r in rows)
        add_section("File Summaries", content, priority=1, budget=100)

    # 2. Target function bodies
    bodies = conn.execute(
        "SELECT file, function_name, class_name, start_line, end_line, body, line_count "
        "FROM function_bodies WHERE function_name LIKE ? OR class_name LIKE ? "
        "ORDER BY line_count ASC LIMIT 5",
        (pattern, pattern)
    ).fetchall()
    if bodies:
        parts = []
        for b in bodies:
            cls = f"{b['class_name']}::" if b['class_name'] else ""
            parts.append(f"// {cls}{b['function_name']} ({b['file']}:{b['start_line']}-{b['end_line']})")
            parts.append(b['body'])
        add_section("Target Functions", "\n".join(parts), priority=1, budget=600)

    # 3. Relevant decisions
    decs = conn.execute(
        "SELECT decision, rationale, file FROM decisions "
        "WHERE status = 'active' AND (decision LIKE ? OR file LIKE ? OR function_name LIKE ?) "
        "ORDER BY created_at DESC LIMIT 5",
        (pattern, pattern, pattern)
    ).fetchall()
    if decs:
        content = "\n".join(f"- {d['decision']}" + (f" (Why: {d['rationale']})" if d['rationale'] else "") for d in decs)
        add_section("Relevant Decisions", content, priority=2, budget=200)

    # 4. Function summaries for context
    sums = conn.execute(
        "SELECT file, function_name, class_name, summary FROM function_summaries "
        "WHERE (function_name LIKE ? OR class_name LIKE ? OR file LIKE ?) AND stale = 0 "
        "LIMIT 15",
        (pattern, pattern, pattern)
    ).fetchall()
    if sums:
        content = "\n".join(f"{s['function_name']}: {s['summary']}" for s in sums if s['summary'])
        add_section("Function Summaries", content, priority=3, budget=200)

    if task_type == "bugfix":
        # Callers (1 hop)
        callers = conn.execute(
            "SELECT caller_symbol, caller_file, callee_symbol, confidence "
            "FROM call_edges WHERE callee_symbol LIKE ? OR callee_file LIKE ? "
            "ORDER BY confidence DESC LIMIT 10",
            (pattern, pattern)
        ).fetchall()
        if callers:
            content = "\n".join(f"{c['caller_file']}::{c['caller_symbol']} -> {c['callee_symbol']} (conf={c['confidence']})" for c in callers)
            add_section("Callers (1-hop)", content, priority=3, budget=200)

        # Crash/null risks
        risks = conn.execute(
            "SELECT file, line, risk_type, severity, expression FROM crash_risks "
            "WHERE file LIKE ? ORDER BY severity LIMIT 10", (pattern,)
        ).fetchall()
        null_risks = conn.execute(
            "SELECT file, line, function_call, risk_type FROM null_risks "
            "WHERE file LIKE ? LIMIT 10", (pattern,)
        ).fetchall()
        if risks or null_risks:
            parts = []
            for r in risks:
                parts.append(f"[{r['severity']}] {r['file']}:{r['line']} {r['risk_type']}: {r['expression']}")
            for n in null_risks:
                parts.append(f"[null] {n['file']}:{n['line']} {n['function_call']} ({n['risk_type']})")
            add_section("Risk Analysis", "\n".join(parts), priority=2, budget=200)

        # Related tests
        tests = conn.execute(
            "SELECT test_file, function_name, mapping_type FROM test_function_map "
            "WHERE target_file LIKE ? OR function_name LIKE ? LIMIT 10",
            (pattern, pattern)
        ).fetchall()
        if tests:
            content = "\n".join(f"{t['test_file']} tests {t['function_name']} ({t['mapping_type']})" for t in tests)
            add_section("Related Tests", content, priority=3, budget=200)

    elif task_type == "feature":
        # Related classes
        classes = conn.execute(
            "SELECT name, header, parent_class, project FROM classes "
            "WHERE name LIKE ? OR parent_class LIKE ? LIMIT 10",
            (pattern, pattern)
        ).fetchall()
        if classes:
            content = "\n".join(f"{c['name']} extends {c['parent_class'] or 'none'} ({c['header']})" for c in classes)
            add_section("Related Classes", content, priority=3, budget=200)

        # Config keys
        cfgkeys = conn.execute(
            "SELECT key_name, getter_type, source_file, context FROM config_keys "
            "WHERE source_file LIKE ? OR key_name LIKE ? LIMIT 10",
            (pattern, pattern)
        ).fetchall()
        if cfgkeys:
            content = "\n".join(f"{c['key_name']} ({c['getter_type']}) in {c['source_file']}" for c in cfgkeys)
            add_section("Config Keys", content, priority=4, budget=150)

        # Constants
        consts = conn.execute(
            "SELECT name, value, header, description FROM constants "
            "WHERE name LIKE ? OR description LIKE ? LIMIT 10",
            (pattern, pattern)
        ).fetchall()
        if consts:
            content = "\n".join(f"{c['name']} = {c['value']} ({c['description']})" for c in consts)
            add_section("Constants", content, priority=4, budget=100)

    elif task_type == "refactor":
        # Callers + callees (2 hop)
        callers = conn.execute(
            "SELECT DISTINCT caller_symbol, caller_file FROM call_edges "
            "WHERE callee_symbol LIKE ? OR callee_file LIKE ? LIMIT 15",
            (pattern, pattern)
        ).fetchall()
        callees = conn.execute(
            "SELECT DISTINCT callee_symbol, callee_file FROM call_edges "
            "WHERE caller_symbol LIKE ? OR caller_file LIKE ? LIMIT 15",
            (pattern, pattern)
        ).fetchall()
        if callers or callees:
            parts = ["Callers:"] + [f"  <- {c['caller_file']}::{c['caller_symbol']}" for c in callers]
            parts += ["Callees:"] + [f"  -> {c['callee_file']}::{c['callee_symbol']}" for c in callees]
            add_section("Call Graph (2-hop)", "\n".join(parts), priority=2, budget=300)

        # Dead methods
        dead = conn.execute(
            "SELECT class_name, method_name, header_file FROM dead_methods "
            "WHERE class_name LIKE ? OR header_file LIKE ? LIMIT 10",
            (pattern, pattern)
        ).fetchall()
        if dead:
            content = "\n".join(f"{d['class_name']}::{d['method_name']} ({d['header_file']})" for d in dead)
            add_section("Dead Methods", content, priority=4, budget=100)

        # Duplicates
        dupes = conn.execute(
            "SELECT file_a, line_a, file_b, line_b, line_count, preview FROM duplicate_blocks "
            "WHERE file_a LIKE ? OR file_b LIKE ? LIMIT 5",
            (pattern, pattern)
        ).fetchall()
        if dupes:
            content = "\n".join(f"{d['file_a']}:{d['line_a']} == {d['file_b']}:{d['line_b']} ({d['line_count']} lines)" for d in dupes)
            add_section("Duplicate Code", content, priority=4, budget=100)

    elif task_type == "audit":
        coverage = conn.execute(
            "SELECT file, coverage_score, unit_test_refs, crash_risk_count, null_risk_count, semantic_risk_max, notes "
            "FROM audit_coverage WHERE file LIKE ? ORDER BY coverage_score ASC, semantic_risk_max DESC LIMIT 12",
            (pattern,)
        ).fetchall()
        if coverage:
            content = "\n".join(
                f"{row['file']}: cov={row['coverage_score']} tests={row['unit_test_refs']} crash={row['crash_risk_count']} null={row['null_risk_count']} semrisk={row['semantic_risk_max']}"
                for row in coverage
            )
            add_section("Coverage Snapshot", content, priority=2, budget=180)

        ct_symbols = conn.execute(
            "SELECT symbol_name, file_path, review_priority, risk_score, audit_coverage_score FROM symbol_metadata "
            "WHERE (symbol_name LIKE ? OR file_path LIKE ?) AND ct_sensitive = 1 "
            "ORDER BY review_priority DESC, risk_score DESC LIMIT 12",
            (pattern, pattern)
        ).fetchall()
        if ct_symbols:
            content = "\n".join(
                f"{row['symbol_name']} @ {row['file_path']} | prio={row['review_priority']} risk={row['risk_score']} cov={row['audit_coverage_score']}"
                for row in ct_symbols
            )
            add_section("CT-Sensitive Symbols", content, priority=2, budget=160)

        review = conn.execute(
            "SELECT file, queue_type, priority_score, rationale FROM review_queue WHERE file LIKE ? OR rationale LIKE ? "
            "ORDER BY priority_score DESC LIMIT 12",
            (pattern, pattern)
        ).fetchall()
        if review:
            content = "\n".join(
                f"{row['file']} [{row['queue_type']}] prio={row['priority_score']} :: {row['rationale']}"
                for row in review
            )
            add_section("Review Queue", content, priority=3, budget=160)

    elif task_type == "optimize":
        bottlenecks = conn.execute(
            "SELECT symbol_name, file_path, overall_priority, perf_priority, hotness_score, gpu_score, audit_gap_score, summary "
            "FROM v_bottleneck_queue WHERE symbol_name LIKE ? OR file_path LIKE ? OR summary LIKE ? "
            "ORDER BY overall_priority DESC, perf_priority DESC LIMIT 12",
            (pattern, pattern, pattern)
        ).fetchall()
        if bottlenecks:
            content = "\n".join(
                f"{row['symbol_name']} @ {row['file_path']} | overall={row['overall_priority']} perf={row['perf_priority']} hot={row['hotness_score']} gpu={row['gpu_score']} gap={row['audit_gap_score']}"
                for row in bottlenecks
            )
            add_section("Bottleneck Queue", content, priority=2, budget=180)

        hotspots = conn.execute(
            "SELECT entity_name, file, gain_score, risk_score, hot_path_candidate, optimization_candidate, fragile_surface "
            "FROM semantic_profiles WHERE entity_name LIKE ? OR file LIKE ? "
            "ORDER BY gain_score DESC, risk_score DESC LIMIT 12",
            (pattern, pattern)
        ).fetchall()
        if hotspots:
            content = "\n".join(
                f"{row['entity_name']} @ {row['file']} | gain={row['gain_score']} risk={row['risk_score']} hot={row['hot_path_candidate']} opt={row['optimization_candidate']} fragile={row['fragile_surface']}"
                for row in hotspots
            )
            add_section("Optimization Hotspots", content, priority=3, budget=160)

    elif task_type == "explore":
        candidate_files = _collect_candidate_files(conn, term, limit=8)
        if candidate_files:
            content = "\n".join(_compact_file_snapshot(conn, file_name) for file_name in candidate_files)
            add_section("Primary Files", content, priority=2, budget=120)

        top_symbols = _iter_candidate_symbols(conn, term, candidate_files, limit=12)
        if top_symbols:
            content = "\n".join(
                f"{row['symbol_name']} @ {row['file_path']} | prio={row['review_priority']} gain={row['gain_score']} risk={row['risk_score']} cov={row['audit_coverage_score']}"
                for row in top_symbols
            )
            add_section("Top Symbols", content, priority=2, budget=160)

        neighbor_files = _build_neighbor_file_scores(conn, candidate_files, limit=10)
        if neighbor_files:
            content = "\n".join(
                f"{file_name} | call_edges={score['call_edges']} dep_edges={score['dep_edges']} call_weight={score['call_weight']}"
                for file_name, score in neighbor_files
            )
            add_section("Neighbor Files", content, priority=3, budget=140)

    else:
        print(f"  Unsupported bundle type '{task_type}'. Use bugfix, feature, refactor, audit, optimize, or explore.")
        return

    # Dependencies
    deps = conn.execute(
        "SELECT source_file, target_file, dep_type FROM dependencies "
        "WHERE source_file LIKE ? OR target_file LIKE ? LIMIT 15",
        (pattern, pattern)
    ).fetchall()
    if deps:
        content = "\n".join(f"{d['source_file']} -> {d['target_file']} ({d['dep_type']})" for d in deps)
        add_section("Dependencies", content, priority=5, budget=200)

    # Output
    if not sections:
        print(f"  No context found for '{term}'. Try a different search term.")
        return

    sections.sort(key=lambda x: x[0])
    print(f"# Context Bundle: {task_type} — {term}")
    print(f"# Budget: {max_lines} lines\n")
    for _, title, content in sections:
        print(f"## {title}")
        print(content)
        print()


def pipeline_cmd(task_type, term, budget=48, core_only=False, json_mode=True):
    """Run focus -> slice -> bundle in a single call.

    Usage: pipeline <task_type> <term> [budget] [--core] [--json]
    task_type: bugfix | optimize | audit | explore
    """
    gv = get_graph_view()
    manifest = gv.pipeline_manifest(task_type, term, budget=budget,
                                    core_only=core_only)
    if json_mode:
        import json as _json
        print(_json.dumps(manifest, indent=2))
    else:
        # Compact text fallback
        s = manifest.get("summary", {})
        print(f"Pipeline: {task_type} / {term}")
        print(f"  Seeds:    {s.get('seed_count', 0)}")
        print(f"  Slice:    {s.get('slice_nodes', 0)} nodes, {s.get('slice_edges', 0)} edges")
        print(f"  Sections: {s.get('bundle_sections', 0)}")
        print(f"  Files:    {len(s.get('files', []))}")
        suspects = s.get("top_suspects", [])
        if suspects:
            print(f"  Top:      {', '.join(suspects[:6])}")
        risks = s.get("risks", [])
        if risks:
            print(f"  Risks:    {len(risks)}")
            for r in risks[:5]:
                sym = r.get("symbol", r.get("source", "?"))
                print(f"    - {sym} ({r.get('file', '?')})")
        constraints = s.get("constraints", [])
        if constraints:
            print(f"  Constraints:")
            for c in constraints[:4]:
                print(f"    - {c}")

        # Print phase details in text mode
        phases = manifest.get("phases", {})

        focus = phases.get("focus", {})
        if focus.get("suspects"):
            print(f"\n--- Focus ({len(focus['suspects'])} suspects) ---")
            for su in focus["suspects"][:8]:
                print(f"  {su['symbol']}  {su['file']}  score={su.get('composite_score', '?')}")

        slc = phases.get("slice", {})
        if slc.get("signatures"):
            print(f"\n--- Slice ({len(slc.get('nodes', []))} nodes) ---")
            for sig in slc["signatures"][:6]:
                print(f"  {sig['symbol']}: {sig.get('signature', '?')}")

        bndl = phases.get("bundle", {})
        if bndl.get("sections"):
            print(f"\n--- Bundle ({len(bndl['sections'])} sections) ---")
            for sec in bndl["sections"]:
                print(f"  [{sec['title']}] {len(sec.get('items', []))} items")


def claudemd_cmd():
    """Generate CLAUDE.md project documentation from database."""
    conn = get_conn()

    project_name = "Project"
    if CONFIG_PATH:
        try:
            cfg = _parse_toml(CONFIG_PATH)
            project_name = cfg.get("project", {}).get("name", REPO_ROOT.name)
        except Exception:
            project_name = REPO_ROOT.name
    else:
        project_name = REPO_ROOT.name

    # Stats
    file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    total_lines = conn.execute("SELECT COALESCE(SUM(line_count), 0) FROM file_lines").fetchone()[0]
    func_count = conn.execute("SELECT COUNT(*) FROM function_index").fetchone()[0]
    class_count = conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
    singleton_count = conn.execute("SELECT COUNT(*) FROM singletons").fetchone()[0]

    print(f"# {project_name} — AI Context Guide")
    print(f"\n## Project Overview")
    print(f"- **{file_count}** source files, **{total_lines:,}** lines of code")
    print(f"- **{func_count}** functions, **{class_count}** classes, **{singleton_count}** singletons")

    # Source dirs
    print(f"- Source directories: {', '.join(label for label, _, _ in SOURCE_DIRS)}")
    print()

    # Key Singletons
    singletons = conn.execute(
        "SELECT macro, class_name, header, category, description FROM singletons "
        "ORDER BY category, macro LIMIT 30"
    ).fetchall()
    if singletons:
        print("## Key Singletons")
        current_cat = None
        for s in singletons:
            if s['category'] != current_cat:
                current_cat = s['category']
                print(f"\n### {current_cat or 'other'}")
            print(f"- `{s['macro']}` — {s['description'] or s['class_name']} (`{s['header']}`)")
        print()

    # File map by category
    cats = conn.execute(
        "SELECT category, COUNT(*) as cnt FROM files GROUP BY category ORDER BY cnt DESC"
    ).fetchall()
    if cats:
        print("## Source File Categories")
        for c in cats:
            print(f"- **{c['category'] or 'uncategorized'}**: {c['cnt']} files")
        print()

    # Naming conventions from category rules
    print("## Naming Conventions")
    for rule in CATEGORY_RULES[:15]:
        match = rule['match']
        pattern = rule['pattern']
        category = rule['category']
        source = f" (in {rule['source_dir']})" if rule.get('source_dir') else ""
        if match == "prefix":
            print(f"- Files starting with `{pattern}` → **{category}**{source}")
        elif match == "suffix":
            print(f"- Files ending with `{pattern}` → **{category}**{source}")
        elif match == "contains":
            print(f"- Files containing `{pattern}` → **{category}**{source}")
    print()

    # Key constants
    consts = conn.execute(
        "SELECT name, value, description FROM constants WHERE category = 'limits' ORDER BY name LIMIT 15"
    ).fetchall()
    if consts:
        print("## Key Constants")
        for c in consts:
            print(f"- `{c['name']}` = {c['value']} — {c['description']}")
        print()

    # Known issues
    crash_count = conn.execute("SELECT COUNT(*) FROM crash_risks").fetchone()[0]
    null_count = conn.execute("SELECT COUNT(*) FROM null_risks").fetchone()[0]
    dead_count = conn.execute("SELECT COUNT(*) FROM dead_methods").fetchone()[0]
    todo_count = conn.execute("SELECT COUNT(*) FROM todos").fetchone()[0]
    leak_count = conn.execute("SELECT COUNT(*) FROM leak_risks WHERE risk_score > 0").fetchone()[0]

    print("## Known Issues")
    print(f"- {crash_count} crash risks")
    print(f"- {null_count} unchecked pointer dereferences")
    print(f"- {dead_count} dead methods")
    print(f"- {leak_count} potential memory leaks")
    print(f"- {todo_count} TODO/FIXME comments")
    print()

    # Usage guide
    print("## Using source_graph.py")
    print("```bash")
    print("python source_graph.py find <term>         # Full-text search")
    print("python source_graph.py func <name>         # Find function with line ranges")
    print("python source_graph.py body <name>         # Print function source code")
    print("python source_graph.py context <file>      # Unified context for a file")
    print("python source_graph.py focus <term> [budget] [--core] # Compact ranked snapshot")
    print("python source_graph.py slice <term> [budget] [--core] # Minimal dependency/call slice")
    print("python source_graph.py trace <term>        # Build minimal code trace")
    print("python source_graph.py bundle bugfix <term> # Assemble bugfix context")
    print("python source_graph.py bundle feature <term># Assemble feature context")
    print("python source_graph.py bundle audit <term> # Assemble audit context")
    print("python source_graph.py bundle optimize <term># Assemble optimization context")
    print("python source_graph.py impact <term>       # Show change impact analysis")
    print("python source_graph.py decide \"<text>\" --why \"<reason>\"  # Log decision")
    print("python source_graph.py decisions [term]    # Query decision log")
    print("python source_graph.py build --incremental # Fast rebuild (changed files only)")
    print("```")


def init_cmd():
    """Generate a starter source_graph.toml by scanning the project."""
    config_path = SCRIPT_DIR / "source_graph.toml"
    if config_path.exists():
        print(f"[!] Config already exists: {config_path}")
        print("    Delete it first to regenerate.")
        return

    lines = []
    lines.append("[project]")
    lines.append(f'name = "{REPO_ROOT.name}"')
    lines.append('language = "cpp"')
    lines.append("")

    # Detect source dirs
    for subdir in sorted(REPO_ROOT.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith(".") or subdir.name in ("tools", "build", "bin", "obj", "Debug", "Release", "x64", ".vs", "packages"):
            continue
        cpp_count = len(list(subdir.glob("*.cpp"))) + len(list(subdir.glob("*.h")))
        if cpp_count > 0:
            lines.append("[[source_dirs]]")
            lines.append(f'label = "{subdir.name}"')
            lines.append(f'path = "{subdir.name}"')
            lines.append('extensions = ["*.cpp", "*.h"]')
            lines.append("")

    # Category rules
    lines.append("# File categorization rules")
    for rule in _DEFAULT_CATEGORY_RULES[:10]:
        lines.append("[[category_rules]]")
        lines.append(f'match = "{rule["match"]}"')
        lines.append(f'pattern = "{rule["pattern"]}"')
        lines.append(f'category = "{rule["category"]}"')
        if rule.get("source_dir"):
            lines.append(f'source_dir = "{rule["source_dir"]}"')
        lines.append("")

    content = "\n".join(lines) + "\n"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Generated: {config_path}")
    print(f"    Edit this file to add singletons, events, constants, etc.")


def export_config_cmd():
    """Export all hardcoded seed data to source_graph.toml format."""
    lines = []
    lines.append("[project]")
    lines.append(f'name = "{REPO_ROOT.name}"')
    lines.append('language = "cpp"')
    lines.append("")

    # Source dirs
    for label, path, exts in SOURCE_DIRS:
        lines.append("[[source_dirs]]")
        lines.append(f'label = "{label}"')
        try:
            rel = path.relative_to(REPO_ROOT)
            lines.append(f'path = "{str(rel).replace(chr(92), "/")}"')
        except ValueError:
            lines.append(f'path = "{str(path).replace(chr(92), "/")}"')
        lines.append(f'extensions = {json.dumps(exts)}')
        lines.append("")

    # External paths
    if EXTERNAL_PATHS:
        lines.append("[external_paths]")
        for key, val in EXTERNAL_PATHS.items():
            lines.append(f'{key} = "{str(val).replace(chr(92), "/")}"')
        lines.append("")
    else:
        lines.append("[external_paths]")
        lines.append(f'runtime_data = "{str(RUNTIME_DATA_ROOT).replace(chr(92), "/")}"')
        lines.append(f'research_dir = "ReversingResearch"')
        lines.append("")

    # Category rules
    for rule in CATEGORY_RULES:
        lines.append("[[category_rules]]")
        lines.append(f'match = "{rule["match"]}"')
        lines.append(f'pattern = "{rule["pattern"]}"')
        lines.append(f'category = "{rule["category"]}"')
        if rule.get("source_dir"):
            lines.append(f'source_dir = "{rule["source_dir"]}"')
        lines.append("")

    # Export seed data from the DB if it exists
    if DB_PATH.exists():
        conn = get_conn()

        # Singletons
        rows = conn.execute("SELECT macro, class_name, header, project, category, description FROM singletons ORDER BY category, macro").fetchall()
        for r in rows:
            lines.append("[[singletons]]")
            lines.append(f'macro = "{r["macro"]}"')
            lines.append(f'class_name = "{r["class_name"]}"')
            lines.append(f'header = "{r["header"]}"')
            lines.append(f'project = "{r["project"]}"')
            if r["category"]:
                lines.append(f'category = "{r["category"]}"')
            if r["description"]:
                lines.append(f'description = "{r["description"]}"')
            lines.append("")

        # Player files
        rows = conn.execute("SELECT file, domain, key_methods, description FROM player_files ORDER BY domain").fetchall()
        for r in rows:
            lines.append("[[player_files]]")
            lines.append(f'file = "{r["file"]}"')
            lines.append(f'domain = "{r["domain"]}"')
            if r["key_methods"]:
                lines.append(f'key_methods = "{r["key_methods"]}"')
            if r["description"]:
                lines.append(f'description = "{r["description"]}"')
            lines.append("")

        # Events
        rows = conn.execute("SELECT name, singleton_macro, cpp_file, header_file, def_file, ai_file, description FROM events ORDER BY name").fetchall()
        for r in rows:
            lines.append("[[events]]")
            lines.append(f'name = "{r["name"]}"')
            for field in ("singleton_macro", "cpp_file", "header_file", "def_file", "ai_file", "description"):
                if r[field]:
                    lines.append(f'{field} = "{r[field]}"')
            lines.append("")

        # AI handlers
        rows = conn.execute("SELECT file, target, description FROM ai_handlers ORDER BY file").fetchall()
        for r in rows:
            lines.append("[[ai_handlers]]")
            lines.append(f'file = "{r["file"]}"')
            lines.append(f'target = "{r["target"]}"')
            if r["description"]:
                lines.append(f'description = "{r["description"]}"')
            lines.append("")

        # Inventory scripts
        rows = conn.execute("SELECT file, description FROM inventory_scripts ORDER BY file").fetchall()
        for r in rows:
            lines.append("[[inventory_scripts]]")
            lines.append(f'file = "{r["file"]}"')
            if r["description"]:
                lines.append(f'description = "{r["description"]}"')
            lines.append("")

        # Constants
        rows = conn.execute("SELECT name, value, header, category, description FROM constants ORDER BY category, name").fetchall()
        for r in rows:
            lines.append("[[constants]]")
            lines.append(f'name = "{r["name"]}"')
            if r["value"]:
                lines.append(f'value = "{r["value"]}"')
            lines.append(f'header = "{r["header"]}"')
            if r["category"]:
                lines.append(f'category = "{r["category"]}"')
            if r["description"]:
                lines.append(f'description = "{r["description"]}"')
            lines.append("")

        # Config files
        rows = conn.execute("SELECT path, description FROM config_files ORDER BY path").fetchall()
        for r in rows:
            lines.append("[[config_files]]")
            lines.append(f'path = "{r["path"]}"')
            if r["description"]:
                lines.append(f'description = "{r["description"]}"')
            lines.append("")

        # Packet handlers
        rows = conn.execute("SELECT headcode_name, headcode_value, headcode_hex, handler_method, source_file, handler_type, category, description FROM packet_handlers ORDER BY handler_type, headcode_value").fetchall()
        for r in rows:
            lines.append("[[packet_handlers]]")
            lines.append(f'headcode_name = "{r["headcode_name"]}"')
            if r["headcode_value"] is not None:
                lines.append(f'headcode_value = {r["headcode_value"]}')
            for field in ("headcode_hex", "handler_method", "source_file", "handler_type", "category", "description"):
                if r[field]:
                    lines.append(f'{field} = "{r[field]}"')
            lines.append("")

        conn.close()

    content = "\n".join(lines) + "\n"
    out_path = SCRIPT_DIR / "source_graph.toml"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Exported config to: {out_path}")
    print(f"    {len(lines)} lines written")


# ============================================================
# MAIN
# ============================================================

def main():
    global _QUIET
    # Pre-scan for --quiet / -q — must happen before load_config()
    if "--quiet" in sys.argv or "-q" in sys.argv:
        _QUIET = True
        sys.argv = [a for a in sys.argv if a not in ("--quiet", "-q")]

    # Handle --project <dir> before anything else — it redirects the tool
    # to a different project's config/DB without copying source_graph.py.
    if "--project" in sys.argv:
        idx = sys.argv.index("--project")
        if idx + 1 < len(sys.argv):
            _apply_project_dir(sys.argv[idx + 1])
            # Remove --project <dir> from argv so it doesn't confuse commands
            del sys.argv[idx:idx + 2]
        else:
            print("[!] --project requires a directory argument")
            sys.exit(1)

    load_config()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "build":
        best_effort = "--best-effort" in sys.argv
        try:
            if "--incremental" in sys.argv or "-i" in sys.argv:
                build_incremental()
            else:
                build()
        except Exception as exc:
            if best_effort:
                print(f"[!] Source graph build failed (non-fatal): {exc}")
            else:
                raise
    elif cmd == "find" and len(sys.argv) > 2:
        find_cmd(" ".join(sys.argv[2:]))
    elif (cmd in ("where", "def")) and len(sys.argv) > 2:
        where_cmd(" ".join(sys.argv[2:]))
    elif cmd == "singleton" and len(sys.argv) > 2:
        singleton_cmd(sys.argv[2])
    elif cmd == "file" and len(sys.argv) > 2:
        file_cmd(sys.argv[2])
    elif cmd == "handler" and len(sys.argv) > 2:
        handler_cmd(sys.argv[2])
    elif cmd == "event" and len(sys.argv) > 2:
        event_cmd(sys.argv[2])
    elif cmd == "method" and len(sys.argv) > 2:
        method_cmd(sys.argv[2])
    elif cmd == "player" and len(sys.argv) > 2:
        player_cmd(sys.argv[2])
    elif cmd == "ai" and len(sys.argv) > 2:
        ai_cmd(sys.argv[2])
    elif cmd == "const" and len(sys.argv) > 2:
        const_cmd(sys.argv[2])
    elif cmd == "config" and len(sys.argv) > 2:
        config_cmd(sys.argv[2])
    elif cmd == "deps" and len(sys.argv) > 2:
        deps_cmd(sys.argv[2])
    elif cmd == "enum" and len(sys.argv) > 2:
        enum_cmd(sys.argv[2])
    elif cmd == "struct" and len(sys.argv) > 2:
        struct_cmd(sys.argv[2])
    elif cmd == "class" and len(sys.argv) > 2:
        class_cmd(sys.argv[2])
    elif cmd == "todo":
        todo_cmd(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "dbtable" and len(sys.argv) > 2:
        dbtable_cmd(sys.argv[2])
    elif cmd == "query" and len(sys.argv) > 2:
        query_cmd(sys.argv[2])
    elif cmd == "cfgkey" and len(sys.argv) > 2:
        cfgkey_cmd(sys.argv[2])
    elif cmd == "define" and len(sys.argv) > 2:
        define_cmd(sys.argv[2])
    elif cmd == "complexity":
        complexity_cmd()
    elif cmd == "leaks":
        leaks_cmd()
    elif cmd == "nullrisks":
        nullrisks_cmd(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "rawptrs":
        rawptrs_cmd(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "casts":
        casts_cmd(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "crashes":
        crashes_cmd(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "looprisks":
        looprisks_cmd(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "deadmethods":
        deadmethods_cmd()
    elif cmd == "duplicates":
        duplicates_cmd()
    elif cmd == "context" and len(sys.argv) > 2:
        context_cmd(sys.argv[2])
    elif cmd == "func" and len(sys.argv) > 2:
        func_cmd(sys.argv[2])
    elif cmd == "api" and len(sys.argv) > 2:
        lim = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        api_cmd(sys.argv[2], limit=lim)
    elif cmd == "gaps":
        gaps_cmd()
    elif cmd == "summary":
        summary_cmd()
    elif cmd == "stats":
        stats_cmd()
    elif cmd == "tags" and len(sys.argv) > 2:
        tags_cmd(" ".join(sys.argv[2:]))
    elif cmd == "hotspots":
        hotspots_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "coverage":
        coverage_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "churn":
        churn_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "reviewqueue":
        reviewqueue_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "ownership":
        ownership_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "testmap":
        testmap_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "calls" and len(sys.argv) > 2:
        calls_cmd(" ".join(sys.argv[2:]))
    elif cmd == "focus" and len(sys.argv) > 2:
        term, budget, core_only, json_mode, mode = _parse_compact_args(sys.argv[2:], 80)
        focus_cmd(term, budget, core_only=core_only, json_mode=json_mode, mode=mode)
    elif cmd == "slice" and len(sys.argv) > 2:
        term, budget, core_only, json_mode, mode = _parse_compact_args(sys.argv[2:], 120)
        slice_cmd(term, budget, core_only=core_only, json_mode=json_mode, mode=mode)
    elif cmd == "trace" and len(sys.argv) > 2:
        trace_cmd(" ".join(sys.argv[2:]))
    elif cmd == "impact" and len(sys.argv) > 2:
        impact_cmd(" ".join(sys.argv[2:]))
    elif cmd == "research":
        research_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "symbols":
        symbols_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "bottlenecks":
        bottlenecks_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "auditmap":
        auditmap_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "aitasks":
        aitasks_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "validateai":
        validateai_cmd(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "sql" and len(sys.argv) > 2:
        sql_cmd(" ".join(sys.argv[2:]))
    elif cmd == "body" and len(sys.argv) > 2:
        body_cmd(sys.argv[2])
    elif cmd == "bodygrep" and len(sys.argv) > 2:
        bodygrep_cmd(" ".join(sys.argv[2:]))
    elif cmd == "summarize":
        summarize_cmd(sys.argv[2:])
    elif cmd == "decide" and len(sys.argv) > 2:
        decide_cmd(sys.argv[2:])
    elif cmd == "decisions":
        decisions_cmd(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "bundle" and len(sys.argv) > 3:
        max_l = 2000
        json_mode = "--json" in sys.argv
        if "--max-lines" in sys.argv:
            idx = sys.argv.index("--max-lines")
            if idx + 1 < len(sys.argv):
                max_l = int(sys.argv[idx + 1])
        bundle_cmd(sys.argv[2], sys.argv[3], max_l, json_mode=json_mode)
    elif cmd == "pipeline" and len(sys.argv) > 3:
        task_type = sys.argv[2]
        # Filter --text before passing to _parse_compact_args (which doesn't know it)
        raw_args = [a for a in sys.argv[3:] if a != "--text"]
        term, budget, core_only, json_mode_flag, _ = _parse_compact_args(raw_args, 48)
        # Pipeline defaults to JSON unless --text is explicitly given
        json_mode = "--text" not in sys.argv
        if json_mode_flag:
            json_mode = True
        pipeline_cmd(task_type, term, budget=int(budget), core_only=core_only, json_mode=json_mode)
    elif cmd == "claudemd":
        claudemd_cmd()
    elif cmd == "init":
        init_cmd()
    elif cmd == "export-config":
        export_config_cmd()
    elif cmd == "dataflow":
        dataflow_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "invariants":
        invariants_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "invcheck":
        invcheck_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "backendmap":
        backendmap_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "exploitpath":
        exploitpath_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "hotpath":
        hotpath_cmd(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
