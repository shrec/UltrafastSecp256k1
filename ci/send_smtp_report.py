#!/usr/bin/env python3
"""Send a plain-text SMTP report from generated subject/body files."""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


def _env(name: str, default: str = '') -> str:
    return os.environ.get(name, default).strip()


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _read_required(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f'missing {label}: {path}')
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError(f'empty {label}: {path}')
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-file', type=Path, required=True)
    parser.add_argument('--body-file', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs without sending mail.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject = _read_required(args.subject_file, 'subject file')
    body = _read_required(args.body_file, 'body file')

    if args.dry_run:
        print(f'dry-run: subject={subject!r} body_chars={len(body)}')
        return 0

    host = _env('RESEARCH_SMTP_HOST')
    port = _env('RESEARCH_SMTP_PORT', '587')
    username = _env('RESEARCH_SMTP_USERNAME')
    password = _env('RESEARCH_SMTP_PASSWORD')
    sender = _env('RESEARCH_REPORT_FROM')
    recipients = [item.strip() for item in _env('RESEARCH_REPORT_TO').split(',') if item.strip()]
    reply_to = _env('RESEARCH_REPORT_REPLY_TO')
    use_ssl = _parse_bool(_env('RESEARCH_SMTP_USE_SSL', 'false'))

    missing = []
    for key, value in [
        ('RESEARCH_SMTP_HOST', host),
        ('RESEARCH_SMTP_PORT', port),
        ('RESEARCH_SMTP_USERNAME', username),
        ('RESEARCH_SMTP_PASSWORD', password),
        ('RESEARCH_REPORT_FROM', sender),
        ('RESEARCH_REPORT_TO', ','.join(recipients)),
    ]:
        if not value:
            missing.append(key)
    if missing:
        raise RuntimeError('missing required SMTP configuration: ' + ', '.join(missing))

    message = EmailMessage()
    message['Subject'] = subject
    message['From'] = sender
    message['To'] = ', '.join(recipients)
    if reply_to:
        message['Reply-To'] = reply_to
    message.set_content(body)

    port_num = int(port)
    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port_num, context=context, timeout=30) as server:
            server.login(username, password)
            server.send_message(message)
        return 0

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port_num, timeout=30) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(username, password)
        server.send_message(message)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise