#!/usr/bin/env python3
"""
gmail_export_to_tsv.py

Fetch messages from Gmail via IMAP and export to a TSV (tab-separated) file with columns:
to, from, subject, body

Requirements:
 - IMAP enabled in Gmail
 - Use an App Password or OAuth2 (App Password shown here)
 - Set EMAIL_ADDR and EMAIL_PASS env vars or edit below

Notes:
 - Body is converted to plain text and collapsed to a single line (no newlines or tabs).
 - Each message is written to the TSV as it's fetched (incremental writes).
"""

import imaplib
import email
from email.header import decode_header
import csv
import os
import sys
import re
import html
from typing import Optional
import email.message

# ---------- Configuration ----------
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

EMAIL = os.getenv("EMAIL_ADDR", "k.ganeshgiri@gmail.com")    # e.g. "your.email@gmail.com"
PASSWORD = os.getenv("EMAIL_PASS", "nbbn mzlp xoee cayd") # app password or OAuth token
OUTPUT_TSV = "gmail_export.tsv"
MAILBOX = "INBOX"   # or "[Gmail]/All Mail"
FLUSH_EVERY = 50    # fsync every N rows (set 1 to flush every row)
PREVIEW_ROWS = 10   # how many rows to show in the terminal preview
# -----------------------------------

def decode_mime_words(s: Optional[str]) -> str:
    if not s:
        return ""
    parts = decode_header(s)
    decoded = []
    for text, encoding in parts:
        if isinstance(text, bytes):
            try:
                decoded.append(text.decode(encoding or "utf-8", errors="replace"))
            except Exception:
                decoded.append(text.decode("utf-8", errors="replace"))
        else:
            decoded.append(text)
    return "".join(decoded)

def html_to_plain_text(html_text: str) -> str:
    """Lightweight HTML -> text. Then return a single-line string with collapsed whitespace."""
    if not html_text:
        return ""
    # Unescape HTML entities first
    html_text = html.unescape(html_text)

    # Remove script/style blocks
    html_text = re.sub(r'(?is)<(script|style).*?>.*?(</\1>)', '', html_text)
    # Replace <br>, <p>, <div>, <li> with spaces/newlines to separate blocks
    html_text = re.sub(r'(?i)<br\s*/?>', ' ', html_text)
    html_text = re.sub(r'(?i)</p\s*>', ' ', html_text)
    html_text = re.sub(r'(?i)</div\s*>', ' ', html_text)
    html_text = re.sub(r'(?i)</li\s*>', ' ', html_text)
    # Remove remaining tags
    text = re.sub(r'<[^>]+>', '', html_text)
    # Normalize whitespace (convert sequences of whitespace to single space)
    text = re.sub(r'[\r\n\t]+', ' ', text)    # remove newlines and tabs
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def get_first_text_part(msg: email.message.Message) -> str:
    """Return best attempt at text body (single-line)."""
    if msg.is_multipart():
        # prefer text/plain, else use text/html
        text_parts = []
        html_parts = []
        for part in msg.walk():
            ctype = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition") or "")
            if ctype == "text/plain" and "attachment" not in content_disposition:
                try:
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue
                    charset = part.get_content_charset() or "utf-8"
                    text_parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    try:
                        text_parts.append(part.get_payload(decode=True).decode("utf-8", errors="replace"))
                    except Exception:
                        pass
            elif ctype == "text/html" and "attachment" not in content_disposition:
                try:
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue
                    charset = part.get_content_charset() or "utf-8"
                    html_parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    pass
        if text_parts:
            plain = "\n\n".join(text_parts)
        elif html_parts:
            plain = html_to_plain_text("\n\n".join(html_parts))
        else:
            plain = ""
    else:
        ctype = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if not payload:
            return ""
        charset = msg.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, errors="replace")
        except Exception:
            try:
                text = payload.decode("utf-8", errors="replace")
            except Exception:
                text = str(payload)
        if ctype == "text/html":
            plain = html_to_plain_text(text)
        else:
            plain = text

    # Ensure single-line: strip and collapse whitespace and remove tabs
    plain = re.sub(r'[\r\n\t]+', ' ', plain)
    plain = re.sub(r'\s{2,}', ' ', plain)
    return plain.strip()

def connect_imap(email_addr: str, password: str) -> imaplib.IMAP4_SSL:
    imap = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
    imap.login(email_addr, password)
    return imap

def fetch_all_message_ids(imap: imaplib.IMAP4_SSL, mailbox: str = "INBOX"):
    status, _ = imap.select(mailbox)
    if status != "OK":
        raise RuntimeError(f"Unable to select mailbox {mailbox}: {status}")
    status, data = imap.search(None, "ALL")
    if status != "OK":
        raise RuntimeError("Failed to search mailbox")
    id_list = data[0].split()
    return id_list

def safe_header(msg: email.message.Message, name: str) -> str:
    return decode_mime_words(msg.get(name) or "")

def export_to_tsv_incremental(email_addr: str, password: str, output_tsv: str = OUTPUT_TSV, mailbox: str = MAILBOX):
    imap = connect_imap(email_addr, password)
    preview_rows = []
    try:
        id_list = fetch_all_message_ids(imap, mailbox=mailbox)
        total = len(id_list)
        print(f"Found {total} messages in {mailbox}")

        # Open TSV for writing (overwrite existing). Use csv with delimiter '\t'.
        with open(output_tsv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["to", "from", "subject", "body"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            rows_written = 0

            for idx, msg_id in enumerate(id_list, start=1):
                try:
                    status, msg_data = imap.fetch(msg_id, "(RFC822)")
                    if status != "OK" or not msg_data or not msg_data[0]:
                        print(f"[{idx}/{total}] Warning: failed to fetch message id {msg_id.decode() if isinstance(msg_id, bytes) else msg_id}")
                        continue
                    raw_email = msg_data[0][1]
                    try:
                        msg = email.message_from_bytes(raw_email)
                    except Exception:
                        msg = email.message_from_string(raw_email.decode("utf-8", errors="replace"))

                    hdr_from = safe_header(msg, "From")
                    hdr_to = safe_header(msg, "To")
                    hdr_subject = safe_header(msg, "Subject")
                    body = get_first_text_part(msg)

                    # Ensure no tab characters in fields (tabs are delimiter)
                    hdr_from = hdr_from.replace('\t', ' ')
                    hdr_to = hdr_to.replace('\t', ' ')
                    hdr_subject = hdr_subject.replace('\t', ' ')
                    body = body.replace('\t', ' ')

                    writer.writerow({
                        "to": hdr_to,
                        "from": hdr_from,
                        "subject": hdr_subject,
                        "body": body
                    })
                    rows_written += 1

                    # collect preview samples
                    if len(preview_rows) < PREVIEW_ROWS:
                        preview_rows.append({
                            "to": hdr_to,
                            "from": hdr_from,
                            "subject": hdr_subject,
                            "body": (body[:200] + '...') if len(body) > 200 else body
                        })

                    # Flush every FLUSH_EVERY rows
                    if FLUSH_EVERY and (rows_written % FLUSH_EVERY == 0):
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            pass

                    # progress
                    if idx % 200 == 0 or idx == total:
                        print(f"[{idx}/{total}] processed, {rows_written} rows written")

                except imaplib.IMAP4.error as im_e:
                    print(f"[{idx}/{total}] IMAP error for id {msg_id}: {im_e} — skipping")
                    continue
                except Exception as e:
                    print(f"[{idx}/{total}] Error processing id {msg_id}: {e} — skipping")
                    continue

            # final flush
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass

        print(f"\nExport complete. {rows_written} messages written to {output_tsv}\n")

        # Print tabular preview
        if preview_rows:
            print("Preview (first {} rows):".format(len(preview_rows)))
            # compute column widths
            cols = ["to", "from", "subject", "body"]
            col_widths = {c: max(len(c), max((len(r[c]) for r in preview_rows), default=0)) for c in cols}
            # cap widths for body/subject to keep terminal readable
            cap = {"subject": 50, "body": 80}
            for c in cap:
                col_widths[c] = min(col_widths[c], cap[c])

            # header
            header = " | ".join(c.ljust(col_widths[c]) for c in cols)
            print(header)
            print("-" * len(header))
            for r in preview_rows:
                row_str = []
                for c in cols:
                    val = r[c]
                    if len(val) > col_widths[c]:
                        val = val[:col_widths[c]-3] + "..."
                    row_str.append(val.ljust(col_widths[c]))
                print(" | ".join(row_str))

    finally:
        try:
            imap.logout()
        except Exception:
            pass

if __name__ == "__main__":
    if not EMAIL or not PASSWORD:
        print("Please set EMAIL_ADDR and EMAIL_PASS environment variables (or edit the script).")
        sys.exit(1)

    try:
        export_to_tsv_incremental(EMAIL, PASSWORD)
    except imaplib.IMAP4.error as e:
        print("IMAP error:", e)
        print("Make sure IMAP is enabled and credentials/app-password are correct.")
    except Exception as e:
        print("Error:", e)
