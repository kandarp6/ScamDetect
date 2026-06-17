"""
fix_existing_jobs.py
One-time cleanup script for incorrect mode and posted_date values
in already-scraped jobs.

Usage:
    python -m backend.scraper.engines.fix_existing_jobs
"""

import re
import time
from ..storage.supabase_client import get_client


# RULES

REMOTE_KEYWORDS = [
    "work from home", "wfh", "remote", "anywhere",
    "fully remote", "telecommute", "virtual",
]

ONSITE_KEYWORDS = ["on-site", "onsite", "in-office", "in office"]

DATE_PATTERNS = [
    r'\bposted\s+',
    r'\bday\b', r'\bdays\b',
    r'\bweek\b', r'\bweeks\b',
    r'\bmonth\b', r'\bmonths\b',
    r'\bhour\b', r'\bhours\b',
    r'\bjust\s+now\b',
    r'\btoday\b', r'\byesterday\b',
    r'\d+\s+(day|week|month|hour)',
]

NON_DATE_VALUES = {
    "internship", "full time", "part time", "full-time", "part-time",
    "permanent", "contract", "remote", "on-site", "onsite", "hybrid",
    "unknown", "n/a", "na", "",
}


# HELPERS

def detect_mode_from_text(title: str, description: str, location: str) -> str:
    """Re-detect mode from job title + description + location."""
    combined = f"{title} {description} {location}".lower()

    if any(kw in combined for kw in REMOTE_KEYWORDS):
        return "Remote"

    if "hybrid" in combined:
        return "Hybrid"

    if any(kw in combined for kw in ONSITE_KEYWORDS):
        return "On-site"

    if location and location.lower() not in ("", "remote", "anywhere"):
        return "On-site"

    return "Unknown"


def is_real_date(text: str) -> bool:
    """Check if the text looks like an actual posted-date string."""
    if not text:
        return False
    text_lower = text.lower().strip()
    if text_lower in NON_DATE_VALUES:
        return False
    return any(re.search(p, text_lower) for p in DATE_PATTERNS)


# MAIN CLEANUP

def cleanup_jobs():
    print("=" * 70)
    print("FIX EXISTING JOBS - mode & posted_date cleanup")
    print("=" * 70)

    sb = get_client()

    print("\nFetching jobs from Supabase...")
    response = sb.table("jobs").select(
        "id, job_title, job_description, location_raw, mode, posted_date"
    ).execute()

    jobs = response.data
    print(f"   Loaded {len(jobs)} jobs")

    if not jobs:
        print("   No jobs to process. Exiting.")
        return

    print("\nAnalyzing and updating jobs...\n")
    start = time.time()

    mode_fixes = 0
    date_fixes = 0
    no_change = 0
    failed = 0

    for i, job in enumerate(jobs, 1):
        updates = {}

        title    = job.get("job_title", "") or ""
        desc     = job.get("job_description", "") or ""
        location = job.get("location_raw", "") or ""
        old_mode = job.get("mode", "") or ""
        old_date = job.get("posted_date", "") or ""

        # Fix 1: mode
        new_mode = detect_mode_from_text(title, desc, location)
        if new_mode != old_mode and new_mode != "Unknown":
            updates["mode"] = new_mode

        # Fix 2: posted_date
        if old_date.strip() and not is_real_date(old_date):
            updates["posted_date"] = ""

        # Apply updates
        if updates:
            try:
                sb.table("jobs").update(updates).eq("id", job["id"]).execute()

                changes = []
                if "mode" in updates:
                    changes.append(f"mode: '{old_mode}' -> '{updates['mode']}'")
                    mode_fixes += 1
                if "posted_date" in updates:
                    changes.append(f"posted_date: '{old_date}' -> '' (cleared)")
                    date_fixes += 1

                print(f"   [{i}/{len(jobs)}] FIXED: {title[:50]}")
                for change in changes:
                    print(f"      {change}")

            except Exception as e:
                failed += 1
                print(f"   [{i}/{len(jobs)}] FAILED: {title[:50]} - {e}")
        else:
            no_change += 1

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)
    print(f"   Total jobs:        {len(jobs)}")
    print(f"   Mode fixed:        {mode_fixes}")
    print(f"   Posted date fixed: {date_fixes}")
    print(f"   No change needed:  {no_change}")
    print(f"   Failed:            {failed}")
    print(f"   Time:              {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    cleanup_jobs()
