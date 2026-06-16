"""
deduplicator.py
Prevent duplicate jobs from being saved to the database.

The same job often appears on multiple platforms or with slightly different
URLs. We generate a deterministic hash from (company + title + city + salary)
to detect content-level duplicates regardless of source URL.

Usage:
    dedup = Deduplicator()
    dedup.load_from_supabase(sb_client)

    for job in scraped_jobs:
        h = make_job_hash(job.company, job.title, job.city, job.salary)
        if dedup.is_duplicate(h):
            continue
        save_job(job)
        dedup.mark_seen(h)
"""

import hashlib
import re
from loguru import logger


# ============================================================================
# HASH GENERATION
# ============================================================================

def make_job_hash(
    company: str = "",
    title: str = "",
    city: str = "",
    salary_raw: str = "",
) -> str:
    """
    Generate a deterministic SHA256 hash for a job posting.

    Properties:
        - Same inputs (case/whitespace-insensitive) -> same hash
        - Different inputs -> different hash
        - One-way (cannot be reversed)

    Returns:
        64-character hex string
    """
    def clean(s: str) -> str:
        """Normalize a string for consistent hashing."""
        if not s:
            return ""
        s = s.lower().strip()
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'[^a-z0-9 ]', '', s)
        return s

    parts = [
        clean(company),
        clean(title),
        clean(city),
        clean(salary_raw),
    ]
    raw_string = "|".join(parts)

    return hashlib.sha256(raw_string.encode('utf-8')).hexdigest()


# ============================================================================
# DEDUPLICATOR CLASS
# ============================================================================

class Deduplicator:
    """In-memory deduplication tracker for both content hashes and source URLs."""

    def __init__(self):
        self._seen_hashes: set = set()
        self._seen_urls:   set = set()
        self._stats = {
            "total_checked":    0,
            "duplicates_found": 0,
            "url_duplicates":   0,
            "new_jobs":         0,
        }

    # ------------------------------------------------------------------------
    # Loading from Supabase
    # ------------------------------------------------------------------------

    def load_from_supabase(self, supabase_client) -> int:
        """
        Load all existing job hashes from Supabase.

        Returns:
            Number of hashes loaded.
        """
        try:
            response = supabase_client.table("jobs").select("job_hash").execute()
            self._seen_hashes = {
                row["job_hash"]
                for row in response.data
                if row.get("job_hash")
            }
            return len(self._seen_hashes)
        except Exception as e:
            logger.warning(f"Could not load hashes from Supabase: {e}")
            return 0

    def load_urls_from_supabase(self, supabase_client) -> int:
        """
        Load all existing source_urls from Supabase.

        Returns:
            Number of URLs loaded.
        """
        try:
            response = supabase_client.table("jobs").select("source_url").execute()
            self._seen_urls = {
                row["source_url"]
                for row in response.data
                if row.get("source_url")
            }
            return len(self._seen_urls)
        except Exception as e:
            logger.warning(f"Could not load source URLs from Supabase: {e}")
            return 0

    # ------------------------------------------------------------------------
    # Duplicate checking
    # ------------------------------------------------------------------------

    def is_duplicate(self, job_hash: str) -> bool:
        """Check if a content hash already exists. O(1)."""
        self._stats["total_checked"] += 1
        if job_hash in self._seen_hashes:
            self._stats["duplicates_found"] += 1
            return True
        return False

    def is_url_seen(self, url: str) -> bool:
        """Check if a source URL already exists. O(1)."""
        if not url:
            return False
        if url in self._seen_urls:
            self._stats["url_duplicates"] += 1
            return True
        return False

    # ------------------------------------------------------------------------
    # Mark as seen
    # ------------------------------------------------------------------------

    def mark_seen(self, job_hash: str, url: str = "") -> None:
        """Add a content hash (and optionally a URL) to the seen set."""
        self._seen_hashes.add(job_hash)
        if url:
            self._seen_urls.add(url)
        self._stats["new_jobs"] += 1

    def mark_url_seen(self, url: str) -> None:
        """Mark a source URL as seen without adding a content hash."""
        if url:
            self._seen_urls.add(url)

    def remove(self, job_hash: str) -> None:
        """Remove a hash (e.g., when a job is deleted)."""
        self._seen_hashes.discard(job_hash)

    # ------------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------------

    def count(self) -> int:
        """Total unique content hashes tracked."""
        return len(self._seen_hashes)

    def url_count(self) -> int:
        """Total unique source URLs tracked."""
        return len(self._seen_urls)

    def get_stats(self) -> dict:
        """Return a copy of the current stats dict."""
        return self._stats.copy()

    def print_stats(self) -> None:
        """Print a summary of deduplication stats."""
        s = self._stats
        print(f"\nDeduplication Stats:")
        print(f"   Total checked:    {s['total_checked']}")
        print(f"   Hash duplicates:  {s['duplicates_found']}")
        print(f"   URL duplicates:   {s['url_duplicates']}")
        print(f"   New jobs:         {s['new_jobs']}")
        print(f"   Hashes in cache:  {self.count()}")
        print(f"   URLs in cache:    {self.url_count()}")


# ============================================================================
# SELF-TEST
# ============================================================================

def _self_test():
    print("=" * 70)
    print("DEDUPLICATOR - SELF-TEST")
    print("=" * 70)

    # Test 1: Hash consistency
    print("\n--- Test 1: Hash consistency ---")
    h1 = make_job_hash("Google", "Python Developer", "Bangalore", "Rs 15 LPA")
    h2 = make_job_hash("google", "python developer", "bangalore", "Rs 15 LPA")
    h3 = make_job_hash("  GOOGLE  ", "Python Developer!", "Bangalore", "Rs 15 LPA")

    print(f"   Hash 1: {h1[:32]}...")
    print(f"   Hash 2: {h2[:32]}...")
    print(f"   Hash 3: {h3[:32]}...")
    print(f"   All same? {h1 == h2 == h3}")

    # Test 2: Different jobs = different hashes
    print("\n--- Test 2: Different jobs ---")
    h_google         = make_job_hash("Google",    "Python Dev", "Bengaluru", "Rs 15 LPA")
    h_microsoft      = make_job_hash("Microsoft", "Python Dev", "Bengaluru", "Rs 15 LPA")
    h_different_role = make_job_hash("Google",    "Java Dev",   "Bengaluru", "Rs 15 LPA")

    print(f"   Google Python:    {h_google[:32]}...")
    print(f"   Microsoft Python: {h_microsoft[:32]}...")
    print(f"   Google Java:      {h_different_role[:32]}...")
    print(f"   All different? {len({h_google, h_microsoft, h_different_role}) == 3}")

    # Test 3: Deduplicator class - hash dedup
    print("\n--- Test 3: Deduplicator class (hash dedup) ---")
    dedup = Deduplicator()

    jobs = [
        ("Google",    "Python Dev",       "Bengaluru", "Rs 15 LPA"),
        ("google",    "Python Developer", "bengaluru", "15 LPA"),     # Variant duplicate
        ("Microsoft", "Python Dev",       "Hyderabad", "Rs 20 LPA"),
        ("Google",    "Python Dev",       "Bengaluru", "Rs 15 LPA"),  # Exact duplicate
        ("Infosys",   "Java Dev",         "Pune",      "Rs 8 LPA"),
    ]

    saved = 0
    for company, title, city, salary in jobs:
        h = make_job_hash(company, title, city, salary)
        if dedup.is_duplicate(h):
            print(f"   SKIPPED: {company} - {title}")
        else:
            print(f"   SAVED:   {company} - {title}")
            dedup.mark_seen(h)
            saved += 1

    print(f"\n   Expected saved: 3 (Google, Microsoft, Infosys)")
    print(f"   Actual saved:   {saved}")
    print(f"   Result: {'PASS' if saved == 3 else 'FAIL'}")

    # Test 4: URL dedup
    print("\n--- Test 4: URL dedup ---")
    dedup2 = Deduplicator()
    urls = [
        "https://example.com/job/1",
        "https://example.com/job/2",
        "https://example.com/job/1",  # duplicate
        "https://example.com/job/3",
    ]
    new_count = 0
    for url in urls:
        if dedup2.is_url_seen(url):
            print(f"   SKIPPED: {url}")
        else:
            print(f"   NEW:     {url}")
            dedup2.mark_url_seen(url)
            new_count += 1

    print(f"   Expected new: 3, Actual: {new_count}")
    print(f"   Result: {'PASS' if new_count == 3 else 'FAIL'}")

    dedup.print_stats()

    # Test 5: Edge cases
    print("\n--- Test 5: Edge cases ---")
    empty_hash = make_job_hash("", "", "", "")
    none_hash  = make_job_hash(None, None, None, None)

    print(f"   Empty hash: {empty_hash[:32]}...")
    print(f"   None hash:  {none_hash[:32]}...")
    print(f"   Same? {empty_hash == none_hash}")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()