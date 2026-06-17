#supabase_client.py


from typing import Optional
from supabase import create_client, Client
from decouple import config
from loguru import logger


# CLIENT INITIALIZATION

_SUPABASE_URL         = config("SUPABASE_URL",         default="")
_SUPABASE_SERVICE_KEY = config("SUPABASE_SERVICE_KEY", default="")

_sb: Optional[Client] = None


def get_client() -> Client:
    """
    Return singleton Supabase client. Lazy-initialized on first call.
    """
    global _sb

    if _sb is None:
        if not _SUPABASE_URL or not _SUPABASE_SERVICE_KEY:
            raise ValueError(
                "Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env file"
            )
        _sb = create_client(_SUPABASE_URL, _SUPABASE_SERVICE_KEY)

    return _sb


# COMPANY OPERATIONS

def upsert_company(
    name: str,
    intel: Optional[object] = None,
) -> Optional[str]:
    """
    Insert or fetch a company. Returns the company UUID.

    Args:
        name:  Company name (unique key)
        intel: Optional CompanyIntelligence object from company_trust.py

    Returns:
        UUID string of the company, or None if failed
    """
    if not name or not name.strip():
        return None

    name = name.strip()
    sb = get_client()

    try:
        existing = (
            sb.table("companies")
            .select("id")
            .eq("name", name)
            .limit(1)
            .execute()
        )

        if existing.data:
            return existing.data[0]["id"]

        company_data = {"name": name}

        if intel:
            company_data.update({
                "domain":              intel.domain or None,
                "website_active":      intel.website_active,
                "company_trust_score": intel.trust_score,
                "trust_factors":       intel.trust_factors,
            })

        result = sb.table("companies").insert(company_data).execute()

        if result.data:
            return result.data[0]["id"]

        return None

    except Exception as e:
        logger.warning(f"upsert_company failed for '{name}': {e}")
        return None


# RECRUITER OPERATIONS

def upsert_recruiter(
    name: str = "",
    title: str = "",
    email_domain: str = "",
    linkedin_url: str = "",
    verification_score: float = 0.0,
    verification_flags: list = None,
    email_hash: str = "",
) -> Optional[str]:
    """
    Insert or find a recruiter. Returns the recruiter UUID.

    Deduplication keys (in priority order):
        1. name + email_domain
        2. name + linkedin_url
        3. name only
    """
    sb = get_client()

    if not name and not email_domain and not linkedin_url:
        return None

    try:
        query = sb.table("recruiters").select("id")

        if name and email_domain:
            query = query.eq("name", name).eq("email_domain", email_domain)
        elif name and linkedin_url:
            query = query.eq("name", name).eq("linkedin_url", linkedin_url)
        elif name:
            query = query.eq("name", name)
        else:
            return None

        existing = query.limit(1).execute()

        if existing.data:
            return existing.data[0]["id"]

        recruiter_data = {
            "name":                         name or "",
            "title":                        title or "",
            "email_hash":                   email_hash or "",
            "email_domain":                 email_domain or "",
            "linkedin_url":                 linkedin_url or "",
            "recruiter_verification_score": verification_score,
            "verification_flags":           verification_flags or [],
        }

        result = sb.table("recruiters").insert(recruiter_data).execute()

        if result.data:
            return result.data[0]["id"]

        return None

    except Exception as e:
        logger.warning(f"upsert_recruiter failed: {e}")
        return None


# JOB OPERATIONS

def insert_job(job_data: dict) -> bool:
    """
    Insert a job into the jobs table.

    Uses upsert with on_conflict='source_url' so re-scraping the same URL
    updates the existing row instead of failing with a constraint error.

    Returns:
        True if inserted/updated successfully, False on error.
    """
    sb = get_client()

    try:
        result = (
            sb.table("jobs")
            .upsert(job_data, on_conflict="source_url")
            .execute()
        )
        return bool(result.data)

    except Exception as e:
        # Fallback: try job_hash conflict (older schema compatibility)
        try:
            result = (
                sb.table("jobs")
                .upsert(job_data, on_conflict="job_hash", ignore_duplicates=True)
                .execute()
            )
            return bool(result.data)
        except Exception as inner:
            logger.warning(
                f"insert_job failed for '{job_data.get('job_title', 'unknown')[:50]}': {inner}"
            )
            return False


def get_existing_job_hashes() -> set:
    """Return all job_hash values from the database."""
    sb = get_client()

    try:
        result = sb.table("jobs").select("job_hash").execute()
        return {row["job_hash"] for row in result.data if row.get("job_hash")}
    except Exception as e:
        logger.warning(f"get_existing_job_hashes failed: {e}")
        return set()


def get_existing_source_urls() -> set:
    """Return all source_url values from the database."""
    sb = get_client()

    try:
        result = sb.table("jobs").select("source_url").execute()
        return {row["source_url"] for row in result.data if row.get("source_url")}
    except Exception as e:
        logger.warning(f"get_existing_source_urls failed: {e}")
        return set()


def get_job_count() -> int:
    """Return total number of jobs in the database."""
    sb = get_client()

    try:
        result = sb.table("jobs").select("id", count="exact").execute()
        return result.count or 0
    except Exception as e:
        logger.warning(f"get_job_count failed: {e}")
        return 0


def delete_expired_jobs() -> int:
    """
    Delete jobs whose application_deadline has passed.
    Keeps high-risk and scam jobs for analysis.

    Returns:
        Number of jobs deleted
    """
    sb = get_client()

    try:
        from datetime import date
        today = date.today().isoformat()

        result = (
            sb.table("jobs")
            .delete()
            .lt("application_deadline", today)
            .neq("scam_risk_level", "Scam Likely")
            .neq("scam_risk_level", "High Risk")
            .execute()
        )

        return len(result.data) if result.data else 0

    except Exception as e:
        logger.warning(f"delete_expired_jobs failed: {e}")
        return 0


# ANALYTICS HELPERS

def get_scam_distribution() -> dict:
    """Return distribution of jobs by risk level."""
    sb = get_client()

    distribution = {}
    risk_levels = ["Safe", "Low Risk", "Medium Risk", "High Risk", "Scam Likely"]

    for level in risk_levels:
        try:
            result = (
                sb.table("jobs")
                .select("id", count="exact")
                .eq("scam_risk_level", level)
                .execute()
            )
            distribution[level] = result.count or 0
        except Exception:
            distribution[level] = 0

    return distribution


# SELF-TEST

def _self_test():
    print("=" * 70)
    print("SUPABASE CLIENT - SELF-TEST")
    print("=" * 70)

    print("\n--- Test 1: Connection ---")
    try:
        get_client()
        print(f"   Connected to: {_SUPABASE_URL}")
    except Exception as e:
        print(f"   FAILED: {e}")
        print("\n   Check your .env file has:")
        print("     SUPABASE_URL=https://...")
        print("     SUPABASE_SERVICE_KEY=eyJh...")
        return

    print("\n--- Test 2: Read job count ---")
    try:
        count = get_job_count()
        print(f"   Current job count: {count}")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n--- Test 3: Get existing job hashes ---")
    try:
        hashes = get_existing_job_hashes()
        print(f"   Existing job hashes: {len(hashes)}")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n--- Test 4: Get existing source URLs ---")
    try:
        urls = get_existing_source_urls()
        print(f"   Existing source URLs: {len(urls)}")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n--- Test 5: Risk distribution ---")
    try:
        dist = get_scam_distribution()
        for level, count in dist.items():
            print(f"   {level:20s}: {count}")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n--- Test 6: Upsert test company ---")
    try:
        company_id = upsert_company("Test Company XYZ")
        if company_id:
            print(f"   Created/found company: {company_id}")
        else:
            print("   FAILED to create company")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
