"""
batch_predict.py
Score every job in the Supabase database and save ML predictions back to DB.

Usage:
    python -m backend.ml.batch_predict
"""

import time
from .predict import predict_job, load_models
from ..scraper.storage.supabase_client import get_client


def batch_score_all_jobs(only_unscored: bool = False, limit: int = None):
    """
    Score every job in Supabase and update ml_score columns.

    Args:
        only_unscored: If True, skip jobs that already have ml_score
        limit: Optional limit (None = score all)
    """
    print("=" * 70)
    print("BATCH ML PREDICTION - Scoring all Supabase jobs")
    print("=" * 70)

    sb = get_client()
    load_models()

    print("\nFetching jobs from Supabase...")
    query = sb.table("jobs").select(
        "*, companies(name, company_trust_score), recruiters(recruiter_verification_score)"
    )

    if only_unscored:
        query = query.is_("ml_score", "null")

    if limit:
        query = query.limit(limit)

    response = query.execute()
    jobs = response.data
    print(f"   Found {len(jobs)} jobs to score")

    if not jobs:
        print("   No jobs to score. Exiting.")
        return

    for job in jobs:
        if job.get("companies"):
            job["company_name"] = job["companies"].get("name", "")
            job["company_trust_score"] = job["companies"].get("company_trust_score", 50)
        if job.get("recruiters"):
            job["recruiter_verification_score"] = job["recruiters"].get("recruiter_verification_score", 30)

    print(f"\nScoring {len(jobs)} jobs...")
    start = time.time()
    success = 0
    failed = 0

    for i, job in enumerate(jobs, 1):
        try:
            prediction = predict_job(job, verbose=False)

            sb.table("jobs").update({
                "ml_score":          prediction.ensemble_score,
                "ml_risk_level":     prediction.risk_level,
                "ml_is_scam":        prediction.is_scam,
                "ml_confidence":     prediction.confidence,
                "ml_xgboost_score":  prediction.xgboost_score,
                "ml_rf_score":       prediction.random_forest_score,
                "ml_iso_score":      prediction.isolation_forest_score,
            }).eq("id", job["id"]).execute()

            success += 1

            if i % 10 == 0 or i == len(jobs):
                elapsed = time.time() - start
                rate = i / elapsed
                print(f"   [{i}/{len(jobs)}] - {rate:.1f} jobs/sec")

        except Exception as e:
            failed += 1
            print(f"   FAILED on job {job.get('id', '?')}: {e}")

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("BATCH SCORING COMPLETE")
    print("=" * 70)
    print(f"   Total:   {len(jobs)}")
    print(f"   Success: {success}")
    print(f"   Failed:  {failed}")
    print(f"   Time:    {elapsed:.1f} seconds")
    if elapsed > 0:
        print(f"   Rate:    {len(jobs)/elapsed:.1f} jobs/sec")
    print("=" * 70)


def main():
    batch_score_all_jobs(only_unscored=False)


if __name__ == "__main__":
    main()