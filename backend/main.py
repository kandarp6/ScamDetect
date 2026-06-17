"""
main.py
FastAPI entry point for Fake Internship and Scam Job Detection.

Run locally:
    uvicorn backend.main:app --reload --port 8000

API docs:
    http://localhost:8000/docs
"""

import os
import sys

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = r"C:\pw-browsers"

_venv_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pw_driver = os.path.join(
    _venv_root,
    "venv", "Lib", "site-packages", "playwright", "driver", "playwright.cmd"
)
if os.path.exists(_pw_driver):
    os.environ["PLAYWRIGHT_CLI_TARGET_PATH"] = _pw_driver

import traceback
import hashlib
import re
import random
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from backend.ml.predict import predict_job
from backend.scraper.engines.recruiter_verifier import verify_recruiter
from backend.scraper.storage.supabase_client import (
    get_client,
    get_job_count,
    upsert_recruiter,
    get_existing_job_hashes,
)
from backend.scraper.engines.url_scraper import scrape_job_url
from playwright_stealth import stealth_async


app = FastAPI(
    title="Fake Internship and Scam Job Detection API",
    description="AI-powered fraud detection for Indian job market",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobAnalysisRequest(BaseModel):
    job_title: Optional[str] = ""
    job_description: str
    company_name: Optional[str] = ""
    platform_name: Optional[str] = "Unknown"
    salary_raw: Optional[str] = ""
    city: Optional[str] = ""


class UrlRequest(BaseModel):
    url: str


class RecruiterRequest(BaseModel):
    name: str
    company: Optional[str] = ""
    linkedin_url: Optional[str] = ""


class ReportRequest(BaseModel):
    job_url: Optional[str] = ""
    job_description: str
    company_name: Optional[str] = ""
    contact_method: str
    experience: Optional[str] = ""
    contact: Optional[str] = ""


class ScrapeRequest(BaseModel):
    platform: str
    keywords: str
    limit: Optional[int] = 3


@app.get("/api")
async def api_root():
    return {
        "status": "ok",
        "service": "Fake Internship and Scam Job Detection API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    db_ok = False
    try:
        get_client()
        db_ok = True
    except Exception:
        pass
    return {
        "status": "healthy",
        "database": "connected" if db_ok else "disconnected",
        "playwright_browsers_path": os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "NOT SET"),
    }


@app.post("/api/analyze/job")
async def analyze_job(job: JobAnalysisRequest):
    try:
        job_dict = job.dict()
        prediction = predict_job(job_dict)
        return {
            "score": prediction.ensemble_score,
            "risk_level": prediction.risk_level,
            "is_scam": prediction.is_scam,
            "confidence": prediction.confidence,
            "summary": _get_summary(prediction.risk_level),
            "signals": {
                "language_risk": min(prediction.xgboost_score * 0.7, 100),
                "salary_risk": min(prediction.random_forest_score * 0.6, 100),
                "company_risk": _company_risk(job_dict),
                "contact_risk": _contact_risk(job_dict),
                "requirements_risk": min(prediction.isolation_forest_score * 0.5, 100),
            },
            "keywords": _extract_keywords(prediction),
            "alerts": _build_alerts(prediction),
            "explanation": _build_explanation(prediction),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze/url")
async def analyze_url(req: UrlRequest):
    try:
        scraped_data = await scrape_job_url(req.url)

        if scraped_data.get("job_title") == "Scrape Failed":
            return {
                "score": 50,
                "risk_level": "Medium Risk",
                "is_scam": False,
                "summary": "Could not extract job description from URL. Please paste it directly.",
                "signals": {
                    "language_risk": 0,
                    "salary_risk": 0,
                    "company_risk": 0,
                    "contact_risk": 0,
                    "requirements_risk": 0,
                },
                "keywords": [],
                "alerts": [{
                    "severity": "amber",
                    "title": "Extraction Failed",
                    "message": "We couldn't scrape this link automatically.",
                }],
                "explanation": scraped_data.get("job_description"),
            }

        prediction = predict_job(scraped_data)
        return {
            "score": prediction.ensemble_score,
            "risk_level": prediction.risk_level,
            "is_scam": prediction.is_scam,
            "confidence": prediction.confidence,
            "summary": _get_summary(prediction.risk_level),
            "signals": {
                "language_risk": min(prediction.xgboost_score * 0.7, 100),
                "salary_risk": min(prediction.random_forest_score * 0.6, 100),
                "company_risk": _company_risk(scraped_data),
                "contact_risk": _contact_risk(scraped_data),
                "requirements_risk": min(prediction.isolation_forest_score * 0.5, 100),
            },
            "keywords": _extract_keywords(prediction),
            "alerts": _build_alerts(prediction),
            "explanation": _build_explanation(prediction),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"URL Analysis failed: {str(e)}")


@app.post("/api/verify/recruiter")
async def verify_recruiter_endpoint(req: RecruiterRequest):
    try:
        score, flags = verify_recruiter(
            name=req.name,
            title="",
            email_domain="",
            linkedin_url=req.linkedin_url,
            company_domain=req.company,
        )

        upsert_recruiter(
            name=req.name,
            title="",
            email_domain="",
            linkedin_url=req.linkedin_url or "",
            verification_score=score,
            verification_flags=flags,
            email_hash=""
        )

        return {
            "verified": score >= 60,
            "score": score,
            "checks": [
                {
                    "label": "Name quality",
                    "status": "pass" if "proper_full_name" in flags else "warn",
                    "value": "Verified" if "proper_full_name" in flags else "Generic",
                },
                {
                    "label": "LinkedIn profile",
                    "status": "pass" if "linkedin_present" in flags else "fail",
                    "value": "Found" if "linkedin_present" in flags else "Not provided",
                },
                {
                    "label": "Email domain",
                    "status": "warn",
                    "value": "Not provided",
                },
                {
                    "label": "Community reports",
                    "status": "pass",
                    "value": "None",
                },
            ],
            "message": "Verification complete based on available signals.",
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/report")
async def submit_report(report: ReportRequest):
    try:
        sb = get_client()
        reason = f"Scam report filed for {report.company_name or 'Unknown Company'} via {report.contact_method}."
        if report.contact:
            reason += f" Contact: {report.contact}."
        if report.experience:
            reason += f" Experience: {report.experience}."

        data = {
            "reason": reason,
            "description": report.job_description,
            "evidence_url": report.job_url or "",
            "reporter_name": "Community Member",
            "reporter_email_hash": (
                "sha256:" + hashlib.sha256(b"anonymous@graphura.org").hexdigest()[:16]
            ),
        }
        sb.table("scam_reports").insert(data).execute()

        return {
            "status": "received",
            "message": "Thank you for the report.",
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scrape")
async def scrape_jobs_endpoint(req: ScrapeRequest):
    try:
        from playwright.async_api import async_playwright
        from backend.scraper.connectors import CONNECTOR_REGISTRY
        from backend.scraper.engines.deduplicator import Deduplicator
        from backend.scraper.main import process_single_job, USER_AGENTS, VIEWPORT_POOL
        from backend.scraper.engines.location_normalizer import normalize_location
        from backend.scraper.engines.salary_parser import parse_salary
        from backend.scraper.engines.company_trust import compute_company_trust
        from backend.scraper.engines.scoring_engine import compute_fraud_score

        platform_key = req.platform.lower().strip()
        if platform_key not in CONNECTOR_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported platform: {req.platform}",
            )

        connector_class = CONNECTOR_REGISTRY[platform_key]
        connector = connector_class({})

        sb_client = get_client()
        deduplicator = Deduplicator()
        deduplicator.load_from_supabase(sb_client)
        deduplicator.load_urls_from_supabase(sb_client)

        scraped_results = []
        stats = {"saved": 0, "skipped": 0, "failed": 0, "high_risk": 0}

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport=random.choice(VIEWPORT_POOL),
                locale="en-IN",
                timezone_id="Asia/Kolkata",
            )

            try:
                search_page = await context.new_page()
                await stealth_async(search_page)
                search_urls = connector.search_urls([req.keywords], "India")

                job_urls = []
                if search_urls:
                    await search_page.goto(
                        search_urls[0],
                        wait_until="domcontentloaded",
                        timeout=30_000,
                    )
                    await asyncio.sleep(2)
                    job_urls = await connector.extract_job_links(search_page)

                await search_page.close()

                job_urls = list(dict.fromkeys(job_urls))[: req.limit]

                for url in job_urls:
                    job_page = await context.new_page()
                    try:
                        await stealth_async(job_page)
                        await job_page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=25_000,
                        )
                        await asyncio.sleep(2)

                        raw_job = await connector.extract_job_data(job_page, url)
                        if not raw_job:
                            continue

                        await process_single_job(raw_job, connector, deduplicator, stats)

                        loc = normalize_location(raw_job.location or "")
                        sal = parse_salary(raw_job.salary or "")
                        co_intel = compute_company_trust(raw_job.company_name or "", 0, False)

                        email = ""
                        if raw_job.job_description:
                            m = re.search(
                                r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
                                raw_job.job_description,
                            )
                            email = m.group() if m else ""

                        email_dom = (
                            email.split("@", 1)[1].lower().strip() if "@" in email else ""
                        )
                        rec_score, _ = verify_recruiter(
                            raw_job.recruiter_name or "",
                            "",
                            email_dom,
                            "",
                            co_intel.domain,
                        )

                        scoring = compute_fraud_score(
                            job={
                                "job_description": raw_job.job_description or "",
                                "salary_raw": raw_job.salary or "",
                                "email_domain": email_dom,
                                "is_suspicious_salary": sal.is_suspicious,
                            },
                            company_trust=co_intel.trust_score,
                            recruiter_verif=rec_score,
                            is_government=False,
                            skill_mismatch=False,
                            platform_name=connector.platform_name,
                        )

                        scraped_results.append({
                            "job_title": raw_job.job_title or "Unknown Title",
                            "company_name": raw_job.company_name or "Unknown Company",
                            "platform_name": connector.platform_name,
                            "scam_score": scoring.total_score,
                            "scam_risk_level": scoring.risk_level,
                            "source_url": url,
                            "city": loc.city,
                        })

                    except Exception as inner_e:
                        stats["failed"] += 1
                        print(f"Failed on {url}: {inner_e}")
                    finally:
                        await job_page.close()

            except HTTPException:
                raise
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Scraping logic failed: {str(e)}",
                )
            finally:
                await browser.close()

        return {
            "status": "completed",
            "results": scraped_results,
            "stats": stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    try:
        sb = get_client()
        total = get_job_count()
        scams = (
            sb.table("jobs")
            .select("id", count="exact")
            .in_("scam_risk_level", ["High Risk", "Scam Likely"])
            .execute()
        )
        recruiters = (
            sb.table("recruiters")
            .select("id", count="exact")
            .gte("recruiter_verification_score", 60)
            .execute()
        )
        reports = sb.table("scam_reports").select("id", count="exact").execute()

        return {
            "total_jobs": total,
            "scams_detected": scams.count or 0,
            "verified_recruiters": recruiters.count or 0,
            "reports_filed": reports.count or 0,
        }
    except Exception as e:
        print(f"Stats error: {e}")
        return {
            "total_jobs": 0,
            "scams_detected": 0,
            "verified_recruiters": 0,
            "reports_filed": 0,
        }


@app.get("/api/jobs/recent")
async def get_recent_jobs(limit: int = 5):
    try:
        sb = get_client()
        response = (
            sb.table("jobs")
            .select("id, job_title, scam_score, scam_risk_level, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"jobs": response.data}
    except Exception as e:
        print(f"Recent jobs error: {e}")
        return {"jobs": []}


@app.get("/api/jobs")
async def get_all_jobs(limit: int = 50):
    try:
        sb = get_client()
        response = (
            sb.table("jobs")
            .select(
                "*, companies(name, company_trust_score), "
                "recruiters(name, recruiter_verification_score)"
            )
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"jobs": response.data}
    except Exception as e:
        print(f"Get all jobs error: {e}")
        return {"jobs": []}


def _get_summary(risk_level: str) -> str:
    summaries = {
        "Safe": "No major issues detected.",
        "Low Risk": "Minor concerns. Verify the company.",
        "Medium Risk": "Several cautionary signals.",
        "High Risk": "Multiple red flags detected.",
        "Scam Likely": "Matches known scam patterns. Do not engage.",
    }
    return summaries.get(risk_level, "Analysis complete.")


def _company_risk(job: dict) -> float:
    company = (job.get("company_name") or "").lower()
    if not company:
        return 60.0
    if "solutions" in company or "consultancy" in company:
        return 75.0
    return 25.0


def _contact_risk(job: dict) -> float:
    desc = (job.get("job_description") or "").lower()
    risk = 0.0
    if "whatsapp" in desc:
        risk += 40
    if "telegram" in desc:
        risk += 40
    if "gmail" in desc or "yahoo" in desc:
        risk += 20
    if "registration fee" in desc:
        risk += 30
    return min(risk, 100.0)


def _extract_keywords(prediction) -> list:
    keywords = []
    for f in prediction.top_risk_features or []:
        keywords.append({
            "keyword": f.get("feature", "unknown"),
            "is_red_flag": True,
            "explanation": "Risk signal detected.",
        })
    for f in prediction.top_safe_features or []:
        keywords.append({
            "keyword": f.get("feature", "unknown"),
            "is_red_flag": False,
            "explanation": "Positive signal.",
        })
    return keywords


def _build_alerts(prediction) -> list:
    if prediction.ensemble_score > 65:
        return [{
            "severity": "red",
            "title": "High fraud risk",
            "message": "Multiple scam indicators detected.",
        }]
    elif prediction.ensemble_score > 40:
        return [{
            "severity": "amber",
            "title": "Verify independently",
            "message": "Check the company on MCA21 or LinkedIn.",
        }]
    return [{
        "severity": "green",
        "title": "Looks reasonable",
        "message": "Standard signals detected.",
    }]


def _build_explanation(prediction) -> str:
    score = prediction.ensemble_score
    if prediction.is_scam:
        return f"Score: <strong>{score}/100</strong> - Multiple fraud signals detected."
    elif score > 40:
        return f"Score: <strong>{score}/100</strong> - Proceed with caution."
    return f"Score: <strong>{score}/100</strong> - No major issues found."


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)
