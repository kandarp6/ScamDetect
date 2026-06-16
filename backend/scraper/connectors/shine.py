"""
connectors/shine.py
Shine.com scraper - HT Media's job portal, strong in India.

Notes:
    - React-rendered pages need JS hydration wait
    - Salary often a range (e.g. "3-6 LPA")
    - Has experience + education fields not on LinkedIn/Internshala
    - Recruiter contact sometimes in listing itself (fraud signal)
"""

import asyncio
import re
from typing import Optional
from playwright.async_api import Page

from .base import BaseConnector, RawJob


# Error / block page indicators
ERROR_PAGE_INDICATORS = [
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "404 not found",
    "page not found",
    "access denied",
    "you have been blocked",
    "captcha",
    "internal server error",
    "rate limit exceeded",
]


class ShineConnector(BaseConnector):

    platform_name = "Shine"

    SEARCH_URL_TEMPLATE = (
        "https://www.shine.com/job-search/{keywords}-jobs-in-{location}/"
    )
    SEARCH_URL_GENERIC = (
        "https://www.shine.com/job-search/{keywords}-jobs/"
    )

    SELECTORS = {
        # Search results
        "job_cards": ".jobCard, .job-list-item, article.job-tile",
        "job_links": "a.job-title-link, a.jobTitle, h2 a",

        # Job detail
        "job_title":    "h1.job-head, h1.jobTitle, .job-detail-title h1",
        "company":      ".cname a, .company-name, .jobDetail-company",
        "location":     ".loc, .job-location, .jobDetail-loc",
        "salary":       ".salary, .sal, .ctcDetail",
        "experience":   ".exp, .experience, .expDetail",
        "description":  ".jd-desc, .job-description, #jobDescription",
        "posted_date":  ".posted, .job-post-date",
        "skills":       ".skill-name, .keySkill",
        "education":    ".edu, .qualification",
        "company_size": "",   # Not available on Shine
    }

    # ------------------------------------------------------------------------
    # Search URL builder
    # ------------------------------------------------------------------------

    def search_urls(self, keywords: list[str], location: str) -> list[str]:
        urls = []
        for kw in keywords:
            slug_kw  = kw.strip().lower().replace(" ", "-")
            slug_loc = location.strip().lower().replace(" ", "-").replace(",", "")
            if location and location.lower() != "india":
                url = self.SEARCH_URL_TEMPLATE.format(keywords=slug_kw, location=slug_loc)
            else:
                url = self.SEARCH_URL_GENERIC.format(keywords=slug_kw)
            urls.append(url)
        return urls

    # ------------------------------------------------------------------------
    # Popup handling
    # ------------------------------------------------------------------------

    async def handle_popups(self, page: Page) -> None:
        """Dismiss newsletter/signup modals."""
        await asyncio.sleep(1)
        try:
            close = page.locator(
                ".modal-close, .popup-close, button[aria-label='Close']"
            ).first
            if await close.count() > 0:
                await close.click()
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # Job link extraction
    # ------------------------------------------------------------------------

    async def extract_job_links(self, page: Page) -> list[str]:
        if await self._is_error_page(page):
            return []

        try:
            await page.wait_for_selector(self.SELECTORS["job_cards"], timeout=10000)
        except Exception:
            pass

        await self.handle_popups(page)

        try:
            links = await page.eval_on_selector_all(
                self.SELECTORS["job_links"],
                "els => els.map(e => e.href).filter(h => h && h.includes('shine.com'))"
            )
        except Exception:
            links = []

        # Fallback: regex over raw HTML if no links found
        if not links:
            try:
                html = await page.content()
                raw = re.findall(r'href=["\'](https?://[^"\']*shine\.com[^"\']*jobs?[^"\']*)["\']', html)
                links = raw
            except Exception:
                pass

        seen = set()
        clean = []
        for link in links:
            base = link.split("?")[0].rstrip("/")
            if base and base not in seen:
                seen.add(base)
                clean.append(base)

        return clean[: self.jobs_per_query]

    # ------------------------------------------------------------------------
    # Job data extraction
    # ------------------------------------------------------------------------

    async def extract_job_data(self, page: Page, url: str) -> Optional[RawJob]:
        if await self._is_error_page(page):
            return None

        try:
            await page.wait_for_selector(self.SELECTORS["job_title"], timeout=12000)
        except Exception:
            return None

        await self.handle_popups(page)

        title = await self._safe_text(page, self.SELECTORS["job_title"])
        if not title or self._is_invalid_title(title):
            return None

        company  = await self._safe_text(page, self.SELECTORS["company"])
        location = await self._safe_text(page, self.SELECTORS["location"])
        salary   = await self._safe_text(page, self.SELECTORS["salary"])
        exp      = await self._safe_text(page, self.SELECTORS["experience"])
        desc     = await self._safe_text(page, self.SELECTORS["description"])
        posted   = await self._safe_text(page, self.SELECTORS["posted_date"])
        edu      = await self._safe_text(page, self.SELECTORS["education"])

        try:
            skills = await page.eval_on_selector_all(
                self.SELECTORS["skills"],
                "els => els.map(e => e.innerText.trim()).filter(Boolean)"
            )
        except Exception:
            skills = []

        mode = self._detect_mode(location + " " + desc)

        return RawJob(
            source_url      = url,
            platform_name   = self.platform_name,
            job_title       = title,
            company_name    = company,
            location        = location,
            salary          = salary,
            job_description = desc,
            posted_date     = posted,
            mode            = mode,
            extra_fields    = {
                "experience": exp,
                "education":  edu,
                "skills":     ", ".join(skills),
            }
        )

    # ------------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------------

    async def _is_error_page(self, page: Page) -> bool:
        """Detect server errors and block pages."""
        try:
            title = (await page.title() or "").lower()
            for indicator in ERROR_PAGE_INDICATORS:
                if indicator in title:
                    return True

            body_text = await page.evaluate(
                "() => (document.body ? document.body.innerText : '').slice(0, 500).toLowerCase()"
            )
            for indicator in ERROR_PAGE_INDICATORS:
                if indicator in body_text:
                    return True
        except Exception:
            return True
        return False

    def _is_invalid_title(self, title: str) -> bool:
        """Reject obviously-not-job titles."""
        t = title.lower().strip()
        if len(t) < 5:
            return True
        for indicator in ERROR_PAGE_INDICATORS:
            if indicator in t:
                return True
        return False

    async def _safe_text(self, page: Page, selector: str) -> str:
        """Return text of first matching element, or empty string."""
        if not selector:
            return ""
        try:
            el = page.locator(selector).first
            if await el.count() > 0:
                return (await el.inner_text()).strip()
        except Exception:
            pass
        return ""

    def _detect_mode(self, text: str) -> str:
        """Infer work mode from text signals."""
        t = text.lower()
        if "work from home" in t or "remote" in t or "wfh" in t:
            return "Remote"
        if "hybrid" in t:
            return "Hybrid"
        return "On-site"