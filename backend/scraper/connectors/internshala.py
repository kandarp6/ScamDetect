"""
connectors/internshala.py
Internshala scraper.
"""

import re
import asyncio
import random
from typing import Optional
from playwright.async_api import Page

from .base import BaseConnector, RawJob


# Indicators that a scraped page is an error/captcha rather than a real listing
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

# Patterns that confirm a string is an actual posted-date
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

# Strings that are clearly NOT a posted date (job-type labels, etc.)
NON_DATE_VALUES = {
    "internship", "full time", "part time", "full-time", "part-time",
    "permanent", "contract", "remote", "on-site", "onsite", "hybrid",
    "unknown", "n/a", "na", "",
}

# Work-mode detection keywords
REMOTE_KEYWORDS = [
    "work from home", "wfh", "remote", "anywhere",
    "fully remote", "telecommute", "virtual",
]

ONSITE_KEYWORDS = ["on-site", "onsite", "in-office", "in office"]


class InternshalaConnector(BaseConnector):

    platform_name = "Internshala"

    SEARCH_URL = "https://internshala.com/internships/keywords-{keywords}/"

    SELECTORS = {
        "card_signal": "div.individual_internship, .internship_list_container, #internship_list",

        "job_title": [
            "h1.internship_heading",
            ".profile_on_detail_page",
            ".internship_heading h1",
            "h1",
        ],
        "company": [
            ".company_name a",
            "a.link_display_like_text",
            ".internship_heading .heading_6",
            ".heading_6",
        ],
        "stipend": [
            ".stipend",
            "#stipend",
            ".salary_container .stipend",
            "[class*='stipend']",
        ],
        "duration": [
            "#duration",
            "[id='duration']",
        ],
        "description": [
            "#about_internship .text-container",
            "#about_internship",
            ".about_internship_container",
            ".internship_details",
            ".internship_details_section",
            "[class*='about'] [class*='text']",
            "[class*='description']",
        ],
        "posted_date": [
            ".posted_by_container span",
            ".status-inactive",
            "[class*='posted']",
            ".other_detail_item_row span",
        ],
        "apply_by": [
            "#apply_by",
            ".apply_by_date",
            "[id*='apply']",
        ],
        "skills": [
            ".round_tags_container span",
            ".skill-tags span",
            "[class*='skill'] span",
        ],
        "location":       ["a.location_link"],
        "work_from_home": ["#work_from_home_icon", ".work_from_home_tag"],
    }

    # ------------------------------------------------------------------------
    # Search URL builder
    # ------------------------------------------------------------------------

    def search_urls(self, keywords: list[str], location: str) -> list[str]:
        result = []
        for kw in keywords:
            slug = kw.strip().lower().replace(" ", "-").replace("_", "-")
            result.append(self.SEARCH_URL.format(keywords=slug))
        return result

    # ------------------------------------------------------------------------
    # Popup handling
    # ------------------------------------------------------------------------

    async def handle_popups(self, page: Page) -> None:
        await asyncio.sleep(0.5)
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # Job link extraction (from search results page)
    # ------------------------------------------------------------------------

    async def extract_job_links(self, page: Page) -> list[str]:
        if await self._is_error_page(page):
            return []

        try:
            await page.wait_for_selector(self.SELECTORS["card_signal"], timeout=15000)
        except Exception:
            await asyncio.sleep(3)

        for _ in range(5):
            await page.mouse.wheel(0, random.randint(300, 600))
            await asyncio.sleep(random.uniform(0.5, 1.0))

        await self.handle_popups(page)

        links = await page.eval_on_selector_all(
            "a[href*='/internship/detail/']",
            "els => [...new Set(els.map(e => e.href))]"
        )

        if not links:
            html = await page.content()
            raw = re.findall(
                r'href=["\'](/internship/detail/[a-zA-Z0-9\-_%]+/?)["\']',
                html
            )
            links = [f"https://internshala.com{p}" for p in dict.fromkeys(raw)]

        seen = set()
        clean = []
        for link in links:
            base = link.split("?")[0].rstrip("/")
            if base not in seen and "internshala.com/internship" in base:
                seen.add(base)
                clean.append(base)

        return clean[: self.jobs_per_query]

    # ------------------------------------------------------------------------
    # Job data extraction (from job detail page)
    # ------------------------------------------------------------------------

    async def extract_job_data(self, page: Page, url: str) -> Optional[RawJob]:
        await self.handle_popups(page)

        if await self._is_error_page(page):
            return None

        try:
            await page.wait_for_selector("h1, .profile_on_detail_page", timeout=12000)
        except Exception:
            return None

        title = await self._safe_text(page, self.SELECTORS["job_title"])
        if not title:
            return None

        if self._is_invalid_title(title):
            return None

        company  = await self._safe_text(page, self.SELECTORS["company"])
        stipend  = await self._safe_text(page, self.SELECTORS["stipend"])
        duration = await self._safe_text(page, self.SELECTORS["duration"])
        desc     = await self._extract_description(page)
        apply_by = await self._safe_text(page, self.SELECTORS["apply_by"])
        posted   = await self._extract_posted_date(page)
        location = await self._extract_location(page)

        # Multi-signal mode detection (title + description + location)
        mode = self._detect_mode(title, desc, location)

        # Extract skills
        skills = []
        for sel in self.SELECTORS["skills"]:
            try:
                skills = await page.eval_on_selector_all(
                    sel,
                    "els => els.map(e => e.innerText.trim()).filter(Boolean)"
                )
                if skills:
                    break
            except Exception:
                continue

        return RawJob(
            source_url           = url,
            platform_name        = self.platform_name,
            job_title            = title,
            company_name         = company,
            location             = location,
            salary               = stipend,
            job_description      = desc,
            posted_date          = posted,
            application_deadline = apply_by,
            mode                 = mode,
            extra_fields         = {
                "duration": duration,
                "skills":   ", ".join(skills),
            }
        )

    # ------------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------------

    async def _is_error_page(self, page: Page) -> bool:
        """Detect if the loaded page is a server error or block page."""
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
        """Reject titles that are obviously not real job postings."""
        t = title.lower().strip()
        if len(t) < 5:
            return True
        for indicator in ERROR_PAGE_INDICATORS:
            if indicator in t:
                return True
        return False

    def _detect_mode(self, title: str, description: str, location: str) -> str:
        """
        Detect work mode using multiple signals:
            1. Job title text (highest priority)
            2. Description text
            3. Location hints

        Returns: Remote / Hybrid / On-site / Unknown
        """
        combined = f"{title} {description} {location}".lower()

        # Explicit remote signals
        if any(kw in combined for kw in REMOTE_KEYWORDS):
            return "Remote"

        # Hybrid signals
        if "hybrid" in combined:
            return "Hybrid"

        # Explicit on-site signals
        if any(kw in combined for kw in ONSITE_KEYWORDS):
            return "On-site"

        # Has a real location -> assume on-site
        if location and location.lower() not in ("", "remote", "anywhere"):
            return "On-site"

        return "Unknown"

    async def _extract_posted_date(self, page: Page) -> str:
        """
        Extract posted date string.
        Filters out non-date values like "Internship", "Part Time".
        """
        selectors = self.SELECTORS["posted_date"]
        if isinstance(selectors, str):
            selectors = [selectors]

        for sel in selectors:
            try:
                elements = page.locator(sel)
                count = await elements.count()
                for i in range(min(count, 5)):
                    text = (await elements.nth(i).inner_text()).strip()
                    if not text:
                        continue

                    text_lower = text.lower()

                    # Skip clearly-not-a-date strings
                    if text_lower in NON_DATE_VALUES:
                        continue

                    # Match against date patterns
                    if any(re.search(p, text_lower) for p in DATE_PATTERNS):
                        return text
            except Exception:
                continue

        return ""

    async def _safe_text(self, page: Page, selector) -> str:
        """Try each selector in the list; return first non-empty match."""
        sels = selector if isinstance(selector, list) else [selector]
        for sel in sels:
            try:
                el = page.locator(sel).first
                if await el.count() > 0:
                    text = (await el.inner_text()).strip()
                    if text:
                        return text
            except Exception:
                continue
        return ""

    async def _extract_description(self, page: Page) -> str:
        """
        Three-strategy description extraction:
            1. Known CSS selectors
            2. Largest text block on page (>150 chars)
            3. JSON-LD structured data
        """
        text = await self._safe_text(page, self.SELECTORS["description"])
        if text and len(text) > 50:
            return text

        try:
            blocks = await page.eval_on_selector_all(
                "p, div, section",
                """els => els
                    .map(e => e.innerText.trim())
                    .filter(t => t.length > 150)
                    .sort((a, b) => b.length - a.length)
                    .slice(0, 5)
                """
            )
            skip = [
                "copyright", "privacy policy", "terms of", "log in",
                "sign up", "cookie", "all rights reserved",
            ]
            for block in blocks:
                if not any(s in block.lower() for s in skip):
                    return block
        except Exception:
            pass

        try:
            html = await page.content()
            m = re.search(r'"description"\s*:\s*"([^"]{100,})"', html)
            if m:
                return (m.group(1)
                          .replace("\\n", "\n")
                          .replace("\\u003c", "<")
                          .replace("\\u003e", ">"))
        except Exception:
            pass

        return ""

    async def _extract_location(self, page: Page) -> str:
        try:
            locs = await page.eval_on_selector_all(
                "a.location_link",
                "els => els.map(e => e.innerText.trim()).filter(Boolean)"
            )
            return ", ".join(locs[:4]) if locs else ""
        except Exception:
            return ""