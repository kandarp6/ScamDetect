
#connectors/linkedin.py


import re
import asyncio
import random
from typing import Optional
from playwright.async_api import Page

from .base import BaseConnector, RawJob


# Error / login-wall indicators
ERROR_PAGE_INDICATORS = [
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "404 not found",
    "page not found",
    "access denied",
    "you have been blocked",
    "unusual activity",
    "captcha",
    "security verification",
    "join linkedin",
    "sign in to view",
    "authwall",
]


# Rotation pools for fingerprint randomization
USER_AGENT_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

VIEWPORT_POOL = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1600, "height": 900},
    {"width": 1680, "height": 1050},
    {"width": 1920, "height": 1080},
]

LOCALE_POOL = [
    ("en-IN", "Asia/Kolkata"),
    ("en-US", "America/New_York"),
    ("en-GB", "Europe/London"),
    ("en-CA", "America/Toronto"),
    ("en-AU", "Australia/Sydney"),
]

REFERER_POOL = [
    "https://www.google.com/",
    "https://www.google.co.in/",
    "https://duckduckgo.com/",
    "https://www.bing.com/",
    "https://www.linkedin.com/",
]


class LinkedInConnector(BaseConnector):

    platform_name = "LinkedIn"

    # f_TPR=r604800 -> posted in last 7 days; f_WT=2 -> remote filter (optional)
    SEARCH_URL_TEMPLATE = (
        "https://www.linkedin.com/jobs/search"
        "?keywords={keywords}"
        "&location={location}"
        "&f_TPR=r604800"
        "&start={start}"
    )

    # Selectors grouped here for easy update when LinkedIn changes layout
    SELECTORS = {
        "job_links": (
            "a.base-card__full-link, "
            "a.job-card-container__link, "
            "a.job-card-list__title, "
            "a[href*='/jobs/view/']"
        ),
        "job_title": (
            "h1.top-card-layout__title, "
            "h1.topcard__title, "
            "h1.t-24"
        ),
        "company": (
            "a.topcard__org-name-link, "
            "span.topcard__flavor, "
            "a.job-details-jobs-unified-top-card__company-name"
        ),
        "location": (
            "span.topcard__flavor--bullet, "
            "span.job-details-jobs-unified-top-card__bullet"
        ),
        "posted_date": (
            "span.posted-time-ago__text, "
            "span.job-details-jobs-unified-top-card__posted-date"
        ),
        "description": (
            "div.show-more-less-html__markup, "
            "div.description__text, "
            "div.jobs-description__content"
        ),
        "show_more_btn": (
            "button.show-more-less-html__button--more, "
            "button.jobs-description__footer-button"
        ),
        "criteria_items": (
            "span.description__job-criteria-text, "
            "li.description__job-criteria-item span"
        ),
        "recruiter": (
            "div.hirer-card__hirer-information, "
            "div.message-the-recruiter"
        ),
        "workplace_badge": (
            "span.workplace-type, "
            "li.job-details-jobs-unified-top-card__job-insight span"
        ),
    }

    # Per-session fingerprint

    def __init__(self, config: dict):
        super().__init__(config)
        # Rotate fingerprint once per connector instance
        self.session_user_agent = random.choice(USER_AGENT_POOL)
        self.session_viewport   = random.choice(VIEWPORT_POOL)
        self.session_locale, self.session_timezone = random.choice(LOCALE_POOL)
        self.session_referer    = random.choice(REFERER_POOL)

    def get_browser_fingerprint(self) -> dict:
        """
        Return browser context settings for this session.
        main.py can call this when creating the Playwright context.
        """
        return {
            "user_agent":  self.session_user_agent,
            "viewport":    self.session_viewport,
            "locale":      self.session_locale,
            "timezone_id": self.session_timezone,
        }

    # Search URL builder

    def search_urls(self, keywords: list[str], location: str) -> list[str]:
        urls = []
        for kw in keywords:
            kw_encoded  = kw.strip().replace(" ", "+")
            loc_encoded = location.strip().replace(" ", "+")
            urls.append(self.SEARCH_URL_TEMPLATE.format(
                keywords=kw_encoded,
                location=loc_encoded,
                start=0,
            ))
        return urls

    # Pre-search warm-up

    async def pre_search_hook(self, page: Page) -> None:
        """
        Visit a benign page first to look more like a real browser session.
        Sets HTTP_REFERER so LinkedIn sees us coming from Google etc.
        """
        try:
            await page.set_extra_http_headers({
                "Referer":         self.session_referer,
                "Accept-Language": f"{self.session_locale},en;q=0.9",
                "Sec-Ch-Ua":       '"Chromium";v="131", "Not_A Brand";v="24"',
                "Sec-Ch-Ua-Mobile":"?0",
                "Sec-Ch-Ua-Platform":'"Windows"',
            })
            # Brief visit to LinkedIn home so cookies are set naturally
            await page.goto("https://www.linkedin.com/", wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(random.uniform(2.0, 4.0))
        except Exception:
            pass

    # Popup / login-wall handling

    async def handle_popups(self, page: Page) -> None:
        """Dismiss login modals and overlays that block scraping."""
        await asyncio.sleep(random.uniform(1.0, 2.0))
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass

        try:
            await page.evaluate("""
                () => {
                    const selectors = [
                        '.modal-overlay',
                        '.artdeco-modal-overlay',
                        '[data-test-modal-overlay]',
                        '.authwall-join-form',
                        '.contextual-sign-in-modal',
                        '.cold-join-form',
                        '.global-alert',
                    ];
                    selectors.forEach(sel => {
                        document.querySelectorAll(sel).forEach(el => el.remove());
                    });
                    document.body.style.overflow = 'auto';
                    document.documentElement.style.overflow = 'auto';
                }
            """)
        except Exception:
            pass

    # Job link extraction

    async def extract_job_links(self, page: Page) -> list[str]:
        if await self._is_error_page(page):
            return []

        # Human-like scrolling to trigger lazy-loading
        await self._human_scroll(page, scrolls=random.randint(5, 8))
        await self.handle_popups(page)
        await asyncio.sleep(random.uniform(1.0, 2.0))

        try:
            links = await page.eval_on_selector_all(
                self.SELECTORS["job_links"],
                "els => els.map(e => e.href).filter(h => h && h.includes('/jobs/view/'))"
            )
        except Exception:
            links = []

        # Fallback: regex over raw HTML
        if not links:
            try:
                html = await page.content()
                raw = re.findall(r'href=["\'](https?://[^"\']*?/jobs/view/\d+[^"\']*)["\']', html)
                links = raw
            except Exception:
                links = []

        # Canonicalize: keep only /jobs/view/JOBID
        seen = set()
        clean_links = []
        for link in links:
            match = re.match(r"(https?://(?:www\.)?linkedin\.com/jobs/view/\d+)", link)
            if match:
                canonical = match.group(1).replace("http://", "https://")
                if canonical not in seen:
                    seen.add(canonical)
                    clean_links.append(canonical)

        return clean_links[: self.jobs_per_query]

    # Job data extraction

    async def extract_job_data(self, page: Page, url: str) -> Optional[RawJob]:
        await self.handle_popups(page)

        if await self._is_error_page(page):
            return None

        title_text = await self._safe_text(page, self.SELECTORS["job_title"])
        if not title_text or self._is_invalid_title(title_text):
            return None

        # Click "Show more" to reveal full description
        try:
            btn = page.locator(self.SELECTORS["show_more_btn"]).first
            if await btn.count() > 0:
                await btn.click()
                await asyncio.sleep(random.uniform(0.6, 1.2))
        except Exception:
            pass

        description  = await self._extract_description(page)
        company_name = await self._safe_text(page, self.SELECTORS["company"])
        location     = await self._safe_text(page, self.SELECTORS["location"])
        posted_date  = await self._safe_text(page, self.SELECTORS["posted_date"])

        company_size = await self._extract_criteria_item(page, ["employee", "people"])
        seniority    = await self._extract_criteria_item(page, ["seniority", "level"])
        employment   = await self._extract_criteria_item(page, ["employment", "type"])
        industry     = await self._extract_criteria_item(page, ["industries", "industry"])

        recruiter_name, recruiter_title = await self._extract_recruiter(page)

        # Mode detection: workplace badge first, then text fallback
        mode = await self._detect_mode(page, location, description)

        return RawJob(
            source_url       = url,
            platform_name    = self.platform_name,
            job_title        = title_text,
            company_name     = company_name,
            location         = location,
            job_description  = description,
            posted_date      = posted_date,
            company_size     = company_size,
            recruiter_name   = recruiter_name,
            recruiter_title  = recruiter_title,
            mode             = mode,
            extra_fields     = {
                "seniority":  seniority,
                "employment": employment,
                "industry":   industry,
            },
        )

    # Private helpers

    async def _is_error_page(self, page: Page) -> bool:
        """Detect server errors, captchas, and login walls."""
        try:
            title = (await page.title() or "").lower()
            for indicator in ERROR_PAGE_INDICATORS:
                if indicator in title:
                    return True

            body_text = await page.evaluate(
                "() => (document.body ? document.body.innerText : '').slice(0, 800).toLowerCase()"
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
        try:
            el = page.locator(selector).first
            if await el.count() > 0:
                return (await el.inner_text()).strip()
        except Exception:
            pass
        return ""

    async def _extract_description(self, page: Page) -> str:
        """
        Three-strategy description extraction:
            1. Known LinkedIn selectors
            2. Largest meaningful text block
            3. JSON-LD structured data in HTML
        """
        text = await self._safe_text(page, self.SELECTORS["description"])
        if text and len(text) > 50:
            return text

        try:
            blocks = await page.eval_on_selector_all(
                "section, div, p",
                """els => els
                    .map(e => e.innerText.trim())
                    .filter(t => t.length > 150)
                    .sort((a, b) => b.length - a.length)
                    .slice(0, 5)
                """
            )
            skip = ["sign in", "join now", "cookie", "privacy policy",
                    "terms of service", "copyright", "people you may know"]
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

    async def _extract_criteria_item(self, page: Page, keywords: list[str]) -> str:
        """
        Find a value in LinkedIn's 'About the job' criteria list whose
        adjacent label contains any of the given keywords.
        """
        try:
            items = await page.eval_on_selector_all(
                self.SELECTORS["criteria_items"],
                "els => els.map(e => e.innerText.trim())"
            )
            for item in items:
                item_lower = item.lower()
                if any(kw in item_lower for kw in keywords):
                    return item
        except Exception:
            pass
        return ""

    async def _extract_recruiter(self, page: Page) -> tuple[str, str]:
        """Extract recruiter name and title from hiring team card."""
        try:
            card = page.locator(self.SELECTORS["recruiter"]).first
            if await card.count() > 0:
                text = await card.inner_text()
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                name  = lines[0] if len(lines) > 0 else ""
                title = lines[1] if len(lines) > 1 else ""
                return name, title
        except Exception:
            pass
        return "", ""

    async def _detect_mode(self, page: Page, location: str, description: str) -> str:
        """
        Infer work mode. Order of preference:
            1. Explicit workplace badge in top card
            2. Text signals in location string
            3. Text signals in description
        """
        try:
            badge = page.locator(self.SELECTORS["workplace_badge"]).first
            if await badge.count() > 0:
                text = (await badge.inner_text()).lower().strip()
                if "remote"  in text: return "Remote"
                if "hybrid"  in text: return "Hybrid"
                if "on-site" in text or "onsite" in text: return "On-site"
        except Exception:
            pass

        text_all = (location + " " + description).lower()
        if "remote"  in text_all: return "Remote"
        if "hybrid"  in text_all: return "Hybrid"
        if any(s in text_all for s in ["on-site", "onsite", "in-office", "in office"]):
            return "On-site"

        return "Unknown"

    async def _human_scroll(self, page: Page, scrolls: int = 5) -> None:
        """
        Human-like scrolling: variable speeds, occasional reverse,
        random mouse jitter.
        """
        for i in range(scrolls):
            # Occasional small upward scroll to look human
            if i > 0 and random.random() < 0.15:
                await page.mouse.wheel(0, -random.randint(100, 250))
                await asyncio.sleep(random.uniform(0.4, 0.9))

            scroll_amount = random.randint(300, 900)
            await page.mouse.wheel(0, scroll_amount)
            await asyncio.sleep(random.uniform(0.6, 1.6))

            # Random mouse movement to simulate human attention
            if random.random() < 0.3:
                try:
                    x = random.randint(100, self.session_viewport["width"] - 100)
                    y = random.randint(100, self.session_viewport["height"] - 100)
                    await page.mouse.move(x, y, steps=random.randint(5, 15))
                except Exception:
                    pass
