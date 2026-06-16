"""
connectors/base.py
Abstract contract every platform connector must fulfill.

To add a new platform:
    1. Create a file in connectors/ (e.g. naukri.py)
    2. Subclass BaseConnector
    3. Implement the four abstract methods
    4. Register it in connectors/__init__.py

main.py never changes when adding new platforms.

Pattern: Template Method + Strategy
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from playwright.async_api import Page


# ---------------------------------------------------------------------------
# Shared enumerations
# ---------------------------------------------------------------------------

class WorkMode(str, Enum):
    REMOTE  = "Remote"
    HYBRID  = "Hybrid"
    ON_SITE = "On-site"
    UNKNOWN = "Unknown"


class RiskLevel(str, Enum):
    SAFE        = "Safe"
    LOW_RISK    = "Low Risk"
    MEDIUM_RISK = "Medium Risk"
    HIGH_RISK   = "High Risk"
    LIKELY_SCAM = "Likely Scam"


# ---------------------------------------------------------------------------
# Raw data container returned by each connector
# ---------------------------------------------------------------------------

@dataclass
class RawJob:
    """
    Loose container for scraped job data.

    Each platform returns data in different shapes. RawJob accepts whatever
    the connector can find. Downstream engines normalize and validate.

    Fields marked Optional may not exist on every platform.
    extra_fields holds platform-specific data that does not fit the standard
    schema (e.g. {"stipend_type": "fixed", "duration": "3 months"}).
    """
    source_url:    str                # REQUIRED - unique key
    platform_name: str                # REQUIRED - e.g. "LinkedIn"

    job_title:            str             = ""
    company_name:         str             = ""
    location:             str             = ""
    salary:               str             = ""
    mode:                 str             = ""
    job_description:      str             = ""
    posted_date:          str             = ""
    recruiter_name:       str             = ""
    recruiter_title:      str             = ""
    recruiter_contact:    str             = ""
    recruiter_email:      str             = ""
    application_deadline: str             = ""
    company_size:         str             = ""
    company_ratings:      Optional[float] = None
    extra_fields:         dict            = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base connector that every platform must subclass
# ---------------------------------------------------------------------------

class BaseConnector(ABC):
    """
    Abstract base for all platform scrapers.

    Required overrides:
        - platform_name      (property)
        - search_urls()
        - extract_job_links()
        - extract_job_data()

    Optional overrides:
        - handle_popups()
        - pre_search_hook()
        - post_extract_hook()
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict from .env. Always contains:
                JOBS_PER_QUERY, MIN_DELAY, MAX_DELAY
        """
        self.config = config
        self.jobs_per_query: int   = int(config.get("JOBS_PER_QUERY", 10))
        self.min_delay:      float = float(config.get("MIN_DELAY", 4))
        self.max_delay:      float = float(config.get("MAX_DELAY", 8))

    # -- Required overrides --------------------------------------------------

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Human-readable platform name, used in DB records."""
        ...

    @abstractmethod
    def search_urls(self, keywords: list[str], location: str) -> list[str]:
        """
        Build a list of search page URLs for the given keywords and location.
        Called once before scraping starts.
        """
        ...

    @abstractmethod
    async def extract_job_links(self, page: Page) -> list[str]:
        """
        Given a loaded search results page, return all job detail URLs.
        Connector may scroll or click 'load more' before returning.
        """
        ...

    @abstractmethod
    async def extract_job_data(self, page: Page, url: str) -> Optional[RawJob]:
        """
        Given a loaded job detail page, return a RawJob.
        Return None if the page failed to load or is a login wall.
        """
        ...

    # -- Optional overrides --------------------------------------------------

    async def handle_popups(self, page: Page) -> None:
        """Dismiss modals, cookie banners, login walls. Default: no-op."""
        pass

    async def pre_search_hook(self, page: Page) -> None:
        """
        Runs once before the first search URL is opened.
        Use for: setting cookies, warming up the session.
        Default: no-op.
        """
        pass

    async def post_extract_hook(self, page: Page) -> None:
        """
        Runs after each job page is processed.
        Use for: closing modals, clearing state.
        Default: no-op.
        """
        pass