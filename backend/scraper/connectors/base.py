
#connectors.base

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from playwright.async_api import Page


# Shared enumerations

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


# Raw data container returned by each connector

@dataclass
class RawJob:
    
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


# Base connector that every platform must subclass

class BaseConnector(ABC):


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
       
        ...

    @abstractmethod
    def search_urls(self, keywords: list[str], location: str) -> list[str]:
       
        ...

    @abstractmethod
    async def extract_job_links(self, page: Page) -> list[str]:
        ...

    @abstractmethod
    async def extract_job_data(self, page: Page, url: str) -> Optional[RawJob]:
        
        ...

    # -- Optional overrides --------------------------------------------------

    async def handle_popups(self, page: Page) -> None:
        
        pass

    async def pre_search_hook(self, page: Page) -> None:
        
        pass

    async def post_extract_hook(self, page: Page) -> None:
     
        pass
