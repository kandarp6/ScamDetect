"""
url_scraper.py
Playwright-based URL scraper to extract job details for live analysis.
Integrates official platform connectors (Internshala, LinkedIn, Shine, NCS)
for robust multi-platform extraction.
"""

import re
import asyncio
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from loguru import logger

from backend.scraper.connectors import CONNECTOR_REGISTRY
from backend.scraper.connectors.base import RawJob

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
]

def detect_platform_key(url: str) -> str:
    """Extract platform key mapping to CONNECTOR_REGISTRY."""
    domain = urlparse(url).netloc.lower()
    if "internshala" in domain:
        return "internshala"
    elif "linkedin" in domain:
        return "linkedin"
    elif "shine" in domain:
        return "shine"
    elif "ncs.gov" in domain:
        return "ncs"
    return "other"

async def scrape_job_url(url: str) -> dict:
    """
    Launch Playwright, detect platform, use matching connector for parsing.
    Falls back to generic metadata parsing if no specific connector matches.
    """
    logger.info(f"Scraping URL: {url}")
    platform_key = detect_platform_key(url)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = await browser.new_context(
            user_agent=USER_AGENTS[0],
            viewport={"width": 1280, "height": 800},
            locale="en-IN",
            timezone_id="Asia/Kolkata",
        )
        page = await context.new_page()
        
        # Abort heavy assets to save system resources and prevent crashes
        async def block_heavy_resources(route):
            if route.request.resource_type in ["image", "media", "font"]:
                await route.abort()
            else:
                await route.continue_()
        await page.route("**/*", block_heavy_resources)
        
        try:


            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            await asyncio.sleep(2)
            
            # Phase 1: Try matched connector
            if platform_key in CONNECTOR_REGISTRY:
                try:
                    logger.info(f"Using registered connector for {platform_key} to parse URL.")
                    connector_class = CONNECTOR_REGISTRY[platform_key]
                    connector = connector_class({})
                    raw_job = await connector.extract_job_data(page, url)
                    
                    if raw_job:
                        return {
                            "job_title": raw_job.job_title or "Unknown Title",
                            "job_description": raw_job.job_description or "No description extracted.",
                            "company_name": raw_job.company_name or "Unknown Company",
                            "platform_name": connector.platform_name,
                            "salary_raw": raw_job.salary or "Not Specified",
                            "city": raw_job.location or "India",
                        }
                except Exception as conn_err:
                    logger.warning(f"Connector parsing failed: {conn_err}. Falling back to generic parser.")
            
            # Phase 2: Generic Selector Fallback
            logger.info("Running generic selector fallback parser.")
            title = ""
            meta_title = await page.locator("meta[property='og:title']").get_attribute("content")
            if meta_title:
                title = meta_title.strip()
            else:
                title = await page.title()
            
            title = re.sub(r'\s*\|\s*.*$', '', title)
            title = re.sub(r'\s*-\s*.*$', '', title)
            title = title.strip()
            
            description = ""
            meta_desc = await page.locator("meta[property='og:description']").get_attribute("content")
            if meta_desc:
                description = meta_desc.strip()
            
            desc_selectors = [
                "#about_internship .text-container",
                ".description__text",
                ".job-description",
                ".job_description",
                "#job-description",
                ".desc",
                "main",
                "body"
            ]
            for selector in desc_selectors:
                try:
                    locator = page.locator(selector)
                    if await locator.count() > 0:
                        text = await locator.first.inner_text()
                        if text and len(text.strip()) > len(description):
                            description = text.strip()
                            break
                except Exception:
                    continue
            
            company = ""
            company_selectors = [
                ".company_name a",
                "company",
                ".company",
                ".topcard__org-name-link",
                "[class*='company']"
            ]
            for selector in company_selectors:
                try:
                    locator = page.locator(selector)
                    if await locator.count() > 0:
                        text = await locator.first.inner_text()
                        if text:
                            company = text.strip()
                            break
                except Exception:
                    continue
            
            location = ""
            loc_selectors = [
                ".location",
                "[class*='location']",
                ".location_link",
                ".topcard__flavor--bullet"
            ]
            for selector in loc_selectors:
                try:
                    locator = page.locator(selector)
                    if await locator.count() > 0:
                        text = await locator.first.inner_text()
                        if text:
                            location = text.strip()
                            break
                except Exception:
                    continue
            
            salary = ""
            sal_selectors = [
                ".stipend",
                ".salary",
                "[class*='salary']",
                "[class*='stipend']"
            ]
            for selector in sal_selectors:
                try:
                    locator = page.locator(selector)
                    if await locator.count() > 0:
                        text = await locator.first.inner_text()
                        if text:
                            salary = text.strip()
                            break
                except Exception:
                    continue
            
            return {
                "job_title": title or "Unknown Title",
                "job_description": description or "No description extracted.",
                "company_name": company or "Unknown Company",
                "platform_name": platform_key.capitalize(),
                "salary_raw": salary or "Not Specified",
                "city": location or "India",
            }
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return {
                "job_title": "Scrape Failed",
                "job_description": f"Failed to retrieve page contents. Error: {str(e)}",
                "company_name": "Unknown",
                "platform_name": "Other",
                "salary_raw": "Not Specified",
                "city": "Unknown",
            }
        finally:
            await browser.close()
