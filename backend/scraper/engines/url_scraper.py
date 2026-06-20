#url_scraper.py


import os

if os.name == "nt":
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = r"d:\pw-browsers"

import re
import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from loguru import logger

from backend.scraper.connectors import CONNECTOR_REGISTRY
from backend.scraper.connectors.base import RawJob


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
]


def detect_platform_key(url: str) -> str:
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


async def scrape_with_bs4(url: str) -> dict:
    headers = {
        "User-Agent": USER_AGENTS[0],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        logger.info(f"Attempting fast BeautifulSoup scrape for: {url}")
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning(f"BeautifulSoup request returned status code {response.status_code}")
                return None
            
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract Title
            title = ""
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
            if not title:
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text().strip()
            
            title = re.sub(r'\s*\|\s*.*$', '', title)
            title = re.sub(r'\s*-\s*.*$', '', title)
            title = title.strip()
            
            # Extract Description
            description = ""
            og_desc = soup.find("meta", property="og:description")
            if og_desc and og_desc.get("content"):
                description = og_desc["content"].strip()
            
            desc_css_selectors = [
                "#about_internship .text-container",
                ".description__text",
                ".job-description",
                ".job_description",
                "#job-description",
                ".desc"
            ]
            
            for selector in desc_css_selectors:
                el = soup.select_one(selector)
                if el:
                    text = el.get_text(separator="\n").strip()
                    if len(text) > len(description):
                        description = text
                        break
            
            # Extract Company
            company = ""
            company_selectors = [
                ".company_name a",
                "company",
                ".company",
                ".topcard__org-name-link",
            ]
            for selector in company_selectors:
                el = soup.select_one(selector)
                if el:
                    text = el.get_text().strip()
                    if text:
                        company = text
                        break
            
            # Extract Location
            location = ""
            loc_selectors = [
                ".location",
                ".location_link",
                ".topcard__flavor--bullet",
            ]
            for selector in loc_selectors:
                el = soup.select_one(selector)
                if el:
                    text = el.get_text().strip()
                    if text:
                        location = text
                        break
                        
            # Extract Salary
            salary = ""
            sal_selectors = [
                ".stipend",
                ".salary",
            ]
            for selector in sal_selectors:
                el = soup.select_one(selector)
                if el:
                    text = el.get_text().strip()
                    if text:
                        salary = text
                        break
            
            platform_key = detect_platform_key(url)
            
            # If description is substantial, return the parsed results
            if len(description) > 150:
                logger.info(f"Successfully scraped job URL using BeautifulSoup: {url}")
                return {
                    "job_title": title or "Unknown Title",
                    "job_description": description,
                    "company_name": company or "Unknown Company",
                    "platform_name": platform_key.capitalize(),
                    "salary_raw": salary or "Not Specified",
                    "city": location or "India",
                }
            
            logger.info("BeautifulSoup returned insufficient description. Falling back to Playwright.")
            return None
    except Exception as e:
        logger.warning(f"BeautifulSoup scraping failed: {e}. Falling back to Playwright.")
        return None


async def _scrape_page_with_context(context, url: str, platform_key: str) -> dict:
    page = await context.new_page()

    async def block_heavy_resources(route):
        if route.request.resource_type in ["image", "media", "font"]:
            await route.abort()
        else:
            await route.continue_()

    await page.route("**/*", block_heavy_resources)

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        
        # If a specialized connector exists, invoke it immediately (no hardcoded sleep).
        if platform_key in CONNECTOR_REGISTRY:
            try:
                logger.info(f"Using registered connector for {platform_key}.")
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
        else:
            # Give generic dynamic pages a brief moment to render JS before fallback parsing
            await asyncio.sleep(1.5)

        logger.info("Running generic selector fallback parser.")

        title = ""
        meta_title_loc = page.locator("meta[property='og:title']")
        try:
            if await meta_title_loc.count() > 0:
                meta_title = await meta_title_loc.get_attribute("content", timeout=1000)
                if meta_title:
                    title = meta_title.strip()
        except Exception:
            pass
            
        if not title:
            title = await page.title()

        title = re.sub(r'\s*\|\s*.*$', '', title)
        title = re.sub(r'\s*-\s*.*$', '', title)
        title = title.strip()

        description = ""
        meta_desc_loc = page.locator("meta[property='og:description']")
        try:
            if await meta_desc_loc.count() > 0:
                meta_desc = await meta_desc_loc.get_attribute("content", timeout=1000)
                if meta_desc:
                    description = meta_desc.strip()
        except Exception:
            pass

        desc_selectors = [
            "#about_internship .text-container",
            ".description__text",
            ".job-description",
            ".job_description",
            "#job-description",
            ".desc"
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
            "[class*='company']",
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
            ".topcard__flavor--bullet",
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
            "[class*='stipend']",
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
        try:
            await page.close()
        except Exception:
            pass


async def scrape_job_url(url: str, browser=None) -> dict:
    logger.info(f"Scraping URL: {url}")
    platform_key = detect_platform_key(url)
    
    # Try BeautifulSoup first, but ONLY for unknown/generic platforms ("other")
    if platform_key == "other":
        bs4_res = await scrape_with_bs4(url)
        if bs4_res:
            return bs4_res
    else:
        logger.info(f"Platform is '{platform_key}' (dynamic/protected). Bypassing BeautifulSoup.")
        
    if browser:
        logger.info("Using shared background Playwright browser instance...")
        context = await browser.new_context(
            user_agent=USER_AGENTS[0],
            viewport={"width": 1280, "height": 800},
            locale="en-IN",
            timezone_id="Asia/Kolkata",
        )
        try:
            return await _scrape_page_with_context(context, url, platform_key)
        finally:
            try:
                await context.close()
            except Exception:
                pass
    else:
        logger.info("Launching temporary Playwright browser instance...")
        async with async_playwright() as p:
            temp_browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = await temp_browser.new_context(
                user_agent=USER_AGENTS[0],
                viewport={"width": 1280, "height": 800},
                locale="en-IN",
                timezone_id="Asia/Kolkata",
            )
            try:
                return await _scrape_page_with_context(context, url, platform_key)
            finally:
                try:
                    await context.close()
                except Exception:
                    pass
                try:
                    await temp_browser.close()
                except Exception:
                    pass
