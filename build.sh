#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r backend/requirements.txt

# Install Playwright browser binaries for scraping
python -m playwright install chromium
