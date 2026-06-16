"""
test_api.py
Integration tests for Graphura FastAPI endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from backend.main import app

class TestApi(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_serves_frontend(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("<html", response.text)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")


    def test_stats_endpoint(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_jobs", data)
        self.assertIn("scams_detected", data)
        self.assertIn("verified_recruiters", data)
        self.assertIn("reports_filed", data)

    def test_analyze_job_endpoint(self):
        job_payload = {
            "job_title": "Software Engineer",
            "job_description": "We are looking for a Python developer with 2+ years of experience. Job responsibilities include backend APIs and clean code.",
            "company_name": "Tech Corp",
            "platform_name": "LinkedIn",
            "salary_raw": "50,000 / month",
            "city": "Mumbai"
        }
        response = self.client.post("/api/analyze/job", json=job_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("score", data)
        self.assertIn("risk_level", data)
        self.assertIn("is_scam", data)

    def test_verify_recruiter_endpoint(self):
        payload = {
            "name": "Jane Smith",
            "company": "Google",
            "linkedin_url": "https://linkedin.com/in/janesmith"
        }
        response = self.client.post("/api/verify/recruiter", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("verified", data)
        self.assertIn("score", data)

    def test_report_endpoint(self):
        payload = {
            "job_url": "https://example.com/fake-job",
            "job_description": "Suspected fake data entry job requesting money upfront.",
            "company_name": "Fake Data Inc",
            "contact_method": "WhatsApp",
            "experience": "No experience needed, quick money.",
            "contact": "+91-9876543210"
        }
        response = self.client.post("/api/report", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "received")

if __name__ == "__main__":
    unittest.main()
