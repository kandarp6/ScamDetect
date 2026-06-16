"""
test_engines.py
Unit tests for Graphura scraper processing engines.
"""

import unittest
from backend.scraper.engines.recruiter_verifier import verify_recruiter
from backend.scraper.engines.salary_parser import parse_salary
from backend.scraper.engines.location_normalizer import normalize_location

class TestEngines(unittest.TestCase):

    def test_recruiter_verifier(self):
        # Test high trust recruiter
        score, flags = verify_recruiter(
            name="John Doe",
            title="HR Manager",
            email_domain="corporate.com",
            linkedin_url="https://linkedin.com/in/johndoe",
            company_domain="corporate.com"
        )
        self.assertGreaterEqual(score, 60)
        
        # Test low trust / scam generic recruiter
        score_scam, flags_scam = verify_recruiter(
            name="HR Team",
            title="",
            email_domain="gmail.com",
            linkedin_url="",
            company_domain=""
        )
        self.assertLess(score_scam, 40)
        self.assertTrue("personal_email" in flags_scam or "generic_name" in flags_scam)

    def test_salary_parser(self):
        # Test standard monthly salary (annualized: 25k * 12 = 300k)
        res = parse_salary("Rs. 25,000 / month")
        self.assertEqual(res.min_amount, 300000.0)
        self.assertEqual(res.max_amount, 300000.0)
        self.assertFalse(res.is_suspicious)
        
        # Test high suspicious salary
        res_susp = parse_salary("Earn 50000 daily")
        self.assertTrue(res_susp.is_suspicious)




    def test_location_normalizer(self):
        loc = normalize_location("Delhi NCR, India")
        self.assertTrue(loc.city in ["Delhi", "New Delhi"])
        self.assertEqual(loc.country, "India")

if __name__ == "__main__":
    unittest.main()
