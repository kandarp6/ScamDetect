// API Client

const API_BASE = window.APP_CONFIG.BACKEND_URL;

class ApiClient {

  async request(endpoint, options) {
    options = options || {};
    const url = API_BASE + endpoint;

    try {
      const response = await fetch(url, {
        method: options.method || "GET",
        headers: {
          "Content-Type": "application/json"
        },
        body: options.body || null,
      });

      if (!response.ok) {
        const error = await response.json().catch(function() { return {}; });
        throw new Error(error.detail || "HTTP " + response.status);
      }

      return await response.json();
    } catch (err) {
      console.error("API error [" + endpoint + "]:", err);
      throw err;
    }
  }

  analyzeJob(jobData) {
    return this.request("/api/analyze/job", {
      method: "POST",
      body: JSON.stringify(jobData),
    });
  }

  analyzeUrl(url) {
    return this.request("/api/analyze/url", {
      method: "POST",
      body: JSON.stringify({ url: url }),
    });
  }

  verifyRecruiter(name, company, linkedinUrl) {
    return this.request("/api/verify/recruiter", {
      method: "POST",
      body: JSON.stringify({
        name: name,
        company: company || "",
        linkedin_url: linkedinUrl || ""
      }),
    });
  }

  submitReport(reportData) {
    return this.request("/api/report", {
      method: "POST",
      body: JSON.stringify(reportData),
    });
  }

  scrapeJobs(platform, keywords, limit) {
    return this.request("/api/scrape", {
      method: "POST",
      body: JSON.stringify({
        platform: platform,
        keywords: keywords,
        limit: parseInt(limit) || 3
      })
    });
  }

  getStats() {
    return this.request("/api/stats");
  }

  getRecentAnalyses(limit) {
    return this.request("/api/jobs/recent?limit=" + (limit || 5));
  }

  getAllJobs() {
    return this.request("/api/jobs");
  }
}

window.api = new ApiClient();


