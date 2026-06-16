// API Configuration

const ENV = {
  development: {
    BACKEND_URL: "http://localhost:8000",
  },
  production: {
    BACKEND_URL: "https://fake-internship-scam-job-detection-api.onrender.com",
  }
};

const isLocalhost = window.location.hostname === "localhost" ||
                    window.location.hostname === "127.0.0.1";

const CONFIG = isLocalhost ? ENV.development : ENV.production;

window.APP_CONFIG = CONFIG;
console.log("Using backend:", CONFIG.BACKEND_URL);
