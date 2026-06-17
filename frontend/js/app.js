// Navigation

function showPage(id, el) {
  var pages = document.querySelectorAll(".page");
  for (var i = 0; i < pages.length; i++) pages[i].classList.remove("active");
  var tabs = document.querySelectorAll(".ntab");
  for (var j = 0; j < tabs.length; j++) tabs[j].classList.remove("active");
  document.getElementById("p-" + id).classList.add("active");
  if (el) el.classList.add("active");
  else {
    var tab = document.querySelector(".ntab[data-page=" + id + "]");
    if (tab) tab.classList.add("active");
  }
  if (id === "home") {
    loadHomeStats();
    loadRecentAnalyses();
  } else if (id === "dash") {
    loadDashboardRecords();
  }

}

function goCheck() {
  var v = document.getElementById("home-url").value;
  if (v) document.getElementById("check-url").value = v;
  showPage("checker");
}

// Scam Check

async function runCheck() {
  var url = document.getElementById("check-url").value.trim();
  var desc = document.getElementById("check-desc").value.trim();
  var platform = document.getElementById("check-source").value;

  if (!url && !desc) {
    alert("Please enter a job URL or paste a description first.");
    return;
  }

  document.getElementById("result-empty").style.display = "none";
  var card = document.getElementById("result-card");
  card.style.display = "block";
  document.getElementById("ring-num").textContent = "...";
  document.getElementById("result-verdict").textContent = "Analyzing...";
  document.getElementById("result-summary").textContent = "";
  document.getElementById("kw-area").innerHTML = "";
  document.getElementById("alerts-area").innerHTML = "";
  document.getElementById("explain-text").textContent = "Pending...";

  try {
    var result;
    if (url) {
      result = await window.api.analyzeUrl(url);
    } else {
      result = await window.api.analyzeJob({
        job_description: desc,
        platform_name: platform,
      });
    }
    renderAnalysisResult(result);
  } catch (err) {
    document.getElementById("result-verdict").textContent = "Analysis failed";
    document.getElementById("result-summary").textContent = err.message;
    document.getElementById("ring-num").textContent = "X";
    document.getElementById("ring-num").style.color = "var(--ink-3)";
  }
}

function renderAnalysisResult(result) {
  var score = Math.round(result.score || 0);
  var hi = score > 65;
  var mid = score > 40;
  var color = hi ? "var(--red)" : mid ? "var(--amber)" : "var(--green)";

  document.getElementById("ring-num").textContent = score;
  document.getElementById("ring-num").style.color = color;
  var circ = 2 * Math.PI * 38;
  document.getElementById("ring-stroke").style.strokeDashoffset = circ - (circ * score / 100);
  document.getElementById("ring-stroke").style.stroke = hi ? "#C0381A" : mid ? "#9A6500" : "#1A6B3A";

  document.getElementById("result-verdict").textContent = result.risk_level || "Unknown";
  document.getElementById("result-verdict").style.color = color;
  document.getElementById("result-summary").textContent = result.summary || "";

  var signals = result.signals || {};
  setBar("sig-lang", signals.language_risk || 0, hi ? "var(--red)" : "var(--green)");
  setBar("sig-sal", signals.salary_risk || 0, mid ? "var(--amber)" : "var(--green)");
  setBar("sig-co", signals.company_risk || 0, hi ? "var(--red)" : "var(--green)");
  setBar("sig-con", signals.contact_risk || 0, mid ? "var(--amber)" : "var(--green)");
  setBar("sig-req", signals.requirements_risk || 0, mid ? "var(--amber)" : "var(--green)");

  var area = document.getElementById("kw-area");
  area.innerHTML = "";
  (result.keywords || []).forEach(function(k) {
    var c = document.createElement("span");
    c.className = "kchip " + (k.is_red_flag ? "k-red" : "k-neutral");
    c.textContent = k.keyword;
    c.onclick = function() {
      document.getElementById("kw-tooltip").textContent = k.explanation;
    };
    area.appendChild(c);
  });

  var aEl = document.getElementById("alerts-area");
  aEl.innerHTML = (result.alerts || []).map(function(a) {
    return '<div class="alert a-' + (a.severity || "amber") + '">' +
      '<svg class="alert-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>' +
      '<div><div class="alert-title">' + a.title + '</div>' +
      '<div class="alert-sub">' + a.message + '</div></div></div>';
  }).join("");

  document.getElementById("explain-text").innerHTML = result.explanation || "";
}

function setBar(id, pct, color) {
  var el = document.getElementById(id);
  var vEl = document.getElementById(id + "-v");
  el.style.background = color;
  setTimeout(function() {
    el.style.width = Math.min(pct, 100) + "%";
    if (vEl) vEl.textContent = Math.min(Math.round(pct), 100);
  }, 60);
}

// Recruiter

async function runRecruiter() {
  var name = document.getElementById("rec-name").value.trim();
  var co = document.getElementById("rec-company").value.trim();
  var linkedin = document.getElementById("rec-linkedin").value.trim();

  if (!name) {
    alert("Please enter the recruiter's name.");
    return;
  }

  var el = document.getElementById("rec-result");
  el.innerHTML = '<div class="card" style="text-align:center;padding:48px"><div class="dots"><span></span><span></span><span></span></div></div>';

  try {
    var result = await window.api.verifyRecruiter(name, co, linkedin);
    renderRecruiterResult(result, name, co);
  } catch (err) {
    el.innerHTML = '<div class="card"><div class="alert a-red"><div><div class="alert-title">Failed</div><div class="alert-sub">' + err.message + '</div></div></div></div>';
  }
}

function renderRecruiterResult(result, name, co) {
  var safe = result.verified || result.score >= 60;
  var checks = (result.checks || []).map(function(check) {
    var cls = check.status === "pass" ? "b-green" : check.status === "fail" ? "b-red" : "b-amber";
    return '<div class="rec-row"><span>' + check.label + '</span><span class="badge ' + cls + '">' + check.value + '</span></div>';
  }).join("");

  document.getElementById("rec-result").innerHTML =
    '<div class="card">' +
      '<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">' +
        '<div style="flex:1"><div style="font-size:15px;font-weight:700">' + name + '</div>' +
        '<div style="font-size:12px;color:var(--ink-3)">' + (co || "Unknown") + '</div></div>' +
        '<span class="badge ' + (safe ? "b-green" : "b-red") + '">' + (safe ? "Verified" : "Flagged") + '</span>' +
      '</div>' +
      '<div class="divider"></div>' + checks +
      '<div class="divider"></div>' +
      '<div class="alert ' + (safe ? "a-green" : "a-red") + '">' +
        '<svg class="alert-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>' +
        '<div><div class="alert-title">' + (safe ? "Looks legitimate" : "Caution advised") + '</div>' +
        '<div class="alert-sub">' + (result.message || "") + '</div></div></div>' +
    '</div>';
}

// Report

async function submitReport() {
  var desc = document.getElementById("rep-desc").value.trim();
  if (!desc) {
    alert("Please describe the suspicious job.");
    return;
  }

  try {
    await window.api.submitReport({
      job_description: desc,
      company_name: document.getElementById("rep-company").value.trim(),
      contact_method: document.getElementById("rep-contact-method").value,
      experience: document.getElementById("rep-experience").value.trim(),
      contact: document.getElementById("rep-contact").value.trim(),
    });
    alert("Report submitted! Thank you.");
    document.getElementById("rep-desc").value = "";
    document.getElementById("rep-company").value = "";
    document.getElementById("rep-experience").value = "";
    document.getElementById("rep-contact").value = "";
  } catch (err) {
    alert("Could not submit: " + err.message);
  }
}

// Checklist

var CHECKLIST = [
  { q: "Was an upfront payment requested?", w: 25 },
  { q: "Was contact only via WhatsApp?", w: 20 },
  { q: "Is the salary too high for the role?", w: 18 },
  { q: "Cannot find the company on Google?", w: 15 },
  { q: "Was there no formal interview?", w: 12 },
  { q: "Did the offer arrive unsolicited?", w: 10 },
  { q: "Are requirements vague?", w: 8 },
  { q: "Was the offer letter informal?", w: 7 },
  { q: "Were you pressured to decide quickly?", w: 7 },
  { q: "Many spelling errors in description?", w: 5 },
];

var clState = {};
CHECKLIST.forEach(function(_, i) { clState[i] = null; });

function buildChecklist() {
  var wrap = document.getElementById("checklist-items");
  wrap.innerHTML = CHECKLIST.map(function(item, i) {
    return '<div class="check-item">' +
      '<div class="check-text">' + item.q + '<div class="check-weight">Weight: +' + item.w + '</div></div>' +
      '<div class="yn-wrap">' +
        '<button class="yn-btn" id="y' + i + '" onclick="clClick(' + i + ',true)">Yes</button>' +
        '<button class="yn-btn" id="n' + i + '" onclick="clClick(' + i + ',false)">No</button>' +
      '</div></div>';
  }).join("");
}

function clClick(i, val) {
  clState[i] = val;
  document.getElementById("y" + i).className = "yn-btn" + (val === true ? " yes-active" : "");
  document.getElementById("n" + i).className = "yn-btn" + (val === false ? " no-active" : "");
  updateClScore();
}

function updateClScore() {
  var score = 0;
  CHECKLIST.forEach(function(item, i) { if (clState[i] === true) score += item.w; });
  score = Math.min(score, 100);
  document.getElementById("cl-score-num").textContent = score;
  var bar = document.getElementById("cl-bar");
  var color = score > 65 ? "var(--red)" : score > 40 ? "var(--amber)" : "var(--green)";
  bar.style.width = score + "%";
  bar.style.background = color;
  var v = document.getElementById("cl-verdict");
  v.textContent = score > 65 ? "Very likely scam" : score > 40 ? "Proceed with caution" : "Looks safe";
  v.style.color = color;
}

function resetChecklist() {
  CHECKLIST.forEach(function(_, i) { clState[i] = null; });
  buildChecklist();
  updateClScore();
}

// Stats loaders

async function loadHomeStats() {
  try {
    var stats = await window.api.getStats();
    var setStat = function(key, val) {
      var els = document.querySelectorAll("[data-stat='" + key + "']");
      for (var i = 0; i < els.length; i++) els[i].textContent = (val || 0).toLocaleString();
    };
    setStat("total_jobs", stats.total_jobs);
    setStat("scams_detected", stats.scams_detected);
    setStat("verified_recruiters", stats.verified_recruiters);
    setStat("reports_filed", stats.reports_filed);
  } catch (err) {
    console.warn("Stats load failed:", err);
  }
}

async function loadRecentAnalyses() {
  try {
    var data = await window.api.getRecentAnalyses(5);
    var list = document.getElementById("recent-analyses-list");
    if (!list) return;
    if (!data.jobs || data.jobs.length === 0) {
      list.innerHTML = '<div class="hist-row"><div class="hist-title" style="color:var(--ink-3)">No analyses yet</div></div>';
      return;
    }
    list.innerHTML = data.jobs.map(function(job) {
      var score = Math.round(job.scam_score || 0);
      var color = score > 65 ? "var(--red)" : score > 40 ? "var(--amber)" : "var(--green)";
      var badge = score > 65 ? "b-red" : score > 40 ? "b-amber" : "b-green";
      return '<div class="hist-row" onclick="viewRecentDetail(\'' + job.id + '\')">' +
        '<div class="hist-dot" style="background:' + color + '"></div>' +
        '<div class="hist-title">' + (job.job_title || "Untitled") + '</div>' +
        '<div class="hist-score ' + badge + '">' + score + '</div>' +
      '</div>';
    }).join("");
  } catch (err) {
    console.warn("Recent jobs load failed:", err);
  }

}



// Chatbot Logic
function toggleChat() {
  var win = document.getElementById("chat-win");
  win.classList.toggle("open");
}

function chatSend(text) {
  if (!text) return;
  text = text.trim();
  if (!text) return;
  
  var inp = document.getElementById("chat-inp");
  if (inp) inp.value = "";
  
  var msgs = document.getElementById("chat-msgs");
  
  // Append user message
  var uMsg = document.createElement("div");
  uMsg.className = "chat-msg user-msg";
  uMsg.textContent = text;
  msgs.appendChild(uMsg);
  msgs.scrollTop = msgs.scrollHeight;
  
  // Bot responses
  setTimeout(function() {
    var response = "I'm not sure about that. Try checking the job description in the 'Scam Checker' tab for a full analysis.";
    var t = text.toLowerCase();
    
    if (t.includes("fake offer") || t.includes("offer letter")) {
      response = "Fake offer letters often have spelling mistakes, use generic templates, request a 'registration fee', and come from public email domains (like Gmail) instead of official corporate ones.";
    } else if (t.includes("remote") || t.includes("work from home") || t.includes("wfh")) {
      response = "Legitimate remote work exists, but be highly suspicious if the job pays excessively well for simple tasks (like data entry/typing) or demands upfront payment for software/training.";
    } else if (t.includes("scammed") || t.includes("lost money")) {
      response = "If you have been scammed, immediately block the sender, document all chats/payments, and file a formal report on the government portal <strong>cybercrime.gov.in</strong> or call <strong>1930</strong>.";
    } else if (t.includes("verify") || t.includes("recruiter")) {
      response = "You can verify recruiters using the 'Verify Recruiter' tab by entering their name and company to check if they match safety records.";
    }
    
    var bMsg = document.createElement("div");
    bMsg.className = "chat-msg bot-msg";
    bMsg.innerHTML = response;
    msgs.appendChild(bMsg);
    msgs.scrollTop = msgs.scrollHeight;
  }, 600);
}

// Share Logic
function updatePreview() {
  var title = document.getElementById("share-title").value.trim() || "Suspected job fraud";
  var platform = document.getElementById("share-platform").value;
  var flags = document.getElementById("share-flags").value.trim() || "Type details on the left to populate this alert...";

  document.getElementById("prev-title").textContent = title;
  document.getElementById("prev-platform").textContent = "Platform: " + platform;
  document.getElementById("prev-flags").textContent = flags;
}

function triggerShare(dest) {
  var title = document.getElementById("share-title").value.trim() || "Suspected job fraud";
  var platform = document.getElementById("share-platform").value;
  var flags = document.getElementById("share-flags").value.trim() || "Type details on the left to populate this alert...";

  var text = "⚠️ SCAM ALERT: " + title + " on " + platform + "\n\nRed flags:\n" + flags + "\n\nCheck suspicious jobs at Graphura before applying!";

  if (dest === "copy") {
    navigator.clipboard.writeText(text).then(function() {
      alert("Alert text copied to clipboard!");
    }, function() {
      alert("Failed to copy text.");
    });
  } else if (dest === "whatsapp") {
    window.open("https://api.whatsapp.com/send?text=" + encodeURIComponent(text), "_blank");
  } else {
    alert("Sharing preview:\n\n" + text);
  }
}

function shareResult(dest) {
  var score = document.getElementById("ring-num").textContent;
  var verdict = document.getElementById("result-verdict").textContent;
  var summary = document.getElementById("result-summary").textContent;
  
  var text = "📊 Job Fraud Detection Report\nRisk Score: " + score + "/100\nVerdict: " + verdict + "\nSummary: " + summary + "\n\nAnalyze other jobs at Graphura!";

  if (dest === "copy") {
    navigator.clipboard.writeText(text).then(function() {
      alert("Analysis copied to clipboard!");
    });
  } else if (dest === "whatsapp") {
    window.open("https://api.whatsapp.com/send?text=" + encodeURIComponent(text), "_blank");
  }
}

// Dashboard Page Logic
var _loadedJobs = [];

async function loadDashboardRecords() {
  showDashMain();
  var tbody = document.getElementById("dash-records-body");
  if (tbody) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:20px;color:var(--ink-3)">Loading audit logs...</td></tr>';
  }
  
  try {
    var data = await window.api.getAllJobs();
    _loadedJobs = data.jobs || [];
    renderDashboardRecords(_loadedJobs);
  } catch (err) {
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:20px;color:var(--red)">Failed to load records: ' + err.message + '</td></tr>';
    }
  }
}

function renderDashboardRecords(jobs) {
  var tbody = document.getElementById("dash-records-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  
  if (jobs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:20px;color:var(--ink-3)">No records found.</td></tr>';
    return;
  }
  
  var platformCounts = {};
  var safeCount = 0, medCount = 0, highCount = 0;
  
  jobs.forEach(function(job) {
    var score = Math.round(job.scam_score || 0);
    var risk = job.scam_risk_level || "Safe";
    var plat = job.platform_name || "Other";
    
    platformCounts[plat] = (platformCounts[plat] || 0) + 1;
    
    if (score > 65 || risk === "High Risk" || risk === "Scam Likely") {
      highCount++;
    } else if (score > 40 || risk === "Medium Risk") {
      medCount++;
    } else {
      safeCount++;
    }
    
    var tr = document.createElement("tr");
    var dateStr = job.created_at ? new Date(job.created_at).toLocaleDateString() : "N/A";
    var color = score > 65 ? "var(--red)" : score > 40 ? "var(--amber)" : "var(--green)";
    var badgeCls = score > 65 ? "b-red" : score > 40 ? "b-amber" : "b-green";
    
    tr.innerHTML = 
      '<td><strong>' + (job.job_title || "Untitled") + '</strong></td>' +
      '<td>' + (job.companies?.name || "Unknown") + '</td>' +
      '<td><span class="badge b-blue">' + plat + '</span></td>' +
      '<td><strong style="color:' + color + ';font-family:var(--mono)">' + score + '/100</strong></td>' +
      '<td><span class="badge ' + badgeCls + '">' + risk + '</span></td>' +
      '<td>' + dateStr + '</td>' +
      '<td><button class="btn" onclick="viewAuditRecord(\'' + job.id + '\')" style="padding:4px 8px;font-size:11px"><i class="fas fa-chart-pie"></i> View Report</button></td>';
      
    tbody.appendChild(tr);
  });
  
  var total = jobs.length;
  var pbHtml = Object.keys(platformCounts).map(function(plat) {
    var cnt = platformCounts[plat];
    var pct = Math.round((cnt / total) * 100);
    var color = pct > 50 ? "var(--red)" : pct > 20 ? "var(--amber)" : "var(--green)";
    return '<div class="hbar-row"><div class="hbar-label">' + plat + '</div><div class="hbar-track"><div class="hbar-fill" style="width:' + pct + '%;background:' + color + '"></div></div><div class="hbar-val">' + pct + '%</div></div>';
  }).join("");
  
  var pbContainer = document.getElementById("platform-breakdown-bars");
  if (pbContainer) pbContainer.innerHTML = pbHtml;
  
  var setDistBar = function(idBar, idVal, cnt) {
    var pct = total ? Math.round((cnt / total) * 100) : 0;
    var bar = document.getElementById(idBar);
    var val = document.getElementById(idVal);
    if (bar) bar.style.width = pct + "%";
    if (val) val.textContent = cnt + " (" + pct + "%)";
  };
  setDistBar("dist-safe-bar", "dist-safe-val", safeCount);
  setDistBar("dist-med-bar", "dist-med-val", medCount);
  setDistBar("dist-high-bar", "dist-high-val", highCount);
}

function showDashMain() {
  document.getElementById("dash-main").style.display = "block";
  document.getElementById("dash-details").style.display = "none";
}

function viewAuditRecord(jobId) {
  var job = _loadedJobs.find(function(j) { return j.id === jobId; });
  if (!job) return;
  
  document.getElementById("dash-main").style.display = "none";
  document.getElementById("dash-details").style.display = "block";
  
  var score = Math.round(job.scam_score || 0);
  var hi = score > 65;
  var mid = score > 40;
  var color = hi ? "var(--red)" : mid ? "var(--amber)" : "var(--green)";
  var badgeCls = hi ? "b-red" : mid ? "b-amber" : "b-green";
  var verdict = job.scam_risk_level || "Safe";
  
  document.getElementById("dash-detail-score").textContent = score;
  document.getElementById("dash-detail-score").style.color = color;
  
  var circ = 2 * Math.PI * 38;
  var ring = document.getElementById("dash-detail-ring");
  if (ring) {
    ring.style.strokeDashoffset = circ - (circ * score / 100);
    ring.style.stroke = hi ? "#C0381A" : mid ? "#9A6500" : "#1A6B3A";
  }
  
  document.getElementById("dash-detail-title").textContent = job.job_title || "Untitled";
  document.getElementById("dash-detail-platform").textContent = job.platform_name || "Other";
  
  var vBadge = document.getElementById("dash-detail-verdict");
  if (vBadge) {
    vBadge.className = "badge " + badgeCls;
    vBadge.textContent = verdict;
  }
  
  document.getElementById("dash-detail-company").innerHTML = 
    'Company: <strong>' + (job.companies?.name || "Unknown") + '</strong> (Trust Score: ' + (job.companies?.company_trust_score || 0) + '/100)';
    
  var sigLang = Math.min(score * 0.9, 100);
  var sigSal = job.salary_raw ? (hi ? 80 : 30) : 0;
  var sigCo = 100 - (job.companies?.company_trust_score || 50);
  
  setBar("dash-sig-lang", sigLang, hi ? "var(--red)" : "var(--green)");
  setBar("dash-sig-sal", sigSal, mid ? "var(--amber)" : "var(--green)");
  setBar("dash-sig-co", sigCo, sigCo > 60 ? "var(--red)" : "var(--green)");
  
  var rfArea = document.getElementById("dash-detail-risk-factors");
  if (rfArea) {
    if (job.risk_factors && job.risk_factors.length > 0) {
      rfArea.innerHTML = job.risk_factors.map(function(rf) {
        return '<div style="margin-bottom:6px;display:flex;align-items:center;gap:6px"><span style="width:6px;height:6px;background:var(--red);border-radius:50%"></span>' + rf + '</div>';
      }).join("");
    } else {
      rfArea.innerHTML = '<div style="color:var(--ink-3)">No suspicious signals detected.</div>';
    }
  }
  
  var suggArea = document.getElementById("dash-detail-suggestions");
  if (suggArea) {
    var suggHtml = "";
    if (score > 65) {
      suggHtml = '<strong>⚠️ HIGH RISK WARNING</strong><br>This listing matches known job scam patterns. DO NOT pay any registration, training, or security fees. Block any contact via WhatsApp/Telegram and file a report.';
      suggArea.className = "explainer a-red";
    } else if (score > 40) {
      suggHtml = '<strong>⚠️ CAUTION ADVISED</strong><br>Independent verification recommended. Cross-check the recruiter on LinkedIn or query the company name on the Ministry of Corporate Affairs (MCA21) portal before engaging.';
      suggArea.className = "explainer a-amber";
    } else {
      suggHtml = '<strong>✅ LOOKS LEGITIMATE</strong><br>No major warning signs detected. You can apply using standard safety precautions.';
      suggArea.className = "explainer a-green";
    }
    suggArea.innerHTML = suggHtml;
  }
  
  var descArea = document.getElementById("dash-detail-desc");
  if (descArea) descArea.textContent = job.job_description || "No description provided.";
}

async function viewRecentDetail(jobId) {
  showPage('dash');
  setTimeout(async function() {
    if (_loadedJobs.length === 0) {
      await loadDashboardRecords();
    }
    viewAuditRecord(jobId);
  }, 120);
}

// Init
buildChecklist();
loadHomeStats();
loadRecentAnalyses();



