"""
skill_extractor.py
Extract structured skills from job titles and descriptions.

NLP techniques:
    - Tokenization with regex word boundaries (prevents false positives)
    - Case-insensitive matching
    - Alias mapping (e.g. "JS" -> "JavaScript")
    - Categorization (programming, frameworks, databases, etc.)
    - Longest-alias-first matching for accuracy

Output:
    {
        "skills":           ["Django", "PostgreSQL", "Python"],
        "skill_count":      3,
        "skill_categories": {"programming": ["Python"], ...}
    }
"""

import re


# ============================================================================
# SKILL DICTIONARY (alias -> canonical name)
# ============================================================================
# Many skills have multiple names. Map all variations to ONE canonical name.
# Critical for ML - otherwise "Python", "python", "py" become 3 features.
# ============================================================================

SKILL_DICT = {
    # Programming Languages
    "python":         "Python",
    "py":             "Python",
    "javascript":     "JavaScript",
    "js":             "JavaScript",
    "java script":    "JavaScript",
    "typescript":     "TypeScript",
    "ts":             "TypeScript",
    "java":           "Java",
    "java8":          "Java",
    "java 8":         "Java",
    "java11":         "Java",
    "c++":            "C++",
    "cpp":            "C++",
    "c#":             "C#",
    "csharp":         "C#",
    "c sharp":        "C#",
    "golang":         "Go",
    "go lang":        "Go",
    "rust":           "Rust",
    "php":            "PHP",
    "ruby":           "Ruby",
    "kotlin":         "Kotlin",
    "swift":          "Swift",
    "scala":          "Scala",
    "perl":           "Perl",
    "matlab":         "MATLAB",
    "r programming":  "R",
    "r lang":         "R",

    # Web Frameworks and Libraries
    "react":          "React",
    "reactjs":        "React",
    "react.js":       "React",
    "react js":       "React",
    "angular":        "Angular",
    "angularjs":      "Angular",
    "angular.js":     "Angular",
    "vue":            "Vue.js",
    "vuejs":          "Vue.js",
    "vue.js":         "Vue.js",
    "svelte":         "Svelte",
    "next":           "Next.js",
    "nextjs":         "Next.js",
    "next.js":        "Next.js",
    "nuxt":           "Nuxt.js",
    "nuxtjs":         "Nuxt.js",
    "node":           "Node.js",
    "nodejs":         "Node.js",
    "node.js":        "Node.js",
    "express":        "Express.js",
    "expressjs":      "Express.js",
    "django":         "Django",
    "flask":          "Flask",
    "fastapi":        "FastAPI",
    "fast api":       "FastAPI",
    "spring boot":    "Spring Boot",
    "springboot":     "Spring Boot",
    "spring":         "Spring",
    "laravel":        "Laravel",
    "ruby on rails":  "Ruby on Rails",
    "rails":          "Ruby on Rails",
    "asp.net":        "ASP.NET",
    "aspnet":         "ASP.NET",
    "dotnet":         ".NET",
    ".net":           ".NET",

    # Frontend Tech
    "html":           "HTML",
    "html5":          "HTML",
    "css":            "CSS",
    "css3":           "CSS",
    "sass":           "SASS",
    "scss":           "SASS",
    "tailwind":       "Tailwind CSS",
    "tailwindcss":    "Tailwind CSS",
    "bootstrap":      "Bootstrap",
    "material ui":    "Material-UI",
    "material-ui":    "Material-UI",
    "mui":            "Material-UI",
    "redux":          "Redux",
    "vuex":           "Vuex",
    "jquery":         "jQuery",

    # Databases
    "mysql":          "MySQL",
    "my sql":         "MySQL",
    "postgresql":     "PostgreSQL",
    "postgres":       "PostgreSQL",
    "psql":           "PostgreSQL",
    "mongodb":        "MongoDB",
    "mongo":          "MongoDB",
    "redis":          "Redis",
    "sqlite":         "SQLite",
    "oracle":         "Oracle DB",
    "oracle db":      "Oracle DB",
    "mssql":          "MS SQL",
    "sql server":     "MS SQL",
    "ms sql":         "MS SQL",
    "elasticsearch":  "Elasticsearch",
    "cassandra":      "Cassandra",
    "dynamodb":       "DynamoDB",
    "firebase":       "Firebase",
    "firestore":      "Firestore",
    "supabase":       "Supabase",
    "sql":            "SQL",
    "nosql":          "NoSQL",

    # Cloud and DevOps
    "aws":                    "AWS",
    "amazon web services":    "AWS",
    "azure":                  "Azure",
    "microsoft azure":        "Azure",
    "gcp":                    "GCP",
    "google cloud":           "GCP",
    "google cloud platform":  "GCP",
    "docker":                 "Docker",
    "kubernetes":             "Kubernetes",
    "k8s":                    "Kubernetes",
    "git":                    "Git",
    "github":                 "GitHub",
    "gitlab":                 "GitLab",
    "bitbucket":              "Bitbucket",
    "jenkins":                "Jenkins",
    "terraform":              "Terraform",
    "ansible":                "Ansible",
    "ci/cd":                  "CI/CD",
    "cicd":                   "CI/CD",
    "linux":                  "Linux",
    "bash":                   "Bash",
    "shell scripting":        "Shell Scripting",

    # Data and Machine Learning
    "machine learning":             "Machine Learning",
    "ml":                           "Machine Learning",
    "deep learning":                "Deep Learning",
    "dl":                           "Deep Learning",
    "artificial intelligence":      "AI",
    "ai":                           "AI",
    "data science":                 "Data Science",
    "data analysis":                "Data Analysis",
    "data analytics":               "Data Analytics",
    "tensorflow":                   "TensorFlow",
    "pytorch":                      "PyTorch",
    "keras":                        "Keras",
    "pandas":                       "Pandas",
    "numpy":                        "NumPy",
    "scikit-learn":                 "scikit-learn",
    "sklearn":                      "scikit-learn",
    "matplotlib":                   "Matplotlib",
    "seaborn":                      "Seaborn",
    "power bi":                     "Power BI",
    "powerbi":                      "Power BI",
    "tableau":                      "Tableau",
    "excel":                        "Excel",
    "ms excel":                     "Excel",
    "advanced excel":               "Excel",
    "nlp":                          "NLP",
    "natural language processing":  "NLP",
    "computer vision":              "Computer Vision",
    "cv":                           "Computer Vision",
    "opencv":                       "OpenCV",

    # Generative AI
    "generative ai":         "Generative AI",
    "gen ai":                "Generative AI",
    "genai":                 "Generative AI",
    "llm":                   "LLM",
    "large language model":  "LLM",
    "langchain":             "LangChain",
    "openai":                "OpenAI",
    "chatgpt":               "ChatGPT",
    "gpt":                   "GPT",
    "gpt-4":                 "GPT",
    "claude":                "Claude AI",
    "gemini":                "Gemini",
    "huggingface":           "Hugging Face",
    "hugging face":          "Hugging Face",
    "bert":                  "BERT",
    "transformers":          "Transformers",
    "prompt engineering":    "Prompt Engineering",
    "rag":                   "RAG",

    # Mobile Development
    "android":         "Android",
    "ios":             "iOS",
    "react native":    "React Native",
    "flutter":         "Flutter",
    "dart":            "Dart",
    "xamarin":         "Xamarin",
    "ionic":           "Ionic",

    # API and Backend
    "rest api":        "REST API",
    "restful":         "REST API",
    "graphql":         "GraphQL",
    "microservices":   "Microservices",
    "websockets":      "WebSockets",
    "grpc":            "gRPC",

    # Design and UX
    "figma":             "Figma",
    "adobe xd":          "Adobe XD",
    "photoshop":         "Photoshop",
    "illustrator":       "Illustrator",
    "ui/ux":             "UI/UX",
    "ui design":         "UI Design",
    "ux design":         "UX Design",
    "user experience":   "UX Design",
    "user interface":    "UI Design",

    # Business and Marketing
    "seo":                          "SEO",
    "search engine optimization":   "SEO",
    "sem":                          "SEM",
    "google analytics":             "Google Analytics",
    "google ads":                   "Google Ads",
    "facebook ads":                 "Facebook Ads",
    "salesforce":                   "Salesforce",
    "crm":                          "CRM",
    "content writing":              "Content Writing",
    "copywriting":                  "Copywriting",
    "digital marketing":            "Digital Marketing",
    "social media":                 "Social Media Marketing",
    "social media marketing":       "Social Media Marketing",
    "email marketing":              "Email Marketing",
    "sales":                        "Sales",
    "b2b":                          "B2B Sales",
    "b2c":                          "B2C Sales",

    # Soft Skills (limited - hard to detect accurately)
    "communication":     "Communication",
    "teamwork":          "Teamwork",
    "leadership":        "Leadership",
    "problem solving":   "Problem Solving",
    "agile":             "Agile",
    "scrum":             "Scrum",
    "jira":              "Jira",
}


# ============================================================================
# SKILL CATEGORIES (for ML features and UI grouping)
# ============================================================================

CATEGORIES = {
    "programming": {
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#",
        "Go", "Rust", "PHP", "Ruby", "Kotlin", "Swift", "Scala",
        "Perl", "MATLAB", "R", "Dart"
    },
    "frameworks": {
        "React", "Angular", "Vue.js", "Svelte", "Next.js", "Nuxt.js",
        "Node.js", "Express.js", "Django", "Flask", "FastAPI",
        "Spring Boot", "Spring", "Laravel", "Ruby on Rails",
        "ASP.NET", ".NET"
    },
    "frontend": {
        "HTML", "CSS", "SASS", "Tailwind CSS", "Bootstrap",
        "Material-UI", "Redux", "Vuex", "jQuery"
    },
    "databases": {
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite",
        "Oracle DB", "MS SQL", "Elasticsearch", "Cassandra",
        "DynamoDB", "Firebase", "Firestore", "Supabase", "SQL", "NoSQL"
    },
    "cloud_devops": {
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git",
        "GitHub", "GitLab", "Bitbucket", "Jenkins", "Terraform",
        "Ansible", "CI/CD", "Linux", "Bash", "Shell Scripting"
    },
    "data_ml": {
        "Machine Learning", "Deep Learning", "AI", "Data Science",
        "Data Analysis", "Data Analytics", "TensorFlow", "PyTorch",
        "Keras", "Pandas", "NumPy", "scikit-learn", "Matplotlib",
        "Seaborn", "Power BI", "Tableau", "Excel", "NLP",
        "Computer Vision", "OpenCV"
    },
    "generative_ai": {
        "Generative AI", "LLM", "LangChain", "OpenAI", "ChatGPT",
        "GPT", "Claude AI", "Gemini", "Hugging Face", "BERT",
        "Transformers", "Prompt Engineering", "RAG"
    },
    "mobile": {
        "Android", "iOS", "React Native", "Flutter", "Dart",
        "Xamarin", "Ionic"
    },
    "api_backend": {
        "REST API", "GraphQL", "Microservices", "WebSockets", "gRPC"
    },
    "design": {
        "Figma", "Adobe XD", "Photoshop", "Illustrator",
        "UI/UX", "UI Design", "UX Design"
    },
    "business": {
        "SEO", "SEM", "Google Analytics", "Google Ads", "Facebook Ads",
        "Salesforce", "CRM", "Content Writing", "Copywriting",
        "Digital Marketing", "Social Media Marketing", "Email Marketing",
        "Sales", "B2B Sales", "B2C Sales"
    },
    "soft_skills": {
        "Communication", "Teamwork", "Leadership", "Problem Solving",
        "Agile", "Scrum", "Jira"
    },
}


# ============================================================================
# PRE-COMPUTED SORTED ALIASES (longest first for accuracy)
# ============================================================================
# Sort by length so "machine learning" matches before "ml" alone.

_SORTED_SKILL_ALIASES = sorted(
    SKILL_DICT.items(),
    key=lambda x: len(x[0]),
    reverse=True,
)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def extract_skills(title: str = "", description: str = "") -> dict:
    """
    Extract structured skills from job title + description.

    Args:
        title:       Job title text
        description: Full job description text

    Returns:
        {
            "skills":           [...],
            "skill_count":      int,
            "skill_categories": {...}
        }
    """
    # Combine + clean text
    text = f"{title or ''} {description or ''}".lower()

    # Replace special chars with spaces, but preserve + # . for C++, C#, .NET
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)

    # Match skill aliases
    found = set()
    for alias, canonical in _SORTED_SKILL_ALIASES:
        # Word boundary prevents false positives (e.g. "r" inside "react")
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, text):
            found.add(canonical)

    skills_list = sorted(found)

    # Categorize
    categories = {}
    for skill in skills_list:
        for cat_name, cat_skills in CATEGORIES.items():
            if skill in cat_skills:
                categories.setdefault(cat_name, []).append(skill)

    return {
        "skills":           skills_list,
        "skill_count":      len(skills_list),
        "skill_categories": categories,
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_skill_category(skill: str) -> str:
    """Return the category of a single skill, or 'other' if unknown."""
    for cat_name, cat_skills in CATEGORIES.items():
        if skill in cat_skills:
            return cat_name
    return "other"


def detect_skill_title_mismatch(title: str, skills: list) -> bool:
    """
    Detect if the job title mentions a skill that is NOT in the skills list.
    Used as a fraud signal by scoring_engine.

    Example:
        title="Python Developer", skills=["Marketing", "Sales"] -> True
    """
    if not title or not skills:
        return False

    title_skills_data = extract_skills(title=title)
    title_skills = set(title_skills_data["skills"])
    actual_skills = set(skills)

    title_only = title_skills - actual_skills
    return len(title_only) > 0


# ============================================================================
# SELF-TEST
# ============================================================================

def _self_test():
    print("=" * 70)
    print("SKILL EXTRACTOR - SELF-TEST")
    print("=" * 70)

    test_cases = [
        {
            "name":        "Standard Python job",
            "title":       "Python Developer",
            "description": "We need Python, Django, PostgreSQL, Redis experience. AWS knowledge preferred.",
        },
        {
            "name":        "Frontend job with aliases",
            "title":       "React.js Engineer",
            "description": "Skills: JS, TS, ReactJS, Redux, Tailwind, HTML5, CSS3",
        },
        {
            "name":        "Modern AI job",
            "title":       "Generative AI Engineer",
            "description": "Work with LLMs, LangChain, OpenAI, RAG, Prompt Engineering",
        },
        {
            "name":        "Tricky: C++ and C#",
            "title":       "C++ Developer",
            "description": "Need C++, C#, .NET experience",
        },
        {
            "name":        "Multiple categories",
            "title":       "Full Stack Developer",
            "description": "MERN stack: MongoDB, Express, React, Node.js. Also need AWS, Docker, Git",
        },
        {
            "name":        "Marketing job (no programming)",
            "title":       "Digital Marketing Intern",
            "description": "SEO, SEM, Google Ads, content writing, social media marketing",
        },
        {
            "name":        "Empty input",
            "title":       "",
            "description": "",
        },
        {
            "name":        "Mobile development",
            "title":       "Flutter Developer",
            "description": "Build mobile apps using Flutter, Dart, Firebase. iOS/Android experience",
        },
    ]

    for tc in test_cases:
        print(f"\n{'-' * 70}")
        print(f"Test: {tc['name']}")
        print(f"   Title: {tc['title']!r}")
        print(f"   Desc:  {tc['description'][:60]!r}...")

        result = extract_skills(tc['title'], tc['description'])

        print(f"\n   Found {result['skill_count']} skills:")
        for skill in result['skills']:
            print(f"      - {skill}")

        if result['skill_categories']:
            print(f"\n   Categories:")
            for cat, skills in result['skill_categories'].items():
                print(f"      {cat:15s} -> {', '.join(skills)}")

    # Mismatch detection
    print(f"\n{'-' * 70}")
    print("Mismatch Detection Test:")

    mismatch = detect_skill_title_mismatch(
        title="Python Developer",
        skills=["Marketing", "Sales"],
    )
    print(f"   Python Dev with Marketing skills -> mismatch: {mismatch} (should be True)")

    match = detect_skill_title_mismatch(
        title="Python Developer",
        skills=["Python", "Django"],
    )
    print(f"   Python Dev with Python skills -> mismatch: {match} (should be False)")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()