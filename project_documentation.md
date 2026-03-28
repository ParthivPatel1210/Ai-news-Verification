# AI News Verification Platform - Complete Project Documentation (In-Depth Guide)

This document is designed to give you a complete, top-to-bottom understanding of the **AI News Verification Platform**. It not only lists the technologies used but explains **what they are, why they are used, and exactly how they work together** to power the entire application.

---

## 1. What is the Objective and Motivation?
**The Problem:** Fake news and misinformation spread wildly on the internet. Traditional fact-checking agencies involve humans reading and researching claims, which takes hours or days.
**The Solution (Our Project):** We built an automated web platform that can read a news article or web URL and instantly predict if it is **REAL or FAKE**.
**The "Secret Sauce":** A simple Machine Learning (AI) model cannot know about events that happened *after* it was trained. So, we built a **Hybrid System**. Our app uses AI to spot "liar's language" (sensationalism) AND instantly cross-references the claim by automatically searching the live internet to see if highly trusted journalists (like BBC or Reuters) are reporting it right now.

---

## 2. High-Level Architecture: How It All Connects

When you type a news snippet into our app and click "Verify," the following magic happens in less than 3 seconds:
1.  **Frontend (The Face):** The website you interact with takes your text and sends it as a message to the Backend.
2.  **Backend (The Brain - Flask):** Receives the message. If you gave a URL, a bot goes to that URL and reads the text (Web Scraping).
3.  **Machine Learning (The AI):** The Backend asks the AI model, "Does this sound like fake news?" The AI converts the words to math, calculates a probability, and highlights the suspicious words.
4.  **Live Web API (The Researcher):** The Backend instantly Googles/DuckDuckGo's the claim to see who is talking about it.
5.  **Database (The Memory):** The final conclusion is saved permanently so you can see it in your history/dashboard later.
6.  **Frontend (The Result):** The Backend sends a polished report back to your browser, showing you the result visually.

Let's break down each of these components in detail.

---

## 3. The Frontend (The Face of the App)
**What is a Frontend?** The Frontend is everything the user sees and interacts with on a website. It runs directly inside the user's web browser (like Chrome or Safari).
*   **HTML:** Gives the page structure (buttons, text boxes, navigation bars).
*   **CSS:** Makes it look beautiful (colors, layout, modern glassmorphism design, dark mode).
*   **JavaScript (JS):** Makes the page interactive. For example, when you click an "Upvote" button on the Community Feed, JS updates the number instantly without refreshing the whole page.
*   **Jinja2 (Templates):** Because manually coding 10 different pages is tedious, we use "templates". Jinja2 allows our Python backend to inject dynamic data (like your username or your historical predictions) directly into the HTML before sending it to the user.

---

## 4. The Backend & Flask (The Engine Room)
**What is a Backend?** The Backend runs on a hidden server computer. It processes data, talks to databases, runs security checks, and handles the heavy lifting.
**What is an API?** An API (Application Programming Interface) is a messenger that takes requests and tells a system what you want to do, and then returns the response back to you. Think of an API like a waiter in a restaurant: you give the waiter your order, they take it to the kitchen (backend engine), and they bring your food (data) back to your table.
**What is Flask?** Flask is a popular, lightweight web framework written in **Python**. It allows us to build Backend servers easily. We use Python because it has the best libraries for Machine Learning and AI.

### How Our Main API Endpoints Work (`app.py`):
*   **`/predict` API:** This is the core engine. When the frontend sends a text or URL to this API, it runs the entire AI and Web Verification pipeline and returns a JSON (formatted text) response containing the REAL/FAKE probability, explainable highlights, and web sources.
*   **`/api/news`:** This endpoint acts as a middleman. It automatically fetches live RSS feeds from Google News about "Artificial Intelligence", strips out the ugly XML code, and serves a clean list of the top 10 articles to our frontend `api.html` page.

---

## 5. Machine Learning & Algorithms (The AI Engine)
**What is Machine Learning (ML)?** Instead of programming hardcoded rules (e.g., `if word == "alien" then FAKE`), we give a computer thousands of examples of REAL news and FAKE news (our Dataset). The computer "learns" the patterns on its own.

### The Algorithm: Logistic Regression
We use **Logistic Regression**. While it sounds fancy, it's essentially a statistical mathematical equation.
1.  During training (`train.py`), the algorithm learns to assign a "weight" to every word in the English language. 
2.  Words found often in reliable journalism (e.g., "spokesperson", "stated", "according to") might get a negative weight (pushing toward REAL).
3.  Sensationalist, emotionally charged, or poorly capitalized words ("SHOCKING", "BOMBSHELL", "destroying") get highly positive weights (pushing toward FAKE).
4.  When new text arrives, it calculates the sum of all these weights and passes it through a mathematical curve (the Sigmoid function) to give us a probability between 0% and 100%.

### How Text Turns to Math: TF-IDF Vectorization
Computers cannot read English; they only understand numbers. To solve this, we use a **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency).
*   **TF (Term Frequency):** Counts how often a word appears in the article.
*   **IDF (Inverse Document Frequency):** Penalizes highly common words (like "the", "and") because they aren't useful for classification, while boosting rare, highly specific words that strongly indicate fake or real content.
*   This transforms a paragraph into an array of thousands of numbers, which is then fed into the Logistic Regression equation.

### Explainable AI (XAI)
A major feature of our project is that it doesn't just output a blind answer. When it predicts FAKE, it looks backward at the math. It calculates which specific words in the user's text triggered the massive positive weights in the Logistic Regression formula. It then highlights those words to the user, explaining *why* the AI flagged it.

---

## 6. Live Web Verification (The Hybrid System)
What if the ML model is confused, or someone lies in a very professional tone? This is where our heuristic logic kicks in:
1.  **Web Scraping (`BeautifulSoup4`):** If a user provides a link instead of text, Python literally downloads the webpage's background code and extracts only the text paragraphs, ignoring ads and menus.
2.  **Credibility Heuristics:** We check the domain (e.g., `bbc.com`). If it matches our hardcoded dictionary of trusted news, we instantly give it a high credibility score. If it matches known satire sites (like `theonion.com`), it gets a low score.
3.  **Live Cross-Referencing (`DuckDuckGo API`):** If the text is raw and unlinked, the Backend grabs the first 20 words (assuming it's the core claim) and literally performs a live DuckDuckGo web search in the background. If 3 highly reputable sites are currently reporting on that exact string of words, the backend overrides the AI's skepticism and boosts the probability to REAL. If no one on the internet is talking about it, it boosts FAKE.

---

## 7. Database (The Memory)
**What is a Database?** A structured system to save long-term information. We use **SQLite**, which is a lightweight database that saves all data into a single local file (`users.db`).
**What is SQLAlchemy?** Writing raw SQL code (like `SELECT * FROM users`) can be dangerous and hard to read. We use SQLAlchemy, an Object-Relational Mapper (ORM). It lets us represent database tables as standard Python classes (see `models.py`), making it extremely simple to create users, save prediction logs, and securely hash passwords to protect user identities.

---

## 8. Deployment (Putting it on the Internet)
Finally, for an interview, you must explain how the app moved from your personal computer to be accessible by anyone globally.

### GitHub (Version Control)
*   **What is it?** A cloud platform to save and track changes to your code over time.
*   **`.gitignore` File:** Our dataset (`combined.csv`) is 115 Megabytes. Uploading massive datasets to GitHub is bad practice and causes errors. We created a `.gitignore` file to tell Git to ignore the massive datasets and local databases, only uploading the core `.py` code and the `.joblib` AI weight files.

### Render Cloud Hosting
*   **What is Render?** Render is a platform-as-a-service (PaaS). It gives us a virtual server computer on the internet that runs 24/7.
*   **Why Render over Vercel?** Many people use Vercel for free hosting, but Vercel has a strict 50MB file size limit for backend Python projects. Because our highly-accurate AI (`vectorizer.joblib`) alone is 69MB, Vercel would crash. Render has a slightly different architecture that natively supports permanent Python web servers without these restrictive size limits.
*   **What is Gunicorn?** When you run a Flask app locally on your laptop, it handles one user at a time. In the real world, 100 people might click "Verify" at the same exact millisecond. **Gunicorn** is an industrial-strength web server that acts as a traffic cop. We tell Render to run `gunicorn app:app`, which spins up multiple "worker" processes of our app simultaneously to handle heavy internet traffic smoothly.

---

## Summary for your Interview:
If asked "Walk me through this project", you can say:
*"I built a full-stack web platform using Flask and Python. It tackles the spread of misinformation by using a Hybrid AI model. First, we use NLP (TF-IDF Vectorization) and a Logistic Regression algorithm to analyze the linguistic patterns of a claim. Second, because AI can't know breaking news, my backend scrapes the live web and searches DuckDuckGo to cross-reference the claim with known trusted journalistic domains. The entire backend logic is wrapped in a beautiful, responsive user interface with user accounts, community tracking, and secure SQLite database storage, and is deployed globally using Gunicorn on the Render cloud platform."*
