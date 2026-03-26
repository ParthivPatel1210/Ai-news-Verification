# 🚀 Deployment Guide: Getting AI NewsVerifier Live

Your application has been perfectly configured to be deployed for **Free** on [Render.com](https://render.com). 

**Why Render over Vercel?**
Vercel has a strict 50MB serverless limit for Python apps. Because we just massively improved your AI model to 99.45% accuracy, your specific `vectorizer.joblib` AI core grew to **69MB** in size. Render is an incredibly powerful free platform that natively runs permanent Python web servers (Flask + Gunicorn) without size restrictions on ML models!

*(Note: I have already created a strict `.gitignore` and updated your `requirements.txt` with `gunicorn` so the environment is incredibly lightweight and production-ready.)*

---

## Step 1: Upload to GitHub
1. Open your terminal in this folder: `/Users/parthivpatel/Desktop/AI_news_verifitcation`
2. Initialize Git and commit your application:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - AI News Verifier Production Build"
   git branch -M main
   ```
3. Go to GitHub and create a completely blank repository. Then link it and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```
   *Your massive 115MB raw data files and local test databases have been automatically ignored so they don't break the upload!*

## Step 2: Deploy on Render
1. Go to [Render.com](https://render.com) and create a free account using your GitHub login.
2. Click **New +** in the top right and select **Web Service**.
3. Choose **"Build and deploy from a Git repository"** and select your newly uploaded repository.
4. Fill out the exact configuration below:
   - **Name**: `ai-news-verifier` (or anything you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Select the **Free** tier.

## Step 3: Launch!
Scroll to the bottom and click **Create Web Service**. 
Render will automatically download your GitHub code, install all packages from your `requirements.txt`, spin up your `gunicorn` production server, and grant you a live, shareable URL (`https://your-app-name.onrender.com`)!
