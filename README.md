# 🔍 Powerful Resume Screening App using Python & NLP

## 📌 Overview

The **Resume Screening App** is a powerful, intelligent, and interactive web application that classifies uploaded resumes into predefined job categories using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. Built using **Python**, **Streamlit**, and deployed on **Render**, this project is aimed at automating and optimizing the resume shortlisting process for recruiters and HR professionals.

---

## 🚀 Features

* 📂 Upload resumes in **PDF** or **TXT** format.
* 🩹 Preprocesses resume text using custom **text cleaning functions**.
* 🔍 Transforms text data using **TF-IDF vectorization**.
* 🧠 Predicts the job category using a pre-trained **Multi-Class Classification Model**.
* 📊 Provides real-time feedback with **job category prediction**.
* 🌐 Fully deployed on the cloud using **Render** for remote accessibility.

---

## 🧠 Tech Stack

* **Frontend**: Streamlit (for building a fast and beautiful UI)
* **Backend / Logic**: Python
* **NLP Tools**: NLTK, Regular Expressions, TF-IDF (from Scikit-learn)
* **ML Model**: Trained using Scikit-learn’s Logistic Regression / Random Forest / SVM
* **Deployment**: Render (cloud-based deployment platform)
* **Model Artifacts**: `model.pkl`, `tfidf.pkl`

---

## 📂 Job Categories Covered

The model can classify resumes into 25+ job categories such as:

* Data Science
* Web Designing
* Business Analyst
* Java Developer
* Python Developer
* Civil/Mechanical/Electrical Engineering
* DevOps Engineer
* Blockchain, Hadoop, Testing, and more

---

## ✅ Highlights

* ✔️ Used advanced **feature engineering**, **class balancing**, and **GridSearchCV** for model optimization.
* ✔️ Created using modular, clean, and production-ready Python code.
* ✔️ **requirements.txt** for dependency management.
* ✔️ Deployed as a full web app — **ready for real-world usage**.

---

## 🌐 Live Demo

> 💡 Coming Soon / https://resume-screening-app-ttvu.onrender.com

---

## 📁 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/YourUsername/Resume_Screening_App.git
cd Resume_Screening_App

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 🤖 Future Improvements

* Resume parsing with structured output (Name, Email, Skills, etc.)
* Feedback loop for continuous model improvement
* Admin dashboard for batch resume uploads

---

## 🙌 Made With

> ❤️ by \Anish Patnaik — Data Science & AI Enthusiast
