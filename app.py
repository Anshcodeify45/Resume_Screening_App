import streamlit as st
import pickle
import re
import fitz  # PyMuPDF

clf = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Dictionary
categories_with_ids = {
    6: 'Data Science',
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and fitness',
    5: 'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain',
    23: 'Testing'
}

def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+', ' ', resumeText)
    resumeText = re.sub(r'@\S+', ' ', resumeText)
    resumeText = re.sub(r'RT|cc+', ' ', resumeText)
    resumeText = re.sub(r'#\S+', ' ', resumeText)
    resumeText = re.sub(r'[^\w\s]', ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText)
    return resumeText

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader("Upload Resume", type=["txt", "pdf"])

    if upload_file is not None:
        file_name = upload_file.name

        if file_name.endswith(".txt"):
            resumeText = upload_file.read().decode("utf-8", errors="ignore")
        elif file_name.endswith(".pdf"):
            resumeText = extract_text_from_pdf(upload_file)
        else:
            st.error("Unsupported file format.")
            return

        if resumeText.strip() == "":
            st.error("No text found in the resume.")
            return

        cleaned_resume = cleanResume(resumeText)
        transformed_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(transformed_resume)[0]
        
        st.write(f"Prediction ID: {prediction_id}")
        
        category = categories_with_ids.get(prediction_id, "Unknown Category")
        st.success(f"Predicted Job Category: {category}")

if __name__ == '__main__':
    main()
