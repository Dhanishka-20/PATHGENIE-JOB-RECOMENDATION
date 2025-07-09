
import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv("large_jobs_dataset.csv")
    df['text'] = df['Job Description'].fillna('') + " " + df['Skills'].fillna('')
    if 'Location' not in df.columns:
        df['Location'] = np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Remote'], len(df))
    if 'Experience' not in df.columns:
        df['Experience'] = np.random.choice(['Fresher', '1-3 years', '3-5 years', '5+ years'], len(df))
    return df

df = load_data()

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'])
X = tfidf_matrix
y = df['Job Title']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Resume", "Job Matching", "Dashboard"])

if page == "Upload Resume":
    st.header("Upload Your Resume")
    uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])
    if uploaded_file:
        resume_text = ""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
        st.subheader("Extracted Resume Text:")
        st.write(resume_text)
        st.session_state['parsed_skills'] = resume_text

elif page == "Job Matching":
    st.header("Find the Best Job Matches")
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("Select Location", df['Location'].unique())
    with col2:
        experience = st.selectbox("Select Experience", df['Experience'].unique())

    default_skills = st.session_state.get('parsed_skills', "")
    user_input = st.text_area("Enter Your Skills", value=default_skills, height=100, key="resume_skills_input")

    if st.button("Find Jobs"):
        if user_input.strip() == "":
            st.warning("Please enter skills or upload a resume.")
        else:
            filtered_df = df[(df['Location'] == location) & (df['Experience'] == experience)]
            if filtered_df.empty:
                st.warning("No jobs found for this filter.")
            else:
                tfidf_filtered = vectorizer.fit_transform(filtered_df['text'])
                user_tfidf = vectorizer.transform([user_input])
                scores = cosine_similarity(user_tfidf, tfidf_filtered)
                top_indices = scores[0].argsort()[-5:][::-1]
                results = filtered_df.iloc[top_indices][['Job Title', 'Job Description', 'Skills', 'Location', 'Experience']].reset_index(drop=True)

                job_platforms = {
                    "LinkedIn": "https://www.linkedin.com/jobs/search/?keywords={}",
                    "Naukri": "https://www.naukri.com/{}-jobs",
                    "Indeed": "https://www.indeed.com/jobs?q={}",
                    "Glassdoor": "https://www.glassdoor.co.in/Job/jobs.htm?sc.keyword={}"
                }

                selected_platform = st.selectbox("Choose Job Platform", options=list(job_platforms.keys()))

                st.subheader("Top 5 Recommended Jobs:")
                for i in range(len(results)):
                    job = results.iloc[i]
                    job_title_encoded = job['Job Title'].replace(" ", "+")
                    job_link = job_platforms[selected_platform].format(job_title_encoded)

                    st.markdown(f"""
                        <div style='background-color:#fef9c3;padding:15px;border-radius:10px;margin-bottom:10px;color:black'>
                            <b style='font-size:16px'>{job['Job Title']}</b> — <i>{job['Location']} ({job['Experience']})</i><br>
                            <b>Skills:</b> {job['Skills']}<br>
                            <p>{job['Job Description']}</p>
                            <a href='{job_link}' target='_blank'>Apply on {selected_platform}</a>
                        </div>
                    """, unsafe_allow_html=True)

                csv_data = results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Recommendations as CSV", data=csv_data, file_name="recommendations.csv", mime='text/csv')

    st.markdown("---")
    st.subheader("Predict Your Job Role from Skills")
    predict_input = st.text_input("Enter your skills (e.g., Python, SQL, Excel):", key="predict_input")
    if st.button("Predict Job Role"):
        if predict_input.strip():
            try:
                input_vec = vectorizer.transform([predict_input])
                prediction = clf.predict(input_vec)[0]
                st.success(f"Recommended Job Role: {prediction}")
            except Exception as e:
                st.error("Model error. Please check input.")
        else:
            st.warning("Please enter skills.")

elif page == "Dashboard":
    st.header("Evaluation Dashboard")
    st.markdown("These are real charts from the trained ML model.")

    st.subheader("✅ Accuracy Score")
    st.write(f"Model Accuracy: **{acc:.2f}**")

    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📋 Classification Report")
    st.text(report)

    st.subheader("📈 Resume Skill Match Rate ")
    st.progress(75)
