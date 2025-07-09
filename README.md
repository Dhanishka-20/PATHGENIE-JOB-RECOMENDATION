# PATHGENIE-JOB-RECOMENDATION


---

### `README.md`

```markdown
# Job Recommender System using Streamlit

This is a Streamlit-based Job Recommender System that allows users to upload resumes, get job recommendations, and predict suitable job roles based on skillsets. It also includes a dashboard for model evaluation.

## Features

- Upload and parse PDF resumes
- Recommend top 5 matching jobs based on skills, location, and experience
- Predict job role from entered skills using a machine learning model
- Visual dashboard with accuracy, confusion matrix, and classification report
- Option to download job recommendations as CSV

## Technologies Used

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- PyPDF2

## Files Included

- `app.py` - Main Streamlit application file
- `requirements.txt` - List of dependencies for deployment
- `large_jobs_dataset.csv` - Dataset used for training and matching

## Dataset

The `large_jobs_dataset.csv` contains:
- Job Title
- Job Description
- Skills
- Location (randomly assigned if missing)
- Experience (randomly assigned if missing)

## Installation

1. Clone the repository:
```

git clone [https://github.com/yourusername/job-recommender.git](https://github.com/yourusername/job-recommender.git)
cd job-recommender

```

2. Install dependencies:
```

pip install -r requirements.txt

```

3. Run the app locally:
```

streamlit run app.py

```

## Deployment on Render

1. Create a GitHub repository and push all files.
2. Go to https://render.com and create a new Web Service.
3. Connect your GitHub repo.
4. Use the following settings:

- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run app.py --server.port=10000`
- Environment Variable: `PORT = 10000`
- Runtime: Python 3
- Instance Type: Free

5. Deploy and access your app via the public URL provided by Render.

## License

This project is for educational purposes only.
```

---
