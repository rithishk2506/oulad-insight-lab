# OULAD Insight Lab — Local + Online Deployment

Dark-navy interactive dashboard for the OULAD Data Mining / Data Warehousing project.

## Included analytics
- **Clustering:** K-means, Hierarchical, DBSCAN; compare algorithms or inspect one; configurable K, features, DBSCAN parameters, PCA charts, silhouette score and CSV export.
- **Classification:** Dummy baseline, Decision Tree, Random Forest, Extra Trees, Logistic Regression; Accuracy, Macro-F1, confusion matrix and feature importance.
- **Association rules:** configurable minimum support, confidence and lift; downloadable rule table.
- **Prediction:** mean baseline, Linear Regression, Ridge, Decision Tree, Random Forest and Extra Trees; MAE, RMSE, R² and actual-vs-predicted chart.

## A. Run locally on Windows

Expected dataset directory:
`D:/Documents/Datawarehouse Project/Dataset`

Required files:
- `studentInfo.csv`
- `studentVle.csv`
- `vle.csv`
- `assessments.csv`
- `studentAssessment.csv`

Extract the project ZIP, open Command Prompt in `OULAD_Insight_Lab`, and run:

```bat
py -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints, usually `http://localhost:8501`. You can also run `run_dashboard_windows.bat`.

If your CSVs are in a different location, update **Dataset folder** in the sidebar. You may also set `OULAD_DATA_DIR` to the directory containing the CSVs.

## B. Deploy online for faculty

### Recommended: Streamlit Community Cloud
1. Create/sign in to a GitHub account and create a **private repository** for the project (or a public repo if your course permits).
2. Upload `app.py`, `requirements.txt`, and the project files to the repository root. Do not upload private credentials or unrelated files.
3. Make the dataset available to the hosted app. A local `D:/...` folder on your laptop is **not accessible to a cloud server**. Do not commit the complete OULAD VLE log to a public repository. Use an institution-approved private data store or a permitted private repository/storage option and configure the app's data directory accordingly. Confirm the dataset's license and your institution's data-handling rules first.
4. In Streamlit Community Cloud, choose **Deploy an app**, select the repository, branch, and `app.py`.
5. After the app starts, verify all five required CSVs are found and test each module. Share the generated app URL with faculty only after checking access controls and data policy.

### Alternative: Render
A `render.yaml` is included as a starting configuration. Connect the GitHub repository to Render and deploy the web service. Configure the dataset through an approved persistent/private storage solution; the repository's filesystem may be ephemeral and the full dataset may exceed free-tier resource limits.

## Data size and performance
- The full `studentVle.csv` can contain millions of rows. The first aggregation may be slow and require substantial memory.
- Clustering samples records by default to improve responsiveness; choose **All** only if the host has enough memory.
- Association rule mining may be memory-intensive on the full dataset. If it runs out of memory, use a higher support threshold or replace the simple rule generator with an optimized Apriori/FP-Growth implementation.
- The online app must be given data explicitly. The local folder path is only for your own computer.
- Metrics are computed at runtime and may differ from previous experiments because preprocessing, split strategy and model settings can differ.
- Prediction estimates assessment submission scores; `final_result` is not used as a prediction feature.

## Security checklist before sharing
- Do not place student records in a public GitHub repository.
- Use only approved/authorized data and follow your university's data policy.
- Keep any storage credentials in hosting secrets/environment variables, never in source code.
- If the faculty link should not be public, use a host/access configuration that supports authentication.
