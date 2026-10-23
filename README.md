# CourseIQ — Learning Insights by Mining Minds

**A Data Warehouse and Data Mining Project using the Open University Learning Analytics Dataset (OULAD)**

CourseIQ is an academic project that applies data warehousing, OLAP operations, and data mining techniques to student learning analytics. It explores student engagement, academic performance, and learning patterns to support data-driven course recommendations.

## 📌 Project Overview

The project transforms educational data into meaningful insights through data preparation, data warehousing, analytical processing, and machine learning. It provides an interactive Streamlit application for exploring datasets, analyzing student behavior, and generating learning insights.

## 🎯 Objectives

- Build a data warehouse structure for educational analytics.
- Explore data using OLAP operations.
- Identify student learning patterns through clustering.
- Predict student performance using classification algorithms.
- Discover relationships among learning activities using association rule mining.
- Explore course recommendation possibilities based on student characteristics and learning patterns.

## 📊 Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**, which contains information about courses, students, assessments, and virtual learning environment (VLE) interactions.

The dataset supports the analysis of student demographics, assessment performance, course participation, and online learning activity.

### Main Dataset Components

| Dataset | Description |
|---|---|
| `studentInfo` | Student demographic and course information |
| `studentRegistration` | Student registration and withdrawal details |
| `studentAssessment` | Student assessment submissions and scores |
| `assessments` | Assessment dates, types, and weightings |
| `courses` | Course presentation information |
| `studentVle` | Student interactions with virtual learning materials |
| `vle` | Virtual learning environment resources |

*Note: The names above describe the main OULAD tables. Actual filenames and columns should match the dataset files used by the application.*

## 🏗️ Project Workflow

1. **Data Collection** — Load the OULAD dataset files.
2. **Data Preprocessing** — Inspect, clean, and prepare data for analysis.
3. **Data Warehousing** — Organize relevant data into analytical structures.
4. **OLAP Analysis** — Explore data using multidimensional analytical operations.
5. **Data Mining** — Apply clustering, classification, and association rule mining.
6. **Learning Insights** — Present analytical results through an interactive dashboard.

## 🧱 Data Warehouse

The data warehouse component organizes educational data for analytical queries.

Key concepts explored include:

- Fact and dimension tables
- Star-schema modeling
- Data integration and transformation
- Analytical queries for student and course performance

The warehouse structure and table relationships should be interpreted according to the project's implemented schema.

## 🔎 OLAP Operations

The application provides an interactive workspace for exploring Online Analytical Processing (OLAP) operations:

- **Roll-up:** Summarize data at a higher level of aggregation.
- **Drill-down:** Explore more detailed levels of data.
- **Slice:** Filter data using a selected dimension value.
- **Dice:** Filter data using multiple dimension conditions.
- **Pivot:** Rearrange dimensions to view data from different perspectives.

These operations help examine educational data at different levels of detail.

## 🤖 Data Mining Techniques

### 1. Clustering

Groups students with similar characteristics or learning patterns. The resulting clusters can help identify different engagement or performance profiles.

### 2. Classification

Uses labeled historical data to predict student outcome categories. The application explores classification models and evaluates their performance using suitable metrics.

### 3. Association Rule Mining

Identifies relationships and co-occurrence patterns among selected learning activities or attributes.

### 4. Course Recommendation

Explores how student characteristics and learning patterns can inform course recommendations. Recommendations should be interpreted as analytical suggestions rather than guaranteed outcomes.

## 🖥️ Application

The project uses **Streamlit** to provide an interactive web interface for data exploration and analysis.

The application includes the following workspaces, subject to the current implementation:

- Overview and dataset inspection
- Data Warehouse
- OLAP Operations
- Clustering
- Classification
- Association Rules
- Prediction

## 🛠️ Technologies Used

- **Python** — Data processing and analysis
- **Pandas** — Data manipulation
- **NumPy** — Numerical computing
- **Scikit-learn** — Machine learning
- **Streamlit** — Interactive web application
- **SQL** — Analytical query concepts, where implemented
- **Git and GitHub** — Version control and project collaboration

## 🚀 Running the Application Locally

### Prerequisites

- Python installed
- Git installed
- Project source code
- Required dataset files

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rithishk2506/oulad-insight-lab.git
   ```

2. Open the project directory:

   ```bash
   cd oulad-insight-lab
   ```

3. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```bash
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure any required secrets using the project's documented configuration. Never commit API tokens, passwords, or private credentials.

6. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## 📁 Repository Structure

```text
oulad-insight-lab/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── render.yaml                 # Deployment configuration
├── run_dashboard_windows.bat   # Windows dashboard launcher
├── README.md                   # Project documentation
├── .gitignore                  # Git exclusion rules
├── assessments.csv              # Assessment metadata
├── courses.csv                  # Course information
├── studentAssessment.csv        # Student assessment results
├── studentInfo.csv              # Student demographic and course data
├── studentRegistration.csv      # Registration and withdrawal data
├── studentVle.zip               # Compressed VLE interaction dataset
└── vle.csv                      # Virtual learning environment resources
```

**Note:** The `studentVle.zip` archive contains the large VLE interaction dataset. The extracted CSV is not shown in the repository listing. The application should extract or read this data according to its implemented loading logic.

## 📜 Dataset Attribution and License

This project uses the **Open University Learning Analytics Dataset (OULAD)**.

Please consult the original dataset source for the exact license name, version, attribution requirements, and applicable terms:

- [OULAD — Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)

The license information provided with the dataset permits sharing and adaptation under its stated conditions, including appropriate attribution, a link to the license, and an indication of changes where required.

This project processes and analyzes the dataset for educational analytics. Any preprocessing, transformations, or derived analyses performed by this project are separate from the original dataset.

This project is an independent academic work and is not endorsed by the dataset's creators.

## 🔐 Security and Data Handling

- Do not commit API tokens, access tokens, passwords, or Streamlit secrets.
- Keep sensitive configuration files out of version control.
- Review dataset redistribution terms before uploading dataset files.
- Avoid committing unnecessary large files or generated artifacts.

## 🎓 Academic Purpose

CourseIQ demonstrates the application of data warehousing and data mining concepts to educational analytics. It is intended for academic learning, experimentation, and exploratory analysis.

The insights generated by machine learning models depend on the available data, selected features, and model performance. They should not be treated as definitive judgments about individual students.

## 👥 Project Team

**Mining Minds**

Developed as part of an academic Data Warehouse and Data Mining project.

## 📄 License

The dataset license applies to the original OULAD material according to its terms. It does not automatically determine the license for all original code, documentation, or other project contributions.

If you want to license your own source code publicly, select an appropriate software license and ensure that all project contributors agree before adding it.

---

**CourseIQ — Turning Learning Data into Actionable Insights.**
