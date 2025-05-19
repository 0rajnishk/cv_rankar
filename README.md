#### Project Overview

The project "CV Ranker" is designed to analyze resumes against job descriptions using Azure services and various Python libraries. It leverages Azure Blob Storage for file handling and Azure OpenAI for text analysis.

#### Features

* **Azure Blob Storage Integration**: Upload, download, list, and delete files using Azure Blob Storage.
* **Text Analysis**: Utilizes FAISS for similarity scoring and Azure OpenAI for generating JSON-based resume analysis reports.
* **User Interface**: Built with Streamlit to provide a user-friendly interface for uploading resumes and entering job descriptions.

#### Technologies Used

* **Python Libraries**: PyPDF2, docx, pandas, numpy, faiss, sentence-transformers, etc.
* **Azure Services**: Azure OpenAI for natural language processing, Azure Blob Storage for file management.

#### Setup Instructions

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Set up your `.env` file with necessary credentials and configuration (example provided).
4. Run `main.py` using Python to start the application.

#### Usage

1. Provide a job description in the text area.
2. Upload resumes in PDF or DOCX format.
3. Click "Analyze Resumes" to see the matching scores and analysis results.
4. Optionally, download the analysis results as a CSV file.

#### Deployment
