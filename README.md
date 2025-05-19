#### Project Overview

The project **"CV Ranker"** is designed to analyze resumes against job descriptions using Azure services and various Python libraries. It leverages **Azure Blob Storage** for file handling and **Azure OpenAI** for text analysis. The application is **deployed on Azure App Services**, making it accessible as a web-based solution.

### Demo Video

[![Watch the demo](https://img.youtube.com/vi/wX5xuIKrJ4Q/0.jpg)](https://youtu.be/wX5xuIKrJ4Q)  
**Click the image above to watch the full demo on YouTube**

You can also download or view the video directly from the repository:  
[cv_rankar_demo.mp4](./cv_rankar_demo.mp4)


#### Features

* **Azure Blob Storage Integration**: Upload, download, list, and delete files using Azure Blob Storage.
* **Text Analysis**: Utilizes FAISS for similarity scoring and Azure OpenAI for generating JSON-based resume analysis reports.
* **User Interface**: Built with Streamlit to provide a user-friendly interface for uploading resumes and entering job descriptions.
* **Web Deployment**: Hosted on Azure App Services for scalability, availability, and easy access via a browser.

#### Technologies Used

* **Python Libraries**: PyPDF2, docx, pandas, numpy, faiss, sentence-transformers, etc.
* **Azure Services**:

  * **Azure OpenAI** for natural language processing
  * **Azure Blob Storage** for file management
  * **Azure App Services** for hosting and deploying the web application

#### Setup Instructions (for local development)

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Set up your `.env` file with necessary credentials and configuration (example provided).
4. Run `main.py` using Python to start the application locally.

#### Usage

1. Provide a job description in the text area.
2. Upload resumes in PDF or DOCX format.
3. Click **"Analyze Resumes"** to see the matching scores and analysis results.
4. Optionally, download the analysis results as a CSV file.

#### Deployment on Azure App Services

The application is deployed using **Azure App Services**, which handles hosting, scaling, and service availability. Key steps involved in deployment:

1. Create an App Service and configure the runtime (Python).
2. Set up deployment from GitHub or upload code via ZIP.
3. Add necessary environment variables in the App Service configuration (Application Settings).
4. Bind the App Service to Azure Blob Storage and Azure OpenAI credentials.
5. Streamlit is configured to run using `startup command` or `startup.txt` in the App Service settings (e.g., `streamlit run main.py --server.port 8000 --server.enableCORS false`).
