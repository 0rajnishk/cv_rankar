import google.generativeai as genai
import PyPDF2 as pdf
import docx
import json
import streamlit as st
import os
from dotenv import load_dotenv
import re
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
import tempfile
import sys
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Azure Blob Storage functions
# Initialize Azure Blob Storage client
def initialize_blob_client(connection_string):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        return blob_service_client
    except Exception as e:
        st.error(f"Failed to initialize Azure Blob Storage client: {str(e)}")
        return None

# Create unique folder name
def create_unique_folder_name():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"cv_analysis_{timestamp}_{unique_id}"

# Upload files to Azure Blob Storage
def upload_files_to_blob(container_name, files, folder_name, connection_string):
    blob_service_client = initialize_blob_client(connection_string)
    if not blob_service_client:
        return []
        
    try:
        # Create the container if it doesn't exist
        try:
            container_client = blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                container_client = blob_service_client.create_container(container_name)
        except:
            container_client = blob_service_client.create_container(container_name)
        
        uploaded_files = []
        
        for uploaded_file in files:
            destination_blob_name = f"{folder_name}/{uploaded_file.name}"
            blob_client = blob_service_client.get_blob_client(
                container=container_name, 
                blob=destination_blob_name
            )
            
            # Upload file
            file_contents = uploaded_file.getvalue()
            blob_client.upload_blob(file_contents, overwrite=True)
            uploaded_files.append(destination_blob_name)
            
        return uploaded_files
    except Exception as e:
        st.error(f"Error uploading files to Azure Blob Storage: {str(e)}")
        return []

# Download file from Azure Blob Storage
def download_file_from_blob(container_name, blob_name, connection_string):
    blob_service_client = initialize_blob_client(connection_string)
    if not blob_service_client:
        return None
        
    try:
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob_data = blob_client.download_blob()
            blob_data.readinto(temp_file)
            return temp_file.name
    except Exception as e:
        st.error(f"Error downloading file from Azure Blob Storage: {str(e)}")
        return None

# List files in an Azure Blob Storage folder
def list_blob_files(container_name, folder_name, connection_string):
    blob_service_client = initialize_blob_client(connection_string)
    if not blob_service_client:
        return []
        
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs(name_starts_with=folder_name)
        return [blob.name for blob in blob_list 
                if (blob.name.lower().endswith(('.pdf', '.docx')) and 
                    not blob.name.endswith('/') and 
                    blob.name != folder_name)]
    except Exception as e:
        st.error(f"Error listing files in Azure Blob Storage: {str(e)}")
        return []

# Delete a folder and its contents from Azure Blob Storage
def delete_blob_folder(container_name, folder_name, connection_string):
    blob_service_client = initialize_blob_client(connection_string)
    if not blob_service_client:
        return False
        
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blobs_list = container_client.list_blobs(name_starts_with=folder_name)
        
        for blob in blobs_list:
            blob_client = container_client.get_blob_client(blob.name)
            blob_client.delete_blob()
        return True
    except Exception as e:
        st.error(f"Error deleting Azure Blob Storage folder: {str(e)}")



# Determine file type and extract text from Azure Blob file
def extract_resume_text_from_blob(container_name, blob_name, connection_string):
    temp_file_path = download_file_from_blob(container_name, blob_name, connection_string)
    if not temp_file_path:
        return "Error: Could not download file from Azure Blob Storage"
    
    try:
        if blob_name.lower().endswith(".pdf"):
            with open(temp_file_path, "rb") as file:
                return extract_pdf_text(file)
        elif blob_name.lower().endswith(".docx"):
            return extract_docx_text(temp_file_path)
        return "Unsupported file format"
    except Exception as e:
        return f"Error processing file: {str(e)}"
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass  # Silently fail if cleanup doesn't work






# Configure Gemini API
def configure_genai(api_key):
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to configure Generative AI: {str(e)}")

# Generate text embeddings
def get_embedding(text):
    try:
        # Initialize the model outside of Streamlit's file watcher scope
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return np.array(model.encode(text))
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

# Get Gemini AI response with JSON parsing
def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        if not response or not response.text:
            raise Exception("Empty response received from Gemini")

        try:
            response_json = json.loads(response.text)
        except json.JSONDecodeError:
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response.text, re.DOTALL)
            if match:
                response_json = json.loads(match.group())
            else:
                raise Exception("Could not extract valid JSON response")

        # Ensure required fields exist
        required_fields = ["JD Match", "Skills", "MissingSkills", "YearsOfExperience", "Location", "Education", "CandidateName", "ProfileSummary"]
        for field in required_fields:
            response_json.setdefault(field, "Not Provided")

        # Fix JD Match value
        try:
            jd_match_str = str(response_json["JD Match"]).replace("%", "").strip()
            response_json["JD Match"] = float(jd_match_str) if jd_match_str.replace(".", "").isdigit() else 0.0
        except Exception:
            response_json["JD Match"] = 0.0  

        return response_json
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

# Extract text from PDF
def extract_pdf_text(file_content):
    try:
        reader = pdf.PdfReader(file_content)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text if text else "No text extracted"
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

# Extract text from DOCX
def extract_docx_text(file_content):
    try:
        doc = docx.Document(file_content)
        text = " ".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return text if text else "No text extracted"
    except Exception as e:
        return f"Error extracting DOCX text: {str(e)}"


# Prepare prompt for Gemini AI
def prepare_prompt(resume_text, job_description):
    if not resume_text or not job_description:
        return None

    prompt_template = f"""
        Act as an ATS (Applicant Tracking System) expert. Compare the following resume to the job description.

        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Provide a JSON response with:
        {{
            "JD Match": "percentage between 0-100",
            "Skills": ["Matched skills"],
            "MissingSkills": ["Missing skills"],
            "WorkExperienceFit": "Experience analysis",
            "Certifications": ["Relevant certifications"],
            "Qualifications": "Degree comparison",
            "OverallFit": "Suitability summary",
            "YearsOfExperience": "Total years of experience",
            "Location": "Candidate's location",
            "Education": "Highest qualification",
            "CandidateName": "Extracted name",
            "ProfileSummary": "Candidate's profile summary"
        }}
    """
    return prompt_template

# Create FAISS index for a single resume
def create_faiss_index(resume_embedding):
    try:
        dim = resume_embedding.shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(resume_embedding.reshape(1, -1).astype('float32'))
        return index
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return None

# Find similarity score using FAISS
def find_similar_resume(resume_embedding, faiss_index):
    try:
        if faiss_index is None:
            return 0
        distances, indices = faiss_index.search(resume_embedding.reshape(1, -1).astype('float32'), 1)
        return distances[0][0]
    except Exception as e:
        st.error(f"Error finding similarity: {str(e)}")
        return 0


# Streamlit App
def main():    
    load_dotenv()
# ############################################################################################################
    api_key = os.getenv("GOOGLE_API_KEY")
    # Get Azure OpenAI configuration
    azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not azure_openai_key or not azure_openai_endpoint or not azure_openai_deployment:
        st.error("Please set the Azure OpenAI credentials in your .env file")
        return
    
    # Get Azure Blob Storage connection string
    azure_blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    if not azure_blob_connection_string:
        st.error("Please set the AZURE_BLOB_CONNECTION_STRING in your .env file")
        return
    
    # Get Azure Blob Storage container name
    container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME", "cv-ranker-container")

# ############################################################################################################
    
    if not api_key:
        st.error("Please set the GOOGLE_API_KEY in your .env file")
        return
    
    # # Get GCS bucket name
    # bucket_name = os.getenv("GCS_BUCKET_NAME", "cv_ranker_temp_storage")

    configure_genai(api_key)


    with st.sidebar:
        st.title("🎯 CV Ranker")
        st.write("Analyze resumes against job descriptions.")

    st.title("📄 CV Ranker")
    st.subheader("Find the Best Job Match")

    jd = st.text_area("Job Description", placeholder="Paste job description here...")
    files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["pdf", "docx"])

    # Create session state for storing folder name
    if 'blob_folder_name' not in st.session_state:
        st.session_state.blob_folder_name = None

    if st.button("Analyze Resumes"):
        if not jd:
            st.warning("Please provide a job description.")
            return
        if not files:
            st.warning("Please upload at least one resume.")
            return

        try:
            with st.spinner("🔄 Uploading files to Azure Blob Storage..."):
                # Create a unique folder name for this session
                folder_name = create_unique_folder_name()
                st.session_state.blob_folder_name = folder_name
                
                # Upload files to Azure Blob Storage
                uploaded_files = upload_files_to_blob(container_name, files, folder_name, azure_blob_connection_string)
                
                if not uploaded_files:
                    st.error("Failed to upload files to Azure Blob Storage.")
                    return
                
                st.success(f"Uploaded {len(uploaded_files)} files to Azure Blob Storage.")

            with st.spinner("📊 Analyzing resumes..."):
                results = []
                failed_resumes = []

                # Pre-load the sentence transformer model once to avoid PyTorch issues
                transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
                jd_embedding = np.array(transformer_model.encode(jd))

                # List all resume files in the GCS folder
                # List all resume files in the Azure Blob Storage folder
                resume_files = list_blob_files(container_name, folder_name, azure_blob_connection_string)
                
                progress_bar = st.progress(0)
                for i, resume_blob_name in enumerate(resume_files):
                    resume_file_name = os.path.basename(resume_blob_name)
                    progress_bar.progress((i) / len(resume_files))
                    
                    # Extract text from the resume
                    resume_text = extract_resume_text_from_blob(container_name, resume_blob_name, azure_blob_connection_string)

                    if isinstance(resume_text, str) and ("Error" in resume_text or resume_text.strip() == "No text extracted"):
                        failed_resumes.append({"ResumeName": resume_file_name, "Error": resume_text})
                        continue

                    try:
                        # Use the pre-loaded model
                        resume_embedding = np.array(transformer_model.encode(resume_text))

                        # Generate FAISS index for each resume
                        faiss_index = create_faiss_index(resume_embedding)
                        similarity_score = find_similar_resume(jd_embedding, faiss_index)

                        input_prompt = prepare_prompt(resume_text, jd)
                        response_json = get_gemini_response(input_prompt)

                        results.append({
                            "ResumeName": resume_file_name,
                            "CandidateName": response_json["CandidateName"],
                            "JD Match": response_json["JD Match"],
                            "Skills": response_json["Skills"],
                            "MissingSkills": response_json["MissingSkills"],
                            "YearsOfExperience": response_json["YearsOfExperience"],
                            "Location": response_json["Location"],
                            "Education": response_json["Education"],
                            "ProfileSummary": response_json["ProfileSummary"],
                            "SimilarityScore": similarity_score
                        })
                    except Exception as e:
                        failed_resumes.append({"ResumeName": resume_file_name, "Error": str(e)})
                
                progress_bar.progress(1.0)

                if results:
                    df = pd.DataFrame(results).sort_values(by="JD Match", ascending=False)
                    st.dataframe(df)
                    st.download_button("Download CSV", df.to_csv(index=False), "resume_analysis.csv")
                else:
                    st.warning("No resumes were successfully analyzed.")

                if failed_resumes:
                    st.warning("Some resumes could not be processed.")
                    st.dataframe(pd.DataFrame(failed_resumes))

                # Cleanup Azure Blob Storage folder after analysis (optional)
                if st.checkbox("Delete uploaded files after analysis", value=False):
                    delete_blob_folder(container_name, folder_name, azure_blob_connection_string)
                    st.session_state.blob_folder_name = None
                    st.success("Temporary files deleted from Azure Blob Storage.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            # In case of error, try to clean up
            if st.session_state.blob_folder_name:
                try:
                    delete_blob_folder(container_name, st.session_state.blob_folder_name, azure_blob_connection_string)
                except Exception:
                    pass
                    
if __name__ == "__main__":
    main()


