import os
import json
import re
import requests
from openai import AzureOpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure Azure OpenAI
def configure_azure_openai(api_key, endpoint, deployment_name):
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint=endpoint
        )
        return client
    except Exception as e:
        raise Exception(f"Failed to configure Azure OpenAI: {str(e)}")

# Generate text embeddings
def get_embedding(text):
    try:
        # Initialize the model outside of Streamlit's file watcher scope
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return np.array(model.encode(text))
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

# Get Azure OpenAI response with JSON parsing
def get_azure_openai_response(prompt, client, deployment_name):
    try:
        # Call Azure OpenAI API
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an ATS (Applicant Tracking System) expert. Analyze resumes and return results in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        if not response or not response.choices or not response.choices[0].message.content:
            raise Exception("Empty response received from Azure OpenAI")

        response_text = response.choices[0].message.content
        
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)
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
        raise Exception(f"Error generating response from Azure OpenAI: {str(e)}")

# Prepare prompt for Azure OpenAI
def prepare_prompt(resume_text, job_description):
    if not resume_text or not job_description:
        return None

    prompt_template = f"""
        Compare the following resume to the job description.

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