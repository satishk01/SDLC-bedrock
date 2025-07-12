import streamlit as st
import boto3
import json
import pandas as pd
from typing import List, Dict
import fpdf
import zipfile
import io
import os
from datetime import datetime

class PDF(fpdf.FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'JIRA Stories Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def initialize_bedrock():
    """Initialize AWS Bedrock client"""
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        return bedrock
    except Exception as e:
        st.error(f"Error connecting to AWS Bedrock: {str(e)}")
        return None

def expand_requirement(basic_requirement: str, requirement_type: str, bedrock_client) -> str:
    """Expand a basic requirement into a detailed requirement using Claude"""
    
    prompt = f"""Expand the following basic {requirement_type} requirement into a detailed requirement.
    Include specific details about:
    - Functional aspects
    - Technical considerations
    - User interactions
    - Performance criteria
    - Security considerations (if applicable)
    - Integration points (if applicable)
    
    Basic Requirement: {basic_requirement}
    
    Provide a comprehensive, well-structured detailed requirement."""
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        detailed_requirement = response_body['content'][0]['text']
        return detailed_requirement
        
    except Exception as e:
        st.error(f"Error expanding requirement: {str(e)}")
        return None

def breakdown_requirement(detailed_requirement: str, requirement_type: str, bedrock_client) -> List[str]:
    """Break down a detailed requirement into individual task descriptions"""
    
    prompt = f"""Break down the following detailed {requirement_type} requirement into individual tasks.
    Each task should be specific, actionable, and independent.
    Format the response as a JSON array of task descriptions.
    
    Detailed Requirement: {detailed_requirement}
    
    Response format:
    ["task1 description", "task2 description", ...]"""
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        content = response_body['content'][0]['text']
        
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        if start_idx != -1 and end_idx != -1:
            tasks = json.loads(content[start_idx:end_idx])
            return tasks
        else:
            st.error("Could not find valid task list in Claude's response")
            return None
            
    except Exception as e:
        st.error(f"Error breaking down requirement: {str(e)}")
        return None

def create_jira_story(requirement: str, requirement_type: str, bedrock_client) -> Dict:
    """Convert requirement into a JIRA story using Claude"""
    
    prompt = f"""Convert the following {requirement_type} requirement into a JIRA story. 
    Include the following fields:
    - Summary
    - Description
    - AcceptanceCriteria
    - StoryPoints (using Fibonacci sequence: 1,2,3,5,8,13)
    - Priority (High/Medium/Low)
    - Labels
    
    Requirement: {requirement}
    
    Format the response as a JSON object with the above fields as keys.
    The response should be a valid JSON object like this:
    {{
        "Summary": "...",
        "Description": "...",
        "AcceptanceCriteria": "...",
        "StoryPoints": 3,
        "Priority": "High",
        "Labels": ["label1", "label2"]
    }}
    """
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        content = response_body['content'][0]['text']
        
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx]
            story = json.loads(json_str)
            return story
        else:
            st.error("Could not find valid JSON in Claude's response")
            return None
            
    except Exception as e:
        st.error(f"Error creating JIRA story: {str(e)}")
        return None

def create_pdf_report(stories: List[Dict], detailed_requirement: str) -> bytes:
    """Create a PDF report of the JIRA stories"""
    pdf = PDF()
    pdf.add_page()
    
    # Add detailed requirement
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Detailed Requirement:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, detailed_requirement)
    pdf.ln(10)
    
    # Add stories
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'JIRA Stories:', 0, 1)
    
    for i, story in enumerate(stories, 1):
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 10, f"Story {i}: {story['Summary']}", 0, 1)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 5, 'Description:', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, story['Description'])
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 5, 'Acceptance Criteria:', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, story['AcceptanceCriteria'])
        
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 5, f"Story Points: {story['StoryPoints']}", 0, 1)
        pdf.cell(0, 5, f"Priority: {story['Priority']}", 0, 1)
        pdf.cell(0, 5, f"Labels: {', '.join(story['Labels'])}", 0, 1)
        pdf.ln(5)
    
    # Get PDF as bytes directly instead of encoding
    return bytes(pdf.output(dest='S'))

def create_zip_archive(stories: List[Dict], detailed_requirement: str, csv_data: bytes, pdf_data: bytes) -> bytes:
    """Create a ZIP archive containing all export formats"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add CSV
        zip_file.writestr('jira_stories.csv', csv_data)
        
        # Add PDF
        zip_file.writestr('jira_stories.pdf', pdf_data)
        
        # Add JSON
        json_data = json.dumps({
            'detailed_requirement': detailed_requirement,
            'stories': stories
        }, indent=2)
        zip_file.writestr('jira_stories.json', json_data)
        
        # Add README
        readme_content = """JIRA Stories Export
Generated on: {}

This archive contains:
1. jira_stories.csv - CSV format of all stories
2. jira_stories.pdf - PDF report with detailed formatting
3. jira_stories.json - JSON format of all data
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        zip_file.writestr('README.txt', readme_content)
    
    return zip_buffer.getvalue()

def download_stories_csv(stories: List[Dict]) -> bytes:
    """Convert stories to CSV format for download"""
    df = pd.DataFrame(stories)
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.title("Requirements to JIRA Stories Converter")
    
    # Initialize AWS Bedrock
    bedrock_client = initialize_bedrock()
    if not bedrock_client:
        st.stop()
    
    # Session state initialization
    if 'stories' not in st.session_state:
        st.session_state.stories = []
    if 'detailed_requirement' not in st.session_state:
        st.session_state.detailed_requirement = None
    
    # Requirement input section
    st.header("Input Basic Requirement")
    requirement_type = st.selectbox(
        "Requirement Type",
        ["Business", "Technical", "Product", "Quality"]
    )
    
    basic_requirement = st.text_area(
        "Enter Basic Requirement",
        height=100,
        placeholder="Enter your basic requirement here..."
    )
    
    # Expand requirement
    if st.button("Expand Requirement"):
        if basic_requirement:
            with st.spinner("Expanding requirement..."):
                detailed_requirement = expand_requirement(basic_requirement, requirement_type, bedrock_client)
                if detailed_requirement:
                    st.session_state.detailed_requirement = detailed_requirement
                    st.success("Requirement expanded successfully!")
    
    # Display detailed requirement and create tasks
    if st.session_state.detailed_requirement:
        st.header("Detailed Requirement")
        st.write(st.session_state.detailed_requirement)
        
        if st.button("Break Down into Tasks"):
            with st.spinner("Breaking down requirement into tasks..."):
                tasks = breakdown_requirement(st.session_state.detailed_requirement, requirement_type, bedrock_client)
                if tasks:
                    st.success(f"Requirement broken down into {len(tasks)} tasks!")
                    
                    # Create JIRA story for each task
                    for task in tasks:
                        story = create_jira_story(task, requirement_type, bedrock_client)
                        if story:
                            st.session_state.stories.append(story)
    
    # Display stories and download options
    if st.session_state.stories:
        st.header("Generated JIRA Stories")
        for i, story in enumerate(st.session_state.stories, 1):
            with st.expander(f"Story {i}: {story['Summary']}", expanded=True):
                st.write("**Description:**")
                st.write(story['Description'])
                st.write("**Acceptance Criteria:**")
                st.write(story['AcceptanceCriteria'])
                st.write(f"**Story Points:** {story['StoryPoints']}")
                st.write(f"**Priority:** {story['Priority']}")
                st.write("**Labels:**", ", ".join(story['Labels']))
        
        # Download section
        st.header("Download Options")
        col1, col2, col3 = st.columns(3)
        
        # CSV download
        csv_data = download_stories_csv(st.session_state.stories)
        with col1:
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="jira_stories.csv",
                mime="text/csv"
            )
        
        # PDF download
        pdf_data = create_pdf_report(st.session_state.stories, st.session_state.detailed_requirement)
        with col2:
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name="jira_stories.pdf",
                mime="application/pdf"
            )
        
        # ZIP download
        zip_data = create_zip_archive(
            st.session_state.stories,
            st.session_state.detailed_requirement,
            csv_data,
            pdf_data
        )
        with col3:
            st.download_button(
                label="Download ZIP",
                data=zip_data,
                file_name="jira_stories.zip",
                mime="application/zip"
            )
        
        # Clear button
        if st.button("Clear All"):
            st.session_state.stories = []
            st.session_state.detailed_requirement = None
            st.rerun()

if __name__ == "__main__":
    main()

