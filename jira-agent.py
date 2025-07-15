import streamlit as st
import boto3
import json
import pandas as pd
from typing import List, Dict, Optional
import fpdf
import zipfile
import io
import os
from datetime import datetime
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from logger import log_llm_request, log_llm_response, log_llm_error, log_workflow_step, log_info, log_error

# Global configuration
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

class PDF(fpdf.FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'JIRA Stories Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

class AgentState:
    """Manages the state of the agentic workflow"""
    def __init__(self):
        self.current_step = "idle"
        self.progress = 0
        self.messages = []
        self.error = None
        self.completed_steps = []
        
    def update_step(self, step: str, progress: int = None, message: str = None):
        self.current_step = step
        if progress is not None:
            self.progress = progress
        if message:
            self.messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if step not in self.completed_steps:
            self.completed_steps.append(step)
    
    def add_message(self, message: str):
        self.messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def set_error(self, error: str):
        self.error = error
        self.current_step = "error"

def initialize_bedrock():
    """Initialize AWS Bedrock client"""
    try:
        log_info("Initializing AWS Bedrock client", {"region": "us-east-1"})
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        log_info("AWS Bedrock client initialized successfully")
        return bedrock
    except Exception as e:
        error_msg = f"Error connecting to AWS Bedrock: {str(e)}"
        log_error(error_msg, e)
        st.error(error_msg)
        return None

def safe_bedrock_call(bedrock_client, prompt: str, max_retries: int = 3, request_type: str = "general") -> Optional[str]:
    """Make a safe Bedrock API call with retries and error handling"""
    
    # Log the request
    log_llm_request(prompt, request_type, MODEL_ID)
    
    start_time = time.time()
    
    for attempt in range(max_retries):
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
                modelId=MODEL_ID,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            response_text = response_body['content'][0]['text']
            
            # Log successful response
            processing_time = time.time() - start_time
            log_llm_response(response_text, request_type, MODEL_ID, processing_time, 
                           {"attempt_number": attempt + 1, "max_retries": max_retries})
            
            return response_text
            
        except Exception as e:
            # Log the error
            log_llm_error(str(e), request_type, MODEL_ID, 
                         {"max_retries": max_retries, "attempt": attempt + 1})
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise e

def expand_requirement_agent(basic_requirement: str, requirement_type: str, bedrock_client, agent_state: AgentState) -> Optional[str]:
    """Expand a basic requirement into a detailed requirement using Claude"""
    
    agent_state.update_step("expanding", 10, "Starting requirement expansion...")
    log_workflow_step("expand_requirement", "started", {"requirement_type": requirement_type})
    
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
        agent_state.add_message("Calling AWS Bedrock API...")
        detailed_requirement = safe_bedrock_call(bedrock_client, prompt, request_type="requirement_expansion")
        
        if detailed_requirement:
            agent_state.update_step("expanded", 30, f"✅ Requirement expanded successfully! ({len(detailed_requirement)} characters)")
            log_workflow_step("expand_requirement", "completed", 
                            {"requirement_type": requirement_type, "output_length": len(detailed_requirement)})
            return detailed_requirement
        else:
            agent_state.set_error("Failed to expand requirement - empty response")
            log_workflow_step("expand_requirement", "failed", {"error": "empty response"})
            return None
            
    except Exception as e:
        agent_state.set_error(f"Error expanding requirement: {str(e)}")
        log_workflow_step("expand_requirement", "error", {"error": str(e)})
        return None

def breakdown_requirement_agent(detailed_requirement: str, requirement_type: str, bedrock_client, agent_state: AgentState) -> Optional[List[str]]:
    """Break down a detailed requirement into individual task descriptions"""
    
    agent_state.update_step("breaking_down", 40, "Breaking down requirement into tasks...")
    log_workflow_step("breakdown_requirement", "started", {"requirement_type": requirement_type})
    
    prompt = f"""Break down the following detailed {requirement_type} requirement into individual tasks.
    Each task should be specific, actionable, and independent.
    Format the response as a JSON array of task descriptions.
    
    Detailed Requirement: {detailed_requirement}
    
    Response format:
    ["task1 description", "task2 description", ...]"""
    
    try:
        agent_state.add_message("Analyzing requirement structure...")
        content = safe_bedrock_call(bedrock_client, prompt, request_type="requirement_breakdown")
        
        if not content:
            agent_state.set_error("Failed to get breakdown response")
            log_workflow_step("breakdown_requirement", "failed", {"error": "empty response"})
            return None
        
        agent_state.add_message("Parsing task breakdown...")
        
        # Extract JSON from response
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx]
            tasks = json.loads(json_str)
            agent_state.update_step("broken_down", 60, f"✅ Breakdown complete! Found {len(tasks)} tasks")
            log_workflow_step("breakdown_requirement", "completed", 
                            {"requirement_type": requirement_type, "task_count": len(tasks)})
            return tasks
        else:
            agent_state.set_error("Could not find valid task list in response")
            log_workflow_step("breakdown_requirement", "failed", {"error": "invalid JSON format"})
            return None
            
    except json.JSONDecodeError as e:
        agent_state.set_error(f"Error parsing task JSON: {str(e)}")
        log_workflow_step("breakdown_requirement", "error", {"error": f"JSON parsing: {str(e)}"})
        return None
    except Exception as e:
        agent_state.set_error(f"Error breaking down requirement: {str(e)}")
        log_workflow_step("breakdown_requirement", "error", {"error": str(e)})
        return None

def create_jira_story_agent(requirement: str, requirement_type: str, bedrock_client, task_num: int, total_tasks: int, agent_state: AgentState) -> Optional[Dict]:
    """Convert requirement into a JIRA story using Claude"""
    
    progress = 60 + (task_num / total_tasks) * 30
    agent_state.update_step("creating_stories", int(progress), f"Creating JIRA story {task_num}/{total_tasks}...")
    log_workflow_step("create_jira_story", "started", 
                     {"task_num": task_num, "total_tasks": total_tasks, "requirement_type": requirement_type})
    
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
        agent_state.add_message(f"Generating story {task_num}: {requirement[:50]}...")
        content = safe_bedrock_call(bedrock_client, prompt, request_type="jira_story_creation")
        
        if not content:
            agent_state.add_message(f"⚠️ Failed to generate story {task_num}")
            log_workflow_step("create_jira_story", "failed", 
                            {"task_num": task_num, "error": "empty response"})
            return None
        
        # Extract JSON from response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx]
            story = json.loads(json_str)
            agent_state.add_message(f"✅ Story {task_num} created: {story.get('Summary', 'Unknown')}")
            log_workflow_step("create_jira_story", "completed", 
                            {"task_num": task_num, "story_summary": story.get('Summary', 'Unknown'),
                             "story_points": story.get('StoryPoints', 0)})
            return story
        else:
            agent_state.add_message(f"⚠️ Could not parse story {task_num} - invalid JSON format")
            log_workflow_step("create_jira_story", "failed", 
                            {"task_num": task_num, "error": "invalid JSON format"})
            return None
            
    except json.JSONDecodeError as e:
        agent_state.add_message(f"⚠️ JSON parsing error for story {task_num}: {str(e)}")
        log_workflow_step("create_jira_story", "error", 
                        {"task_num": task_num, "error": f"JSON parsing: {str(e)}"})
        return None
    except Exception as e:
        agent_state.add_message(f"⚠️ Error creating story {task_num}: {str(e)}")
        log_workflow_step("create_jira_story", "error", 
                        {"task_num": task_num, "error": str(e)})
        return None

def run_agent_workflow(basic_requirement: str, requirement_type: str, bedrock_client):
    """Run the complete agentic workflow"""
    
    # Initialize agent state
    agent_state = AgentState()
    
    log_workflow_step("agentic_workflow", "started", 
                     {"requirement_type": requirement_type, "requirement_length": len(basic_requirement)})
    
    # Step 1: Expand requirement
    detailed_requirement = expand_requirement_agent(basic_requirement, requirement_type, bedrock_client, agent_state)
    if not detailed_requirement:
        log_workflow_step("agentic_workflow", "failed", {"failed_step": "requirement_expansion"})
        return agent_state, None, []
    
    # Step 2: Break down into tasks
    tasks = breakdown_requirement_agent(detailed_requirement, requirement_type, bedrock_client, agent_state)
    if not tasks:
        log_workflow_step("agentic_workflow", "failed", {"failed_step": "requirement_breakdown"})
        return agent_state, detailed_requirement, []
    
    # Step 3: Create JIRA stories
    stories = []
    for i, task in enumerate(tasks, 1):
        story = create_jira_story_agent(task, requirement_type, bedrock_client, i, len(tasks), agent_state)
        if story:
            stories.append(story)
        
        # Small delay to prevent rate limiting
        time.sleep(0.5)
    
    agent_state.update_step("completed", 100, f"🎉 Workflow completed! Generated {len(stories)} stories from {len(tasks)} tasks")
    
    log_workflow_step("agentic_workflow", "completed", 
                     {"requirement_type": requirement_type, "total_tasks": len(tasks), 
                      "successful_stories": len(stories), "success_rate": len(stories)/len(tasks) if tasks else 0})
    
    return agent_state, detailed_requirement, stories

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
    
    return bytes(pdf.output(dest='S'))

def create_zip_archive(stories: List[Dict], detailed_requirement: str, csv_data: bytes, pdf_data: bytes) -> bytes:
    """Create a ZIP archive containing all export formats"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('jira_stories.csv', csv_data)
        zip_file.writestr('jira_stories.pdf', pdf_data)
        
        json_data = json.dumps({
            'detailed_requirement': detailed_requirement,
            'stories': stories
        }, indent=2)
        zip_file.writestr('jira_stories.json', json_data)
        
        readme_content = f"""JIRA Stories Export
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This archive contains:
1. jira_stories.csv - CSV format of all stories
2. jira_stories.pdf - PDF report with detailed formatting
3. jira_stories.json - JSON format of all data
"""
        zip_file.writestr('README.txt', readme_content)
    
    return zip_buffer.getvalue()

def download_stories_csv(stories: List[Dict]) -> bytes:
    """Convert stories to CSV format for download"""
    df = pd.DataFrame(stories)
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.set_page_config(page_title="Agentic JIRA Converter", layout="wide")
    st.title("🤖 Agentic Requirements to JIRA Stories Converter")
    
    # Initialize AWS Bedrock
    bedrock_client = initialize_bedrock()
    if not bedrock_client:
        st.stop()
    
    # Initialize session state
    if 'agent_state' not in st.session_state:
        st.session_state.agent_state = None
    if 'detailed_requirement' not in st.session_state:
        st.session_state.detailed_requirement = None
    if 'stories' not in st.session_state:
        st.session_state.stories = []
    if 'workflow_running' not in st.session_state:
        st.session_state.workflow_running = False
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Requirements")
        
        requirement_type = st.selectbox(
            "Requirement Type",
            ["Business", "Technical", "Product", "Quality"]
        )
        
        basic_requirement = st.text_area(
            "Enter Basic Requirement",
            height=150,
            placeholder="Enter your basic requirement here...",
            disabled=st.session_state.workflow_running
        )
        
        if st.button("🚀 Start Agentic Workflow", disabled=st.session_state.workflow_running or not basic_requirement):
            st.session_state.workflow_running = True
            st.rerun()
    
    with col2:
        st.header("🤖 Agent Status")
        
        if st.session_state.workflow_running:
            # Create placeholders for dynamic updates
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            messages_placeholder = st.empty()
            
            # Run the workflow
            agent_state, detailed_requirement, stories = run_agent_workflow(
                basic_requirement, requirement_type, bedrock_client
            )
            
            # Update session state
            st.session_state.agent_state = agent_state
            st.session_state.detailed_requirement = detailed_requirement
            st.session_state.stories = stories
            st.session_state.workflow_running = False
            
            st.rerun()
        
        elif st.session_state.agent_state:
            # Display agent status
            agent_state = st.session_state.agent_state
            
            if agent_state.current_step == "error":
                st.error(f"❌ Error: {agent_state.error}")
            elif agent_state.current_step == "completed":
                st.success("✅ Workflow completed successfully!")
            else:
                st.info(f"Current step: {agent_state.current_step}")
            
            # Progress bar
            st.progress(agent_state.progress / 100)
            
            # Messages
            st.subheader("📋 Agent Messages")
            messages_container = st.container()
            with messages_container:
                for msg in agent_state.messages[-10:]:  # Show last 10 messages
                    st.text(msg)
        
        else:
            st.info("Ready to start workflow...")
    
    # Display results
    if st.session_state.detailed_requirement:
        st.header("📋 Detailed Requirement")
        st.write(st.session_state.detailed_requirement)
    
    if st.session_state.stories:
        st.header("📊 Generated JIRA Stories")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stories", len(st.session_state.stories))
        with col2:
            total_points = sum(story.get('StoryPoints', 0) for story in st.session_state.stories)
            st.metric("Total Story Points", total_points)
        with col3:
            high_priority = sum(1 for story in st.session_state.stories if story.get('Priority') == 'High')
            st.metric("High Priority Stories", high_priority)
        
        # Display stories
        for i, story in enumerate(st.session_state.stories, 1):
            with st.expander(f"📋 Story {i}: {story['Summary']}", expanded=False):
                st.write("**Description:**")
                st.write(story['Description'])
                st.write("**Acceptance Criteria:**")
                st.write(story['AcceptanceCriteria'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Story Points", story['StoryPoints'])
                with col2:
                    st.metric("Priority", story['Priority'])
                with col3:
                    st.write("**Labels:**")
                    st.write(", ".join(story['Labels']))
        
        # Download section
        st.header("📥 Download Options")
        col1, col2, col3, col4 = st.columns(4)
        
        csv_data = download_stories_csv(st.session_state.stories)
        with col1:
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"jira_stories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        if st.session_state.detailed_requirement:
            pdf_data = create_pdf_report(st.session_state.stories, st.session_state.detailed_requirement)
            with col2:
                st.download_button(
                    label="📑 Download PDF",
                    data=pdf_data,
                    file_name=f"jira_stories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            
            zip_data = create_zip_archive(
                st.session_state.stories,
                st.session_state.detailed_requirement,
                csv_data,
                pdf_data
            )
            with col3:
                st.download_button(
                    label="📦 Download ZIP",
                    data=zip_data,
                    file_name=f"jira_stories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
        with col4:
            if st.button("🔄 Clear All"):
                st.session_state.agent_state = None
                st.session_state.detailed_requirement = None
                st.session_state.stories = []
                st.session_state.workflow_running = False
                st.rerun()

if __name__ == "__main__":
    main()

