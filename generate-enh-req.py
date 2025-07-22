import streamlit as st
import boto3
import json
import uuid
import time
from typing import Dict, List
import os
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import docx
import docx2txt
import requests
import tempfile

# AWS Configuration
S3_BUCKET = "sdlc-demo-bkt"
AWS_REGION = "us-east-1"

# Initialize AWS clients
@st.cache_resource
def init_aws_clients():
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=AWS_REGION
    )
    s3 = boto3.client('s3', region_name=AWS_REGION)
    transcribe = boto3.client('transcribe', region_name=AWS_REGION)
    return bedrock, s3, transcribe

# Model configuration
MODEL_OPTIONS = {
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0"
}

# Function to extract text from different file formats
def extract_text_from_file(file_content, file_name, file_type):
    """
    Extract text from various file formats
    """
    try:
        if file_type in ['txt', 'md', 'rtf']:
            # Plain text files
            return file_content.decode('utf-8')
        
        elif file_type == 'docx':
            # DOCX files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Extract text using docx2txt
                text = docx2txt.process(tmp_file_path)
                return text
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        elif file_type == 'doc':
            # DOC files - require conversion or special handling
            # For now, we'll show an error message suggesting conversion
            st.error("DOC files are not directly supported. Please convert to DOCX or TXT format.")
            return None
        
        else:
            st.error(f"Unsupported file format: {file_type}")
            return None
            
    except Exception as e:
        st.error(f"Error extracting text from {file_type} file: {str(e)}")
        return None

# Function to upload file to S3
def upload_to_s3(s3_client, file_content, file_name, content_type="text/plain"):
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=file_name,
            Body=file_content,
            ContentType=content_type
        )
        return True
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return False

# Function to read file from S3
def read_from_s3(s3_client, file_name):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_name)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading from S3: {str(e)}")
        return None

# Function to generate PDF from requirement text
def generate_pdf(requirement_text: str, requirement_type: str, model_used: str, input_method: str) -> BytesIO:
    """
    Generate a PDF document from the requirement text
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=20,
        textColor=colors.darkgray
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        leading=14
    )
    
    # Story list to hold the content
    story = []
    
    # Add title
    story.append(Paragraph("Requirements Document", title_style))
    story.append(Spacer(1, 12))
    
    # Add metadata table
    metadata = [
        ['Requirement Type:', requirement_type],
        ['Generated On:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['Model Used:', model_used],
        ['Input Method:', input_method],
        ['Document ID:', str(uuid.uuid4())[:8]]
    ]
    
    metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 20))
    
    # Add requirement content
    story.append(Paragraph("Generated Requirements", subtitle_style))
    
    # Split the requirement text into paragraphs and format
    paragraphs = requirement_text.split('\n\n')
    
    for para in paragraphs:
        if para.strip():
            # Handle bullet points and formatting
            if para.strip().startswith('- ') or para.strip().startswith('• '):
                # Format as bullet point
                formatted_para = para.strip()
                story.append(Paragraph(formatted_para, normal_style))
            elif para.strip().startswith('#'):
                # Format as heading
                heading_text = para.strip().replace('#', '').strip()
                story.append(Paragraph(heading_text, subtitle_style))
            else:
                # Regular paragraph
                story.append(Paragraph(para.strip(), normal_style))
            
            story.append(Spacer(1, 6))
    
    # Add footer
    story.append(Spacer(1, 20))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        alignment=1
    )
    story.append(Paragraph("Generated by Advanced Requirements Generator", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def transcribe_audio(transcribe_client, s3_file_key):
    job_name = f"transcription-{uuid.uuid4().hex[:8]}"
    job_uri = f"s3://{S3_BUCKET}/{s3_file_key}"
    
    try:
        # Start transcription job
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': job_uri},
            MediaFormat=s3_file_key.split('.')[-1].lower(),
            LanguageCode='en-US'
        )
        
        # Wait for job completion
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
        
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            
            # Download and parse transcript
            response = requests.get(transcript_uri)
            transcript_json = response.json()
            
            # Clean up the transcription job
            transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
            
            return transcript_json['results']['transcripts'][0]['transcript']
        else:
            st.error("Transcription failed")
            return None
            
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return None

# Function to generate requirement prompts using Claude
def generate_requirement_prompt(bedrock, model_id: str, requirement_type: str, user_input: str) -> str:
    # Base prompt template based on requirement types
    requirement_templates = {
        "Business Requirements": """
        Please analyze the following business requirement and generate a detailed response covering:
        - Strategic alignment and business outcomes
        - Operational impact and changes
        - Stakeholder analysis
        - Risk assessment and mitigation
        - Timeline and resource requirements
        
        User Input: {input}
        """,
        
        "User Requirements": """
        Please analyze the following user requirement and provide a detailed breakdown including:
        - User persona analysis
        - User stories in 'As a [user], I want to [action], so that [benefit]' format
        - Pain points and goals
        - Usage scenarios and environments
        - Success criteria from user perspective
        
        User Input: {input}
        """,
        
        "Product Requirements": """
        Please analyze the following product requirement and provide detailed specifications covering:
        - Core functional requirements
        - Non-functional requirements
        - Performance criteria
        - Integration requirements
        - Technical constraints
        
        User Input: {input}
        """,
        
        "Technical Requirements": """
        Please analyze the following technical requirement and provide detailed specifications including:
        - System architecture requirements
        - Development and implementation details
        - Integration specifications
        - Security requirements
        - Performance benchmarks
        
        User Input: {input}
        """,
        
        "Quality & Compliance": """
        Please analyze the following quality/compliance requirement and provide detailed specifications covering:
        - Quality standards and metrics
        - Compliance requirements
        - Testing protocols
        - Audit requirements
        - Documentation needs
        
        User Input: {input}
        """
    }
    
    # Prepare the prompt
    prompt = requirement_templates[requirement_type].format(input=user_input)
    
    # Prepare the request payload
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    })
    
    try:
        # Call Claude through Bedrock
        response = bedrock.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
        
    except Exception as e:
        return f"Error generating requirement: {str(e)}"

# Streamlit UI
def main():
    st.title("?? Advanced Requirements Generator")
    st.write("Generate detailed requirements using Claude models with multiple input options")
    
    # Initialize AWS clients
    bedrock, s3, transcribe = init_aws_clients()
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    selected_model = st.sidebar.selectbox(
        "Choose Claude Model",
        list(MODEL_OPTIONS.keys()),
        index=1,  # Default to Claude 3.5 Sonnet
        help="Select the Claude model for requirement generation"
    )
    
    model_id = MODEL_OPTIONS[selected_model]
    st.sidebar.success(f"Selected: {selected_model}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Requirement type selection
        requirement_type = st.selectbox(
            "Select Requirement Type",
            [
                "Business Requirements",
                "User Requirements",
                "Product Requirements",
                "Technical Requirements",
                "Quality & Compliance"
            ]
        )
    
    with col2:
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Type Text", "Upload Document", "Upload Audio File", "Upload Document + Audio"],
            help="Choose how you want to provide the requirement input"
        )
    
    # Initialize variables
    user_input = ""
    processed_input = ""
    document_content = ""
    audio_content = ""
    
    # Handle different input methods
    if input_method == "Type Text":
        user_input = st.text_area(
            "Enter your requirement description",
            height=150,
            help="Describe your requirement in detail"
        )
        processed_input = user_input
        
    elif input_method == "Upload Document":
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['txt', 'md', 'rtf', 'docx'],
            help="Upload a document file containing your requirement description"
        )
        
        if uploaded_file is not None:
            # Get file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Read file content
            file_content = uploaded_file.read()
            
            # Extract text based on file type
            extracted_text = extract_text_from_file(file_content, uploaded_file.name, file_extension)
            
            if extracted_text:
                # Generate unique filename
                file_name = f"document_requirements/{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
                
                # Upload to S3
                content_type = f"application/{file_extension}" if file_extension != 'txt' else "text/plain"
                if upload_to_s3(s3, file_content, file_name, content_type):
                    st.success("?? Document uploaded to S3 successfully!")
                    
                    # Display extracted content
                    st.text_area("Extracted Content:", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=150)
                    processed_input = extracted_text
                else:
                    st.error("? Failed to upload document to S3")
            else:
                st.error("? Failed to extract text from document")
                
    elif input_method == "Upload Audio File":
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'm4a', 'flac'],
            help="Upload an audio file to transcribe and generate requirements"
        )
        
        if uploaded_audio is not None:
            # Generate unique filename
            file_extension = uploaded_audio.name.split('.')[-1]
            audio_file_name = f"audio_requirements/{uuid.uuid4().hex[:8]}.{file_extension}"
            
            # Upload audio to S3
            if upload_to_s3(s3, uploaded_audio.read(), audio_file_name, f"audio/{file_extension}"):
                st.success("?? Audio file uploaded to S3 successfully!")
                
                # Transcribe audio
                with st.spinner("?? Transcribing audio... This may take a few minutes."):
                    transcript = transcribe_audio(transcribe, audio_file_name)
                    
                if transcript:
                    st.success("? Audio transcription completed!")
                    st.text_area("Transcribed Text:", transcript, height=150)
                    processed_input = transcript
                else:
                    st.error("? Failed to transcribe audio")
            else:
                st.error("? Failed to upload audio file to S3")
    
    elif input_method == "Upload Document + Audio":
        st.write("?? Upload both document and audio files for comprehensive requirement analysis")
        
        col_doc, col_audio = st.columns(2)
        
        with col_doc:
            st.subheader("Document Upload")
            uploaded_document = st.file_uploader(
                "Choose a document file",
                type=['txt', 'md', 'rtf', 'docx'],
                help="Upload a document file containing specifications",
                key="doc_upload"
            )
            
            if uploaded_document is not None:
                file_extension = uploaded_document.name.split('.')[-1].lower()
                file_content = uploaded_document.read()
                
                extracted_text = extract_text_from_file(file_content, uploaded_document.name, file_extension)
                
                if extracted_text:
                    file_name = f"combined_requirements/doc_{uuid.uuid4().hex[:8]}_{uploaded_document.name}"
                    content_type = f"application/{file_extension}" if file_extension != 'txt' else "text/plain"
                    
                    if upload_to_s3(s3, file_content, file_name, content_type):
                        st.success("?? Document uploaded successfully!")
                        st.text_area("Document Content:", extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text, height=100, key="doc_preview")
                        document_content = extracted_text
                    else:
                        st.error("? Failed to upload document")
        
        with col_audio:
            st.subheader("Audio Upload")
            uploaded_audio_combined = st.file_uploader(
                "Choose an audio file",
                type=['mp3', 'wav', 'm4a', 'flac'],
                help="Upload an audio file for additional context",
                key="audio_upload"
            )
            
            if uploaded_audio_combined is not None:
                file_extension = uploaded_audio_combined.name.split('.')[-1]
                audio_file_name = f"combined_requirements/audio_{uuid.uuid4().hex[:8]}.{file_extension}"
                
                if upload_to_s3(s3, uploaded_audio_combined.read(), audio_file_name, f"audio/{file_extension}"):
                    st.success("?? Audio uploaded successfully!")
                    
                    with st.spinner("?? Transcribing audio..."):
                        transcript = transcribe_audio(transcribe, audio_file_name)
                        
                    if transcript:
                        st.success("? Audio transcribed!")
                        st.text_area("Audio Transcript:", transcript[:500] + "..." if len(transcript) > 500 else transcript, height=100, key="audio_preview")
                        audio_content = transcript
                    else:
                        st.error("? Failed to transcribe audio")
        
        # Combine document and audio content
        if document_content and audio_content:
            processed_input = f"DOCUMENT CONTENT:\n{document_content}\n\nAUDIO TRANSCRIPT:\n{audio_content}"
            st.success("? Both document and audio processed successfully!")
        elif document_content:
            processed_input = document_content
            st.info("?? Using document content only")
        elif audio_content:
            processed_input = audio_content
            st.info("?? Using audio content only")
    
    # Generate requirement button
    if st.button("?? Generate Requirement", type="primary"):
        if processed_input:
            with st.spinner(f"Generating detailed requirement using {selected_model}..."):
                # Call Claude to generate the requirement
                generated_requirement = generate_requirement_prompt(
                    bedrock,
                    model_id,
                    requirement_type,
                    processed_input
                )
                
                # Display the generated requirement
                st.subheader("?? Generated Requirement:")
                st.markdown(generated_requirement)
                
                # Add export options
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.download_button(
                        label="?? Download as Text",
                        data=generated_requirement,
                        file_name=f"{requirement_type.lower().replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Generate and download PDF
                    pdf_buffer = generate_pdf(
                        generated_requirement, 
                        requirement_type, 
                        selected_model, 
                        input_method
                    )
                    st.download_button(
                        label="?? Download as PDF",
                        data=pdf_buffer,
                        file_name=f"{requirement_type.lower().replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                
                with col3:
                    # Save to S3 option
                    if st.button("?? Save to S3"):
                        save_filename = f"generated_requirements/{uuid.uuid4().hex[:8]}_{requirement_type.lower().replace(' ', '_')}.txt"
                        if upload_to_s3(s3, generated_requirement, save_filename):
                            st.success(f"? Saved to S3: {save_filename}")
                        else:
                            st.error("? Failed to save to S3")
                
                with col4:
                    # Copy to clipboard (display only)
                    if st.button("??? Preview"):
                        st.code(generated_requirement[:200] + "..." if len(generated_requirement) > 200 else generated_requirement)
                    
        else:
            st.warning("?? Please provide input using one of the available methods")
    
    # Footer
    st.markdown("---")
    st.markdown("? **Features:** Text input, Document upload (TXT, MD, RTF, DOCX), Audio transcription, Combined document + audio, Multiple Claude models, S3 integration, PDF export")

if __name__ == "__main__":
    main()

