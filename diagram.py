import streamlit as st
import streamlit.components.v1 as components
import boto3
import json
from botocore.config import Config
import re
import base64
from io import BytesIO

def initialize_bedrock_client():
    """Initialize and return the AWS Bedrock client"""
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'  # Replace with your AWS region
        )
        return bedrock
    except Exception as e:
        st.error(f"Error connecting to AWS Bedrock: {str(e)}")
        return None

def clean_mermaid_code(mermaid_code):
    """Clean and validate Mermaid code"""
    # Remove any potential markdown code block markers and extra whitespace
    mermaid_code = re.sub(r'```mermaid\s*|\s*```', '', mermaid_code.strip())
    
    # Ensure proper line endings
    mermaid_code = mermaid_code.replace('\r\n', '\n').replace('\r', '\n')
    
    # Ensure diagram type declaration is correct
    diagram_types = {
        "stateDiagram": "stateDiagram-v2",
        "classDiagram": "classDiagram",
        "sequenceDiagram": "sequenceDiagram",
        "erDiagram": "erDiagram"
    }
    
    # Check and fix diagram type declarations
    first_line = mermaid_code.split('\n')[0] if mermaid_code else ""
    for old_type, new_type in diagram_types.items():
        if old_type in first_line:
            mermaid_code = mermaid_code.replace(first_line, new_type, 1)
            break
    
    # Special cleaning for class diagrams
    if 'classDiagram' in mermaid_code:
        # Remove any problematic characters that might cause syntax errors
        lines = mermaid_code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Clean up class names and remove problematic characters
            line = re.sub(r'[^\w\s\{\}\+\-\#\(\)\:\*\|\>\<\.\-\~]', '', line)
            
            # Ensure proper spacing around relationship arrows
            line = re.sub(r'(\w)\s*(--[>\*o\|]?)\s*(\w)', r'\1 \2 \3', line)
            
            cleaned_lines.append(line)
        
        mermaid_code = '\n'.join(cleaned_lines)
    
    # Remove any empty lines at the start or end
    mermaid_code = mermaid_code.strip()
    
    return mermaid_code

def get_claude_response(bedrock_client, prompt, temp=0.1):
    """Get response from Claude with improved prompt"""
    instructions = """Create a Mermaid diagram following these exact rules:
    1. Start with the correct diagram declaration (stateDiagram-v2, classDiagram, sequenceDiagram, or erDiagram)
    2. Use proper syntax specific to the diagram type
    3. Avoid any special characters or formatting that could break the Mermaid syntax
    4. Include all necessary elements and relationships
    5. Return only the Mermaid code without any additional text or markdown
    6. For class diagrams: Use proper class member syntax with {} brackets or : notation
    7. For class diagrams: Avoid special characters in class names, use simple alphanumeric names
    8. For class diagrams: Use proper relationship syntax (--|>, -->, --*, --o, ..|>, ..>, ..*)
    9. Ensure all class names are simple without spaces or special characters
    10. Use proper indentation and line endings
    """
    
    full_prompt = f"{instructions}\n\nCreate the following diagram:\n{prompt}"
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        "temperature": temp,
        "top_p": 0.9,
    })

    try:
        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=body
        )
        response_body = json.loads(response.get('body').read())
        return clean_mermaid_code(response_body['content'][0]['text'])
    except Exception as e:
        raise Exception(f"Error getting response from Claude: {str(e)}")

def generate_diagram_prompt(requirements, diagram_type):
    """Generate appropriate prompt based on diagram type"""
    type_specific_prompts = {
        "State Diagram": f"Create a state diagram with these requirements:\n{requirements}\n\nUse these syntax rules:\n- Start with 'stateDiagram-v2'\n- Define states using simple names without spaces\n- Use proper arrow syntax (-->)\n- Add [*] for start/end states if needed\n- Use notes with 'note' keyword\n- End each relationship with a colon and description",
        
        "Class Diagram": f"""Create a class diagram with these requirements:
{requirements}

CRITICAL: Follow these exact syntax rules for class diagrams:
- Start with 'classDiagram'
- Use simple class names without spaces (e.g., User, Product, OrderItem)
- Define class members using curly braces syntax:
  class ClassName {{
    +attribute1 type
    +attribute2 type
    +method1() returnType
    +method2() returnType
  }}
- Use proper relationship arrows: --> (association), --* (composition), --o (aggregation), --|> (inheritance)
- Example relationship: User --> Order : creates
- Use + for public, - for private, # for protected
- Avoid special characters, parentheses in class names
- Each class definition should be on separate lines
- No semicolons needed for class diagrams""",
        
        "Sequence Diagram": f"Create a sequence diagram with these requirements:\n{requirements}\n\nUse these syntax rules:\n- Start with 'sequenceDiagram'\n- Define participants using 'participant' keyword\n- Use proper arrow types (->, -->>, --x)\n- Group related actions with 'opt' or 'alt'\n- Use 'Note' for additional information",
        
        "ER Diagram": f"Create an ER diagram with these requirements:\n{requirements}\n\nUse these syntax rules:\n- Start with 'erDiagram'\n- Define entities and relationships\n- Use proper relationship symbols (||--o{{, }}|--|}}\n- Add relationship labels with proper syntax\n- Include attributes with proper types"
    }
    
    return type_specific_prompts.get(diagram_type, "")

def create_download_functionality(diagram_id):
    """Create improved download functionality using html2canvas"""
    return f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <script>
        async function downloadDiagramAsPNG_{diagram_id}() {{
            try {{
                const diagramElement = document.querySelector('#{diagram_id}');
                if (!diagramElement) {{
                    alert('Diagram not found. Please make sure the diagram is fully loaded.');
                    return;
                }}
                
                // Wait for Mermaid to finish rendering
                await new Promise(resolve => setTimeout(resolve, 2000));
                
                // Configure html2canvas options
                const options = {{
                    backgroundColor: '#ffffff',
                    scale: 2, // Higher resolution
                    useCORS: true,
                    allowTaint: true,
                    scrollX: 0,
                    scrollY: 0,
                    width: diagramElement.scrollWidth,
                    height: diagramElement.scrollHeight,
                    onclone: function(clonedDoc) {{
                        // Ensure the cloned document has the same styling
                        const clonedElement = clonedDoc.querySelector('#{diagram_id}');
                        if (clonedElement) {{
                            clonedElement.style.backgroundColor = '#ffffff';
                            clonedElement.style.padding = '20px';
                        }}
                    }}
                }};
                
                // Show loading message
                const button = document.querySelector('#download-btn-{diagram_id}');
                const originalText = button.innerHTML;
                button.innerHTML = '⏳ Generating...';
                button.disabled = true;
                
                // Generate canvas
                const canvas = await html2canvas(diagramElement, options);
                
                // Create download link
                const link = document.createElement('a');
                link.download = 'mermaid_diagram_' + new Date().toISOString().slice(0, 10) + '.png';
                link.href = canvas.toDataURL('image/png');
                
                // Trigger download
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                // Reset button
                button.innerHTML = originalText;
                button.disabled = false;
                
            }} catch (error) {{
                console.error('Error generating PNG:', error);
                alert('Error generating PNG. Please try again or check the console for details.');
                
                // Reset button
                const button = document.querySelector('#download-btn-{diagram_id}');
                button.innerHTML = '📥 Download as PNG';
                button.disabled = false;
            }}
        }}
    </script>
    <button id="download-btn-{diagram_id}" 
            onclick="downloadDiagramAsPNG_{diagram_id}()" 
            style="background-color: #4CAF50; color: white; padding: 12px 24px; 
                   border: none; border-radius: 6px; cursor: pointer; margin: 10px 0;
                   font-size: 14px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                   transition: background-color 0.3s;"
            onmouseover="this.style.backgroundColor='#45a049'"
            onmouseout="this.style.backgroundColor='#4CAF50'">
        📥 Download as PNG
    </button>
    """

def display_mermaid_diagram(mermaid_code, diagram_id="diagram"):
    """Display Mermaid diagram with improved configuration and working download functionality"""
    download_html = create_download_functionality(diagram_id)
    
    html = f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                logLevel: 'error',
                securityLevel: 'loose',
                fontSize: 16,
                fontFamily: 'arial',
                flowchart: {{
                    curve: 'linear',
                    diagramPadding: 8
                }},
                sequence: {{
                    showSequenceNumbers: true,
                    actorMargin: 50,
                    messageMargin: 40,
                    mirrorActors: false
                }},
                class: {{
                    useMaxWidth: true,
                    htmlLabels: false
                }},
                er: {{
                    entityPadding: 15,
                    layoutDirection: 'TB',
                    minEntityWidth: 100,
                    minEntityHeight: 75,
                    entitySpacing: 50,
                    wrap: true
                }},
                // Add render callback to ensure diagram is fully loaded
                callback: function(id) {{
                    console.log('Mermaid diagram rendered:', id);
                }}
            }});
            
            // Add error handling
            window.mermaid.parseError = function(err, hash) {{
                console.error('Mermaid parse error:', err);
                const errorDiv = document.querySelector('.mermaid-error');
                if (errorDiv) {{
                    errorDiv.innerHTML = 
                        '<div style="color: red; padding: 20px; border: 1px solid red; background: #ffe6e6; border-radius: 5px; margin: 10px 0;">Error: ' + err + '</div>';
                }}
            }};
            
            // Wait for DOM to be ready
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('DOM ready, Mermaid should initialize');
            }});
        </script>
        
        <div class="mermaid-error"></div>
        <div style="margin: 20px 0; padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 4px solid #4CAF50;">
            <strong>📋 Export Options:</strong>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>Click the green button below to download as PNG image</li>
                <li>Use the "Show Mermaid Code" section to get the raw code</li>
                <li>Wait for the diagram to fully load before downloading</li>
            </ul>
        </div>
        
        {download_html}
        
        <div id="{diagram_id}" class="mermaid" style="
            background-color: white; 
            padding: 30px; 
            border-radius: 8px; 
            width: 100%; 
            min-height: 400px; 
            overflow: auto;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        ">
            {mermaid_code}
        </div>
        
        <div style="margin-top: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; font-size: 12px; color: #666;">
            💡 <strong>Tip:</strong> If the download doesn't work immediately, wait a few seconds for the diagram to fully render and try again.
        </div>
    """
    
    # Increase the height for ER diagrams and add extra space for download functionality
    diagram_height = 950 if 'erDiagram' in mermaid_code else 800
    components.html(html, height=diagram_height, scrolling=True)

def create_mermaid_download_link(mermaid_code, filename="mermaid_diagram.txt"):
    """Create a download link for the Mermaid code"""
    b64 = base64.b64encode(mermaid_code.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}" style="color: #4CAF50; text-decoration: none; font-weight: bold;">📄 Download Mermaid Code (.txt)</a>'
    return href

def main():
    st.set_page_config(page_title="Diagram Generator", layout="wide")
    
    st.title("🎨 AWS Bedrock Claude Diagram Generator")
    st.markdown("Generate professional diagrams with AI and export them as PNG images!")
    
    # Initialize AWS Bedrock client
    try:
        bedrock_client = initialize_bedrock_client()
        if not bedrock_client:
            st.error("Failed to initialize AWS Bedrock client. Please check your credentials.")
            return
    except Exception as e:
        st.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
        return

    # Diagram type selection
    diagram_type = st.selectbox(
        "Select diagram type:",
        ["State Diagram", "Class Diagram", "Sequence Diagram", "ER Diagram"]
    )

    # Context-specific examples
    with st.expander("📖 See example requirements for " + diagram_type):
        examples = {
            "State Diagram": """Example: Create a login system with these states:
- LoggedOut (initial state)
- Authenticating
- LoggedIn
- Error
Include transitions for login attempt, success, failure, and logout.""",
            "Class Diagram": """Example: Create an online shopping system with:
- User class (id, name, email, methods: login(), logout())
- Product class (id, name, price, methods: updateStock())
- Order class (id, userId, total, methods: placeOrder())
- Show relationships between classes""",
            "Sequence Diagram": """Example: Show the user registration process:
1. User submits registration form
2. Frontend validates input
3. Backend checks if email exists
4. Database creates new user
5. Email service sends welcome message""",
            "ER Diagram": """Example: Create a library system with:
- Book (ISBN, title, author_id, copies)
- Author (id, name, nationality)
- Member (id, name, email)
- Loan (id, book_id, member_id, due_date)"""
        }
        st.write(examples.get(diagram_type, ""))

    # Requirements input
    requirements = st.text_area(
        "Enter your requirements:",
        height=200,
        placeholder="Describe your system requirements here..."
    )

    # Column layout for additional settings
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature slider
        temperature = st.slider("Creativity Level (Temperature)", 0.0, 1.0, 0.1, 0.1)
    
    with col2:
        # Display settings for ER diagrams
        if diagram_type == "ER Diagram":
            st.info("ER diagrams may require scrolling to view all relationships")

    if st.button("🚀 Generate Diagram", type="primary"):
        if not requirements:
            st.warning("Please enter requirements first.")
            return

        with st.spinner("Generating diagram..."):
            try:
                # Generate diagram
                prompt = generate_diagram_prompt(requirements, diagram_type)
                mermaid_code = get_claude_response(bedrock_client, prompt, temperature)
                
                # Store in session state for persistence
                st.session_state.mermaid_code = mermaid_code
                st.session_state.diagram_type = diagram_type
                
                # Display success message
                st.success("✅ Diagram generated successfully!")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("If you see a syntax error, try regenerating the diagram or adjusting your requirements.")

    # Display diagram if it exists in session state
    if 'mermaid_code' in st.session_state:
        st.subheader("📊 Generated Diagram")
        
        # Display the diagram with download functionality
        display_mermaid_diagram(st.session_state.mermaid_code, "main_diagram")

        # Show the Mermaid code and download options
        with st.expander("🔍 Show Mermaid Code & Download Options"):
            st.code(st.session_state.mermaid_code, language="mermaid")
            
            # Create download link for Mermaid code
            st.markdown("**Download Options:**")
            st.markdown(create_mermaid_download_link(st.session_state.mermaid_code), unsafe_allow_html=True)
            
            # Additional export information
            st.markdown("""
            **📋 Export Information:**
            - **🖼️ PNG Export**: Click the green 'Download as PNG' button above the diagram
            - **📄 Mermaid Code**: Use the download link above to get the raw Mermaid code
            - **🔗 Integration**: You can use the Mermaid code in documentation, wikis, or other platforms that support Mermaid diagrams
            - **⚠️ Note**: Wait for the diagram to fully render before downloading as PNG
            """)

    # Additional information
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool uses AWS Bedrock with Claude AI to generate professional diagrams in Mermaid format.
        
        **🎯 Features:**
        - Generate 4 types of diagrams: State, Class, Sequence, and ER diagrams
        - Export diagrams as high-quality PNG images
        - Download raw Mermaid code for integration
        - Adjustable creativity levels
        - Syntax validation and error handling
        - Improved download functionality with html2canvas
        
        **💡 Usage Tips:**
        - Be specific in your requirements for better results
        - Use the examples as templates
        - Lower temperature values (0.1-0.3) for more structured diagrams
        - Higher temperature values (0.5-0.8) for more creative variations
        - Wait for diagrams to fully load before downloading
        - If PNG download fails, try refreshing and regenerating the diagram
        """)

if __name__ == "__main__":
    main()

