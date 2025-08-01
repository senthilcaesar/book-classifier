import streamlit as st
import pandas as pd
import openai
import time
from typing import List, Dict
import io
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Book Classification Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional styling inspired by diabetes nutrition app
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fa;
    color: #2c3e50;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding: 1rem 2rem;
    max-width: 1200px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    margin: 1rem auto;
}

/* Header styling */
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #4A90E2;
    text-align: left;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.main-subtitle {
    font-size: 1.1rem;
    color: #6c757d;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #f8f9fa;
    padding: 1rem;
}

.sidebar .sidebar-content {
    background: transparent;
}

/* Navigation cards in sidebar */
.nav-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    transition: all 0.2s ease;
    cursor: pointer;
}

.nav-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transform: translateY(-1px);
}

.nav-card.active {
    background: #4A90E2;
    color: white;
    border-color: #4A90E2;
}

.nav-card h4 {
    margin: 0;
    font-size: 0.9rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Modern cards */
.modern-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border: 1px solid #e9ecef;
    transition: all 0.2s ease;
}

.modern-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Metric cards */
.metric-card {
    background: #4A90E2;
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(74, 144, 226, 0.2);
    border: none;
    transition: all 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
}

.metric-card h4 {
    font-size: 0.85rem;
    font-weight: 500;
    opacity: 0.9;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-card h2 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}

/* Status boxes */
.success-box {
    background: #28a745;
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    border: none;
    box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
    margin: 1rem 0;
}

.warning-box {
    background: #ffc107;
    color: #212529;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    border: none;
    box-shadow: 0 2px 8px rgba(255, 193, 7, 0.2);
    margin: 1rem 0;
}

.info-box {
    background: #17a2b8;
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    border: none;
    box-shadow: 0 2px 8px rgba(23, 162, 184, 0.2);
    margin: 1rem 0;
}

.error-box {
    background: #dc3545;
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    border: none;
    box-shadow: 0 2px 8px rgba(220, 53, 69, 0.2);
    margin: 1rem 0;
}

/* Progress indicators */
.step-indicator {
    display: flex;
    justify-content: space-between;
    margin: 1.5rem 0;
    padding: 1rem;
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border: 1px solid #e9ecef;
}

.step-item {
    text-align: center;
    flex: 1;
    padding: 0.75rem;
    border-radius: 8px;
    transition: all 0.2s ease;
    font-size: 0.9rem;
}

.step-item.completed {
    background: #28a745;
    color: white;
    font-weight: 500;
}

.step-item.active {
    background: #4A90E2;
    color: white;
    font-weight: 500;
}

.step-item.inactive {
    color: #6c757d;
    background: #f8f9fa;
}

/* Buttons */
.stButton > button {
    background: #4A90E2;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(74, 144, 226, 0.2);
}

.stButton > button:hover {
    background: #357abd;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(74, 144, 226, 0.3);
}

.stButton > button:focus {
    box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.25);
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: white;
    border-radius: 12px;
    padding: 0.25rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border: 1px solid #e9ecef;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    font-weight: 500;
    transition: all 0.2s ease;
    color: #6c757d;
}

.stTabs [aria-selected="true"] {
    background: #4A90E2;
    color: white !important;
}

/* File uploader */
.stFileUploader {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    border: 2px dashed #4A90E2;
    transition: all 0.2s ease;
    text-align: center;
}

.stFileUploader:hover {
    border-color: #357abd;
    background: rgba(74, 144, 226, 0.02);
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border: 1px solid #e9ecef;
}

/* Selectbox and input styling */
.stSelectbox > div > div {
    background: white;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    transition: all 0.2s ease;
}

.stSelectbox > div > div:focus-within {
    border-color: #4A90E2;
    box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
}

.stTextInput > div > div {
    background: white;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    transition: all 0.2s ease;
}

.stTextInput > div > div:focus-within {
    border-color: #4A90E2;
    box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
}

/* Progress bar */
.stProgress > div > div {
    background: #4A90E2;
    border-radius: 8px;
}

/* Expander */
.streamlit-expanderHeader {
    background: #f8f9fa;
    border-radius: 8px;
    font-weight: 500;
    border: 1px solid #e9ecef;
}

/* Sidebar text styling */
.sidebar .markdown-text-container {
    color: #2c3e50;
}

/* Custom section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 1rem;
    text-align: left;
}

/* Processing table styling */
.processing-table {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    margin: 1rem 0;
    border: 1px solid #e9ecef;
}

/* Alert styling */
.stAlert {
    border-radius: 8px;
    border: none;
}

/* Metric styling */
.stMetric {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border: 1px solid #e9ecef;
}
</style>
""", unsafe_allow_html=True)

def classify_book_batch(books: List[Dict[str, str]], client, progress_bar, status_text) -> List[str]:
    """Classify a batch of books using OpenAI API with progress updates"""
    
    book_list = []
    for i, book in enumerate(books):
        book_info = f"{i+1}. \"{book['Title']}\" by {book['Author']}"
        if book.get('Summary') and str(book['Summary']).strip() and str(book['Summary']).lower() not in ['null', 'none', '']:
            book_info += f" - {book['Summary']}"
        book_list.append(book_info)
    
    book_text = "\n".join(book_list)
    
    prompt = f"""Please classify each of the following books into ONE of these categories:

CATEGORIES:
- Biography
- Business
- Economics  
- Finance
- History
- Philosophy
- Science
- Technology
- Psychology
- Self-Help
- Politics
- Health/Wellness
- Fiction
- Mystery/Thriller
- Romance
- Fantasy/Sci-Fi
- Memoir
- Travel
- Cooking
- Art/Design
- Education
- Sports
- Religion/Spirituality
- Culture
- Non-fiction (general)
- Other

BOOKS TO CLASSIFY:
{book_text}

INSTRUCTIONS:
- Respond with ONLY a numbered list matching the input
- Use the exact category names from the list above
- Base classification on the book title, author, and summary (if provided)
- If uncertain, choose the most likely category
- For very broad or unclear books, use "Non-fiction (general)" or "Other"

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
1. Category Name
2. Category Name
3. Category Name
(etc.)"""

    try:
        status_text.text("🤖 Sending request to OpenAI...")
        
        response = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=[
                {"role": "system", "content": "You are a book classification expert. Follow the instructions precisely and respond only with the requested format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        status_text.text("📝 Processing response...")
        
        content = response.choices[0].message.content
        
        # Parse the response to extract categories
        lines = content.strip().split('\n')
        categories = []
        
        for line in lines:
            if line.strip() and any(char.isdigit() for char in line[:3]):
                category = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                categories.append(category)
        
        return categories
        
    except Exception as e:
        st.error(f"Error in API call: {e}")
        return ["Other"] * len(books)

def generate_book_summary(title: str, author: str, client) -> str:
    """Generate a brief summary for a book using OpenAI API"""
    
    prompt = f"""Please provide a brief 1-2 sentence summary of the book "{title}" by {author}.

Focus on:
- What the book is about (main topic/theme)
- The type/genre of book it is
- Key subject matter

Keep it concise and factual. If you're not familiar with this specific book, provide a general description based on the title and author's typical work.

Format: Just the summary, no additional text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4", 
            messages=[
                {"role": "system", "content": "You are a book expert who provides concise, accurate book summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        st.error(f"Error generating summary for '{title}': {e}")
        return ""

def render_sidebar():
    """Render the sidebar configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key handling
        env_api_key = os.getenv("OPENAI_API_KEY")
        
        if env_api_key:
            st.success("✅ API Key loaded from .env file")
            api_key = env_api_key
            masked_key = f"{env_api_key[:8]}...{env_api_key[-4:]}" if len(env_api_key) > 12 else "***"
            #st.info(f"🔑 Using key: {masked_key}")
        else:
            st.warning("⚠️ No API key found in .env file")
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys or add OPENAI_API_KEY to your .env file"
            )
        
        st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # Model selection
        st.markdown("### Model Settings")
        model_options = {
            "GPT-4": "gpt-4",
            "GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        selected_model_name = st.selectbox(
            "Classification Model",
            list(model_options.keys()),
            help="GPT-4 is more accurate but more expensive"
        )
        st.session_state.selected_model = model_options[selected_model_name]
        
        # Batch size
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=15,
            value=8,
            help="Number of books to classify in each API call"
        )
        st.session_state.batch_size = batch_size
        
        

def render_upload_step_tab():
    """Render the upload step in tab format"""
    # Add input fields for book name and author name
    st.markdown("### 📚 Add a Book Name or Upload it as a CSV File or Attach Sample Data")
    book_title = st.text_input("Book Title", help="Enter the title of the book")
    col1, col2 = st.columns([3, 1])
    with col1:
        author_name = st.text_input("Author Name", help="Enter the author's name")
    with col2:
        if st.button("Find Author"):
            if book_title:
                try:
                    # Use AI model to find the author name based on the book title
                    author_name = generate_book_summary(book_title, "", openai.OpenAI(api_key=st.session_state.api_key))
                    st.session_state.author_name = author_name
                    st.success(f"✅ Author found: {author_name}")
                except Exception as e:
                    st.error(f"❌ Error finding author: {e}")
            else:
                st.error("❌ Please enter the book title to find the author.")
    
    # Button to add the book to the table
    if st.button("Add Book"):
        if book_title and author_name:
            # Create a DataFrame for the new book
            new_book_df = pd.DataFrame([{
                'Title': book_title,
                'Author': author_name,
                'Category': '',
                'Summary': ''
            }])
            
            # Add the new book to the session state
            if 'df' in st.session_state and st.session_state.df is not None:
                st.session_state.df = pd.concat([st.session_state.df, new_book_df], ignore_index=True)
            else:
                st.session_state.df = new_book_df
            
            st.success(f"✅ Book '{book_title}' by {author_name} added successfully!")
            
            # Display the updated data
            st.markdown("---")
            with st.expander("📊 Data Preview", expanded=True):
                preview_cols = ['Title', 'Author', 'Category', 'Summary']
                st.dataframe(st.session_state.df[preview_cols].head(10))
        else:
            st.error("❌ Please enter both the book title and author name.")
    
    # Existing upload section
    
    st.markdown("**Note:** The CSV file should have the columns 'Title', 'Author', 'Category', and 'Summary'.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: Title, Author, Category (Summary column is optional)"
    )
    
    # Move the button to a new row
    if st.button("Attach sample data"):
            try:
                sample_df = pd.read_csv("sample.csv")
                
                # Validate required columns
                required_columns = ['Title', 'Author', 'Category']
                if not all(col in sample_df.columns for col in required_columns):
                    st.error(f"❌ Missing required columns. Found: {list(sample_df.columns)}")
                    return
                
                st.session_state.df = sample_df
                st.success(f"✅ Sample data loaded successfully! {len(sample_df)} books found.")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                
                missing_mask = (sample_df['Category'].isna() |
                               (sample_df['Category'].astype(str).str.strip() == '') |
                               (sample_df['Category'].astype(str).str.lower() == 'none'))
                books_to_classify = missing_mask.sum()
                categorized_books = len(sample_df) - books_to_classify
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📚 Total Books</h4>
                        <h2>{len(sample_df)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>✅ Already Categorized</h4>
                        <h2>{categorized_books}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>❓ Need Classification</h4>
                        <h2>{books_to_classify}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data preview
                st.markdown("---")
                with st.expander("📊 Data Preview", expanded=False):
                    preview_cols = ['Title', 'Author', 'Category']
                    if 'Summary' in sample_df.columns:
                        preview_cols.append('Summary')
                    st.dataframe(sample_df[preview_cols].head(10))
                
                st.success("✅ Data uploaded successfully! Please click on the 'Classification' tab above to continue.")
                    
            except Exception as e:
                st.error(f"❌ Error loading sample data: {e}")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['Title', 'Author', 'Category']
            if not all(col in df.columns for col in required_columns):
                st.error(f"❌ Missing required columns. Found: {list(df.columns)}")
                return
            
            st.session_state.df = df
            st.success(f"✅ File loaded successfully! {len(df)} books found.")
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            missing_mask = (df['Category'].isna() |
                           (df['Category'].astype(str).str.strip() == '') |
                           (df['Category'].astype(str).str.lower() == 'none'))
            books_to_classify = missing_mask.sum()
            categorized_books = len(df) - books_to_classify
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📚 Total Books</h4>
                    <h2>{len(df)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>✅ Already Categorized</h4>
                    <h2>{categorized_books}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>❓ Need Classification</h4>
                    <h2>{books_to_classify}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Data preview
            st.markdown("---")
            with st.expander("📊 Data Preview", expanded=False):
                preview_cols = ['Title', 'Author', 'Category']
                if 'Summary' in df.columns:
                    preview_cols.append('Summary')
                st.dataframe(df[preview_cols].head(10))
            
            st.success("✅ Data uploaded successfully! Please click on the 'Classification' tab above to continue.")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

def render_classification_step_tab():
    """Render the classification step in tab format"""
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Please upload data first in the Upload tab.")
        return
    
    df = st.session_state.df
    missing_mask = (df['Category'].isna() |
                   (df['Category'].astype(str).str.strip() == '') |
                   (df['Category'].astype(str).str.lower() == 'none'))
    books_to_classify = missing_mask.sum()
        
    if 'classified_df' in st.session_state and st.session_state.classified_df is not None:
        
        st.success("✅ Classification complete! Please click on the 'Summarization' tab above to continue.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if books_to_classify == 0:
                st.markdown("""
                <div class="info-box">
                    <h4>ℹ️ All books already have categories</h4>
                    <p>You can still run classification to improve or verify the categories.</p>
                </div>
                """, unsafe_allow_html=True)
                # Create mask for all books when none need classification
                classification_mask = pd.Series([True] * len(df), index=df.index)
                estimated_cost = len(df) * (0.035 if st.session_state.selected_model == "gpt-4" else 0.005)
                button_text = "🎯 Classify All Books"
            else:
                estimated_cost = books_to_classify * (0.035 if st.session_state.selected_model == "gpt-4" else 0.005)
                classification_mask = missing_mask
                button_text = "Start Classification"
                st.markdown(f"""
                <div class="warning-box">
                    <h4>📋 Ready to classify {books_to_classify} books</h4>
                    <p><strong>Model:</strong> {st.session_state.selected_model}</p>
                    <p><strong>Batch size:</strong> {st.session_state.batch_size} books per request</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Move the buttons to a new row
        if st.button(button_text, type="primary", use_container_width=True, disabled=not st.session_state.api_key):
            if not st.session_state.api_key:
                st.error("❌ Please enter your OpenAI API key in the sidebar.")
            else:
                classify_books_with_live_table(classification_mask)
                
        # Skip classification button
        st.markdown("---")
        if st.button("Skip Classification", type="secondary", use_container_width=True):
            # Set classified_df to current df and proceed
            st.session_state.classified_df = df
            st.info("✅ Classification skipped! Please click on the 'Summarization' tab above to continue.")

def render_summarization_step_tab():
    """Render the summarization step in tab format"""
    if 'classified_df' not in st.session_state or st.session_state.classified_df is None:
        st.warning("⚠️ Please complete classification first in the Classification tab.")
        return
    
    df = st.session_state.classified_df
    
    # Check for missing summaries
    if 'Summary' not in df.columns:
        df['Summary'] = ''
    
    missing_summary_mask = (df['Summary'].isna() |
                           (df['Summary'].astype(str).str.strip() == '') |
                           (df['Summary'].astype(str).str.lower() == 'none') |
                           (df['Summary'].astype(str).str.lower() == 'null'))
    
    books_needing_summary = missing_summary_mask.sum()
        
    if books_needing_summary == 0:
        st.markdown("""
        <div class="success-box">
            <h4>✅ All books already have summaries!</h4>
            <p>No summary generation needed. Proceed to Analysis tab.</p>
        </div>
        """, unsafe_allow_html=True)
        if 'summarized_df' not in st.session_state:
            st.session_state.summarized_df = df
        
        st.success("✅ All books have summaries! Please click on the 'Analysis' tab above to continue.")
    elif 'summarized_df' in st.session_state and st.session_state.summarized_df is not None:

        
        st.success("✅ Summary generation complete! Please click on the 'Analysis' tab above to continue.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            estimated_cost = books_needing_summary * 0.002
            st.markdown(f"""
            <div class="warning-box">
                <h4>Ready to generate summaries for {books_needing_summary} books</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Move the buttons to a new row
        if st.button("Generate Summaries", type="primary", use_container_width=True, disabled=not st.session_state.api_key):
            if not st.session_state.api_key:
                st.error("❌ Please enter your OpenAI API key in the sidebar.")
            else:
                generate_summaries(missing_summary_mask)
                
        # Skip summarization button
        st.markdown("---")
        if st.button("Skip Summaries", type="secondary", use_container_width=True):
            # Set summarized_df to current df and proceed
            st.session_state.summarized_df = st.session_state.classified_df
            st.info("✅ Summaries skipped! Please click on the 'Analysis' tab above to continue.")

def render_analysis_step_tab():
    """Render the final analysis step in tab format"""
    if 'summarized_df' not in st.session_state or st.session_state.summarized_df is None:
        st.warning("⚠️ Please complete previous steps first.")
        return
    
    st.markdown("## Results & Analysis")
    
    df = st.session_state.summarized_df
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📚 Total Books Processed</h4>
            <h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        category_count = len(df['Category'].fillna('Uncategorized').unique())
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏷️ Unique Categories</h4>
            <h2>{category_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        summary_count = len(df[df['Summary'].notna() & (df['Summary'].astype(str).str.strip() != '')])
        st.markdown(f"""
        <div class="metric-card">
            <h4>📝 Books with Summaries</h4>
            <h2>{summary_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["📈 Category Analysis", "📚 Book Explorer", "💾 Download Results"])
    
    with tab1:
        st.subheader("Category Distribution")
        category_counts = df['Category'].fillna('Uncategorized').value_counts()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(category_counts)
        
        with col2:
            st.markdown("**Top Categories:**")
            for i, (category, count) in enumerate(category_counts.head(10).items()):
                percentage = (count / len(df)) * 100
                st.metric(
                    label=f"{i+1}. {category}",
                    value=f"{count} books",
                    delta=f"{percentage:.1f}%"
                )
    
    with tab2:
        st.subheader("Browse Your Books")
        
        # Category filter
        categories = ['All'] + sorted(df['Category'].fillna('Uncategorized').unique().tolist())
        selected_category = st.selectbox("Filter by category:", categories)
        
        if selected_category == 'All':
            filtered_df = df
        else:
            if selected_category == 'Uncategorized':
                filtered_df = df[df['Category'].isna()]
            else:
                filtered_df = df[df['Category'] == selected_category]
        
        st.write(f"📖 Showing {len(filtered_df)} books")
        
        # Display books
        available_cols = ['Title', 'Author', 'Category']
        if 'Summary' in filtered_df.columns:
            available_cols.append('Summary')
        
        display_cols = [col for col in available_cols if col in filtered_df.columns]
        display_df = filtered_df[display_cols].copy()
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    with tab3:
        st.subheader("Download Your Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download complete dataset
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Complete Dataset",
                data=csv_data,
                file_name=f"book_analysis_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the complete dataset with classifications and summaries",
                use_container_width=True
            )
        
        with col2:
            # Download summary statistics
            summary_stats = {
                'Total Books': len(df),
                'Categories': len(df['Category'].unique()),
                'Books with Summaries': len(df[df['Summary'].notna()]),
                'Most Common Category': df['Category'].mode()[0] if len(df['Category'].mode()) > 0 else 'N/A'
            }
            
            stats_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
            stats_csv = stats_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Summary Statistics",
                data=stats_csv,
                file_name=f"book_analysis_stats_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download summary statistics about your book collection",
                use_container_width=True
            )
        
        st.success("✅ Analysis complete! Your book collection has been fully processed and analyzed.")
        
        # Navigation options
        st.markdown("---")


def classify_books_with_live_table(missing_mask):
    """Perform book classification with live updating table"""
    try:
        client = openai.OpenAI(api_key=st.session_state.api_key)
        df = st.session_state.df.copy()
        
        books_to_classify = df[missing_mask].copy()
        total_books = len(books_to_classify)
        
        st.markdown("### 🔄 Live AI Classification - Watch the table update in real-time!")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        processing_table = st.empty()
        
        # Initialize processing results table with all books as "Pending"
        processing_results = []
        books = []
        
        for _, row in books_to_classify.iterrows():
            book_data = {
                'Title': str(row['Title']).strip(),
                'Author': str(row['Author']).strip(),
                'Summary': str(row['Summary']) if pd.notna(row['Summary']) else ''
            }
            books.append(book_data)
            
            processing_results.append({
                'Status': '⏳ Pending',
                'Title': book_data['Title'][:50] + '...' if len(book_data['Title']) > 50 else book_data['Title'],
                'Author': book_data['Author'][:30] + '...' if len(book_data['Author']) > 30 else book_data['Author'],
                'Category': '...',
                'Summary': book_data['Summary'][:80] + '...' if len(book_data['Summary']) > 80 else book_data['Summary'] or 'No summary'
            })
        
        # Display initial table with all pending books
        with processing_table.container():
            st.markdown("#### 📊 Live Processing Status")
            results_df = pd.DataFrame(processing_results)
            st.dataframe(results_df, hide_index=True, use_container_width=True)
        
        # Process in batches with live updates
        all_categories = []
        batch_size = st.session_state.batch_size
        total_batches = (total_books + batch_size - 1) // batch_size
        processed_count = 0
        
        for i in range(0, total_books, batch_size):
            batch = books[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            status_text.text(f"🎯 Processing batch {batch_num}/{total_batches} ({len(batch)} books)")
            
            # Update status to "Processing" for current batch
            for j, book in enumerate(batch):
                book_idx = i + j
                processing_results[book_idx]['Status'] = '🔄 Processing...'
            
            # Refresh table to show processing status
            with processing_table.container():
                st.markdown(f"#### 📊 Live Processing Status ({processed_count}/{total_books} completed)")
                results_df = pd.DataFrame(processing_results)
                st.dataframe(results_df, hide_index=True, use_container_width=True)
            
            categories = classify_book_batch(batch, client, progress_bar, status_text)
            all_categories.extend(categories)
            
            # Update each book individually as "Completed"
            for j, (book, category) in enumerate(zip(batch, categories)):
                book_idx = i + j
                processing_results[book_idx]['Status'] = '✅ Completed'
                processing_results[book_idx]['Category'] = category
                processed_count += 1
                
                # Update progress
                progress = processed_count / total_books
                progress_bar.progress(progress)
                
                # Refresh table after each book completion
                with processing_table.container():
                    st.markdown(f"#### 📊 Live Processing Status ({processed_count}/{total_books} completed)")
                    results_df = pd.DataFrame(processing_results)
                    st.dataframe(results_df, hide_index=True, use_container_width=True)
                
                # Small delay to show the live update effect
                time.sleep(0.5)
            
            # Rate limiting between batches
            if i + batch_size < total_books:
                time.sleep(2)
        
        # Update dataframe with results - fix pandas warning
        categories_series = pd.Series(all_categories[:len(books_to_classify)], dtype='object')
        df.loc[missing_mask, 'Category'] = categories_series
        st.session_state.classified_df = df
        
        progress_bar.progress(1.0)
        status_text.text("✅ All books classified successfully!")
        
        st.success(f"🎉 Successfully classified {total_books} books with live AI processing!")
        st.balloons()
        
        # Auto-refresh to show updated state
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Classification failed: {e}")

def generate_summaries(missing_summary_mask):
    """Generate AI summaries for books"""
    try:
        client = openai.OpenAI(api_key=st.session_state.api_key)
        df = st.session_state.classified_df.copy()
        
        books_needing_summary = df[missing_summary_mask].copy()
        total_books = len(books_needing_summary)
        
        st.markdown("### 📝 Summary Generation in Progress")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        updated_summaries = []
        
        for i, (idx, row) in enumerate(books_needing_summary.iterrows()):
            status_text.text(f"📝 Generating summary {i+1}/{total_books}: {row['Title'][:50]}...")
            
            summary = generate_book_summary(row['Title'], row['Author'], client)
            updated_summaries.append((idx, summary))
            
            # Update progress
            progress = (i + 1) / total_books
            progress_bar.progress(progress)
            
            # Show progress in results container
            with results_container.container():
                st.markdown(f"**📚 Recently Generated ({i+1}/{total_books}):**")
                recent_summaries = updated_summaries[-3:]  # Show last 3
                for _, summary in recent_summaries:
                    book_row = df.loc[_]
                    st.write(f"• **{book_row['Title']}** by {book_row['Author']}")
                    st.write(f"  _{summary}_")
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
        
        # Update the dataframe with new summaries
        for idx, summary in updated_summaries:
            df.at[idx, 'Summary'] = summary
        
        st.session_state.summarized_df = df
        
        progress_bar.progress(1.0)
        status_text.text("✅ Summary generation completed!")
        
        st.success(f"📝 Successfully generated summaries for {total_books} books!")
        st.balloons()
        
        # Auto-refresh to show updated state
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Summary generation failed: {e}")

def main():
    """Main application function"""
    # Header with clean, professional styling
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 class="main-header">Book Classification & Summary Tool</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    render_sidebar()
    
    # Main content with horizontal tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "🎯 Classification", "📝 Summarization", "📊 Analytics"])
    
    with tab1:
        render_upload_step_tab()
    
    with tab2:
        render_classification_step_tab()
    
    with tab3:
        render_summarization_step_tab()
    
    with tab4:
        render_analysis_step_tab()


if __name__ == "__main__":
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'classified_df' not in st.session_state:
        st.session_state.classified_df = None
    if 'summarized_df' not in st.session_state:
        st.session_state.summarized_df = None
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gpt-4"
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 8
    
    main()
