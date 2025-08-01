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

# Google AI Studio inspired clean design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600;700&display=swap');

/* Clean Google AI Studio styling */
.stApp {
    background-color: #ffffff;
}

.main .block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
    background-color: #ffffff;
    margin: 0 auto;
}

/* Clean header like Google AI Studio */
.main-header {
    font-family: 'Google Sans', sans-serif;
    font-size: 2.5rem;
    font-weight: 400;
    color: #202124;
    text-align: center;
    margin-bottom: 3rem;
    padding-bottom: 1rem;
}

/* Clean metric cards */
.metric-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e8eaed;
    margin-bottom: 1rem;
    transition: box-shadow 0.2s ease;
}

.metric-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.metric-card h4 {
    color: #202124;
    margin-bottom: 0.5rem;
    font-weight: 500;
    font-family: 'Google Sans', sans-serif;
}

.metric-card p {
    color: #5f6368;
    margin: 0.25rem 0;
    font-family: 'Google Sans', sans-serif;
}

/* Clean status boxes */
.success-box {
    background-color: #e8f5e8;
    color: #137333;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #34a853;
    margin: 1rem 0;
    font-family: 'Google Sans', sans-serif;
}

.warning-box {
    background-color: #fef7e0;
    color: #b06000;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #fbbc04;
    margin: 1rem 0;
    font-family: 'Google Sans', sans-serif;
}

.info-box {
    background-color: #e8f0fe;
    color: #1967d2;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #4285f4;
    margin: 1rem 0;
    font-family: 'Google Sans', sans-serif;
}

/* Clean section headers */
.section-header {
    font-family: 'Google Sans', sans-serif;
    font-size: 1.5rem;
    font-weight: 500;
    color: #202124;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
}

/* Clean sidebar like Google AI Studio */
.css-1d391kg {
    background-color: #f8f9fa;
    border-right: 1px solid #e8eaed;
}

/* Google-style buttons */
.stButton > button {
    background-color: #87CEEB;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    font-family: 'Google Sans', sans-serif;
    transition: background-color 0.2s ease;
    box-shadow: 0 1px 2px 0 rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15);
}

.stButton > button:hover {
    background-color: #87CEEB;
    box-shadow: 0 1px 3px 0 rgba(60,64,67,.3), 0 4px 8px 3px rgba(60,64,67,.15);
}

/* Clean file uploader */
.stFileUploader {
    border: 2px dashed #dadce0;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    background-color: #fafafa;
    transition: border-color 0.2s ease;
}

.stFileUploader:hover {
    border-color: #1a73e8;
    background-color: #f8f9fa;
}

/* Clean progress bar */
.stProgress > div > div > div {
    background-color: #1a73e8;
    border-radius: 4px;
}

/* Clean inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    border: 1px solid #dadce0;
    border-radius: 4px;
    background-color: #ffffff;
    font-family: 'Google Sans', sans-serif;
}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus {
    border-color: #1a73e8;
    box-shadow: 0 0 0 1px #1a73e8;
}

/* Clean dataframe */
.stDataFrame {
    border: 1px solid #e8eaed;
    border-radius: 8px;
    overflow: hidden;
}

/* Clean tabs */
.stTabs {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #e8eaed;
    padding: 1rem;
    margin-bottom: 2rem;
}

/* Clean typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Google Sans', sans-serif;
    color: #202124;
}

p, div, span {
    font-family: 'Google Sans', sans-serif;
    color: #5f6368;
}

/* Clean cost cards */
.cost-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: box-shadow 0.2s ease;
}

.cost-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.cost-card-header {
    display: flex;
    align-items: center;
    font-weight: 400;
    color: #202124;
    font-family: 'Google Sans', sans-serif;
}

.cost-icon {
    margin-right: 0.5rem;
    font-size: 1.1rem;
}

/* Feature badges */
.feature-badge {
    display: inline-flex;
    align-items: center;
    background: #f8f9fa;
    border: 1px solid #e8eaed;
    border-radius: 20px;
    padding: 0.75rem 1rem;
    margin: 0.25rem;
    color: #5f6368;
    font-weight: 400;
    font-family: 'Google Sans', sans-serif;
    transition: all 0.2s ease;
}

.feature-badge:hover {
    background: #e8f0fe;
    border-color: #4285f4;
    color: #1967d2;
}

.feature-icon {
    margin-right: 0.5rem;
    font-size: 1.2rem;
}

/* Clean scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f3f4;
}

::-webkit-scrollbar-thumb {
    background: #dadce0;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #bdc1c6;
}

/* Center content like Google AI Studio */
.centered-content {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem 0;
}

/* Large upload area */
.upload-area {
    max-width: 600px;
    margin: 2rem auto;
    text-align: center;
}

/* Processing section */
.processing-section {
    max-width: 700px;
    margin: 2rem auto;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def generate_summary_batch(books: List[Dict[str, str]], client, progress_bar, status_text) -> List[str]:
    """Generate summaries for a batch of books using OpenAI API"""
    
    book_list = []
    for i, book in enumerate(books):
        book_info = f"{i+1}. \"{book['Title']}\" by {book['Author']}"
        book_list.append(book_info)
    
    book_text = "\n".join(book_list)
    
    prompt = f"""Please provide a brief, concise summary (2-3 sentences) for each of the following books based on their title and author. Write about what the book is about, its main themes, concepts, or plot - do NOT start with the book title or author name.

BOOKS TO SUMMARIZE:
{book_text}

INSTRUCTIONS:
- Provide exactly 2-3 sentences per book
- Write about what the book covers, its main themes, concepts, or plot
- Do NOT start the summary with the book title or author name
- Be informative and concise about the book's content
- If you're not familiar with a specific book, provide a general summary based on the title and author's typical work
- Respond with ONLY a numbered list matching the input

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
1. This book explores [main theme/concept/plot]. It discusses [key points]. [Additional relevant information].
2. This work examines [main theme/concept/plot]. It covers [key points]. [Additional relevant information].
3. This book focuses on [main theme/concept/plot]. It presents [key points]. [Additional relevant information].
(etc.)"""

    try:
        status_text.text("📝 Generating summaries...")
        
        response = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=[
                {"role": "system", "content": "You are a knowledgeable book summarizer. Provide concise, informative summaries based on book titles and authors."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        
        # Parse the response to extract summaries
        lines = content.strip().split('\n')
        summaries = []
        
        for line in lines:
            if line.strip() and any(char.isdigit() for char in line[:3]):
                summary = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                summaries.append(summary)
        
        return summaries
        
    except Exception as e:
        st.error(f"Error generating summaries: {e}")
        return ["Summary not available."] * len(books)

def classify_book_batch(books: List[Dict[str, str]], client, progress_bar, status_text) -> List[str]:
    """Classify a batch of books using OpenAI API with progress updates"""
    
    book_list = []
    for i, book in enumerate(books):
        book_info = f"{i+1}. \"{book['Title']}\" by {book['Author']}"
        if book.get('Summary') and str(book['Summary']).strip() and str(book['Summary']).lower() != 'null':
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

def main():
    # Google AI Studio style header
    st.markdown('<h1 class="main-header">Book Classification & Summary Tool</h1>', unsafe_allow_html=True)
    
    # Centered configuration section like Google AI Studio
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    
    # Configuration in horizontal layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    env_api_key = os.getenv("OPENAI_API_KEY")
    api_key = env_api_key

    
    with col1:
        # Model selection
        model_options = {
            "GPT-4": "gpt-4",
            "GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        selected_model_name = st.selectbox(
            "Model",
            list(model_options.keys()),
            help="GPT-4 is more accurate but more expensive"
        )
        st.session_state.selected_model = model_options[selected_model_name]
    

    batch_size = 8
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main upload area - centered like Google AI Studio
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    
    # Large, prominent upload section
    uploaded_file = st.file_uploader(
        "Drop your CSV file here or click to browse",
        type=['csv'],
        help="Upload CSV with columns: Title, Author, Category, Summary (optional)",
        key="main_uploader"
    )
    
    if uploaded_file is None:
        # Show requirements when no file is uploaded
        st.markdown("""
        <div class="info-box">
            <h4>📋 Requirements</h4>
            <p><strong>Required columns:</strong> Title, Author, Category</p>
            <p><strong>Optional:</strong> Summary (will be AI-generated if missing)</p>
            <p><strong>Format:</strong> CSV file</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        st.markdown("""
        <div style="margin: 2rem 0;">
            <div class="feature-badge">
                <span class="feature-icon">🤖</span>
                AI Classification
            </div>
            <div class="feature-badge">
                <span class="feature-icon">📖</span>
                Smart Summaries
            </div>
            <div class="feature-badge">
                <span class="feature-icon">⚡</span>
                Batch Processing
            </div>
            <div class="feature-badge">
                <span class="feature-icon">📊</span>
                Analytics
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Validate required columns
            required_columns = ['Title', 'Author', 'Category']
            if not all(col in df.columns for col in required_columns):
                st.error(f"❌ Missing required columns. Found: {list(df.columns)}")
                st.stop()
            
            # Ensure Summary column exists
            if 'Summary' not in df.columns:
                df['Summary'] = ''
                st.session_state.df = df
                st.info("📝 Added Summary column to dataset")
            
            st.success(f"✅ Successfully loaded {len(df)} books!")
            
            # Analytics Dashboard - Clean Google style
            st.markdown('<h2 class="section-header">Dataset Analytics</h2>', unsafe_allow_html=True)
            
            # Metrics in a clean row
            col1, col2, col3, col4 = st.columns(4)
            
            total_books = len(df)
            missing_mask = (df['Category'].isna() |
                          (df['Category'].astype(str).str.strip() == '') |
                          (df['Category'].astype(str).str.lower() == 'none'))
            books_to_classify = missing_mask.sum()
            categorized_books = total_books - books_to_classify
            
            summary_mask = (df['Summary'].isna() |
                           (df['Summary'].astype(str).str.strip() == '') |
                           (df['Summary'].astype(str).str.lower() == 'null'))
            books_needing_summaries = summary_mask.sum()
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4>📚 {total_books}</h4>
                    <p>Total Books</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4>✅ {categorized_books}</h4>
                    <p>Categorized</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4>🎯 {books_to_classify}</h4>
                    <p>Need Classification</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4>📝 {books_needing_summaries}</h4>
                    <p>Need Summaries</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Processing Section - Centered like Google AI Studio
            if books_to_classify > 0:
                st.markdown('<div class="processing-section">', unsafe_allow_html=True)
                st.markdown('<h2 class="section-header">AI Processing</h2>', unsafe_allow_html=True)
                
                # Cost calculation
                classification_cost = books_to_classify * (0.035 if selected_model_name == 'GPT-4' else 0.005)
                summary_cost = books_needing_summaries * (0.025 if selected_model_name == 'GPT-4' else 0.004)
                total_cost = classification_cost + summary_cost
                
                # Processing info
                st.markdown(f"""
                <div class="warning-box">
                    <p>• Books to classify: <strong>{books_to_classify}</strong></p>
                    <p>• Summaries to generate: <strong>{books_needing_summaries}</strong></p>
                    <p>• Estimated cost: <strong>${total_cost:.2f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Large, centered action button
                if st.button("Start Processing",
                            type="primary",
                            disabled=not api_key,
                            use_container_width=True):
                    if not api_key:
                        st.error("Please configure your OpenAI API key above.")
                    else:
                        classify_books(api_key, batch_size, missing_mask)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box" style="max-width: 600px; margin: 2rem auto;">
                    <h4>🎉 Processing Complete!</h4>
                    <p>All books are already categorized. Ready for download!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Data preview in expandable section
            with st.expander("📊 Data Preview", expanded=False):
                preview_cols = ['Title', 'Author', 'Category']
                if 'Summary' in df.columns:
                    preview_cols.append('Summary')
                st.dataframe(df[preview_cols].head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.stop()
    
    # Results section
    if 'classified_df' in st.session_state and st.session_state.classified_df is not None:
        display_results()

def classify_books(api_key: str, batch_size: int, missing_mask):
    """Main classification function with progress tracking and summary generation"""
    try:
        client = openai.OpenAI(api_key=api_key)
        df = st.session_state.df.copy()
        
        books_to_classify = df[missing_mask].copy()
        total_books = len(books_to_classify)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        # Prepare books for classification
        books = []
        books_needing_summaries = []
        summary_indices = []
        
        for idx, (_, row) in enumerate(books_to_classify.iterrows()):
            book_data = {
                'Title': str(row['Title']).strip(),
                'Author': str(row['Author']).strip(),
                'Summary': str(row['Summary']) if pd.notna(row['Summary']) and str(row['Summary']).strip() and str(row['Summary']).lower() != 'null' else ''
            }
            books.append(book_data)
            
            # Check if book needs a summary
            if not book_data['Summary']:
                books_needing_summaries.append(book_data)
                summary_indices.append(idx)
        
        # Generate summaries for books that need them
        all_summaries = [''] * len(books)
        if books_needing_summaries:
            st.info(f"📝 Generating summaries for {len(books_needing_summaries)} books without summaries...")
            
            summary_batches = (len(books_needing_summaries) + batch_size - 1) // batch_size
            generated_summaries = []
            
            for i in range(0, len(books_needing_summaries), batch_size):
                batch = books_needing_summaries[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                status_text.text(f"📝 Generating summaries - batch {batch_num}/{summary_batches}")
                
                summaries = generate_summary_batch(batch, client, progress_bar, status_text)
                generated_summaries.extend(summaries)
                
                # Rate limiting
                if i + batch_size < len(books_needing_summaries):
                    time.sleep(2)
            
            # Update books with generated summaries
            for i, summary_idx in enumerate(summary_indices):
                if i < len(generated_summaries):
                    books[summary_idx]['Summary'] = generated_summaries[i]
                    all_summaries[summary_idx] = generated_summaries[i]
        
        # Process classification in batches
        all_categories = []
        total_batches = (total_books + batch_size - 1) // batch_size
        
        for i in range(0, total_books, batch_size):
            batch = books[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            status_text.text(f"🔄 Classifying batch {batch_num}/{total_batches} ({len(batch)} books)")
            
            categories = classify_book_batch(batch, client, progress_bar, status_text)
            all_categories.extend(categories)
            
            # Update progress (50% for summaries, 50% for classification)
            base_progress = 0.5 if books_needing_summaries else 0
            classification_progress = (min((i + batch_size) / total_books, 1.0)) * 0.5
            total_progress = base_progress + classification_progress
            progress_bar.progress(total_progress)
            
            # Show current batch results
            with results_container.container():
                st.subheader(f"📝 Batch {batch_num} Results:")
                batch_results = []
                for book, category in zip(batch, categories):
                    batch_results.append({
                        'Title': book['Title'][:50] + '...' if len(book['Title']) > 50 else book['Title'],
                        'Author': book['Author'],
                        'Category': category,
                        'Summary': book['Summary'][:100] + '...' if len(book['Summary']) > 100 else book['Summary']
                    })
                st.dataframe(pd.DataFrame(batch_results), hide_index=True)
            
            # Rate limiting
            if i + batch_size < total_books:
                time.sleep(2)
        
        # Update dataframe with results
        df.loc[missing_mask, 'Category'] = all_categories[:len(books_to_classify)]
        
        # Update summaries for books that didn't have them
        if books_needing_summaries:
            summary_mask = df[missing_mask].index
            for i, summary_idx in enumerate(summary_indices):
                if i < len(generated_summaries):
                    original_idx = summary_mask[summary_idx]
                    df.loc[original_idx, 'Summary'] = generated_summaries[i]
        
        st.session_state.classified_df = df
        
        progress_bar.progress(1.0)
        status_text.text("✅ Classification and summary generation completed!")
        
        summary_count = len(books_needing_summaries)
        success_msg = f"🎉 Successfully classified {total_books} books!"
        if summary_count > 0:
            success_msg += f" Generated {summary_count} new summaries!"
        st.success(success_msg)
        
    except Exception as e:
        st.error(f"❌ Classification failed: {e}")

def display_results():
    """Display classification results and analysis"""
    st.markdown('<h2 class="section-header">📊 Results & Analysis</h2>', unsafe_allow_html=True)
    
    df = st.session_state.classified_df
    
    # Safety check to ensure df is valid
    if df is None or len(df) == 0:
        st.error("No data available for display.")
        return
    
    # Professional completion message
    st.markdown(f"""
    <div class="success-box">
        <h4>Processing Complete!</h4>
        <p>Successfully processed {len(df)} books with AI-powered classification and summary generation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "📚 Books by Category", "💾 Download"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            st.subheader("Category Distribution")
            # Handle None/NaN values in Category column
            category_counts = df['Category'].fillna('Uncategorized').value_counts()
            st.bar_chart(category_counts)
        
        with col2:
            # Top categories
            st.subheader("Top 10 Categories")
            top_categories = category_counts.head(10)
            for category, count in top_categories.items():
                percentage = (count / len(df)) * 100
                st.metric(
                    label=category,
                    value=f"{count} books",
                    delta=f"{percentage:.1f}%"
                )
    
    with tab2:
        # Filter by category
        st.subheader("Filter Books by Category")
        
        # Get unique categories, handling None/NaN values
        unique_categories = df['Category'].fillna('Uncategorized').unique()
        sorted_categories = sorted([cat for cat in unique_categories if pd.notna(cat)])
        
        selected_category = st.selectbox(
            "Select a category:",
            options=['All'] + sorted_categories
        )
        
        if selected_category == 'All':
            filtered_df = df
        elif selected_category == 'Uncategorized':
            filtered_df = df[df['Category'].isna() | (df['Category'].str.strip() == '')]
        else:
            filtered_df = df[df['Category'] == selected_category]
        
        st.write(f"📖 Showing {len(filtered_df)} books")
        
        # Display books (excluding Status column completely)
        available_cols = ['Title', 'Author', 'Category']
        if 'Summary' in filtered_df.columns:
            available_cols.append('Summary')
        
        # Only use columns that actually exist in the dataframe
        display_cols = [col for col in available_cols if col in filtered_df.columns]
        display_df = filtered_df[display_cols].copy()
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    with tab3:
        # Download options
        st.subheader("Download Results")
        
        st.info("💡 Right-click the download buttons and select 'Save link as...' if direct download doesn't work.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download full results
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Complete Dataset",
                data=csv_data,
                file_name="reading_list_classified.csv",
                mime="text/csv",
                help="Download the complete dataset with all classifications",
                use_container_width=True
            )
        
        with col2:
            # Download only newly classified books
            if ('df' in st.session_state and st.session_state.df is not None and 
                len(st.session_state.df) > 0):
                original_df = st.session_state.df
                missing_mask = (original_df['Category'].isna() | 
                               (original_df['Category'].astype(str).str.strip() == '') |
                               (original_df['Category'].astype(str).str.lower() == 'none'))
                newly_classified = df[missing_mask]
                
                if len(newly_classified) > 0:
                    csv_buffer2 = io.StringIO()
                    newly_classified.to_csv(csv_buffer2, index=False)
                    csv_data2 = csv_buffer2.getvalue()
                    
                    st.download_button(
                        label="📥 Download Only New Classifications",
                        data=csv_data2,
                        file_name="newly_classified_books.csv",
                        mime="text/csv",
                        help="Download only the books that were classified in this session",
                        use_container_width=True
                    )
        
        # Add summary at the bottom
        # st.markdown("---")
        # st.success(f"✅ Classification complete! {len(df)} total books processed.")

if __name__ == "__main__":
    # Initialize session state with proper defaults
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'classified_df' not in st.session_state:
        st.session_state.classified_df = None
    
    # Clear any invalid session state on app restart
    if st.session_state.classified_df is not None and not hasattr(st.session_state.classified_df, 'columns'):
        st.session_state.classified_df = None
    
    main()

# Instructions for running the app
# """
# SETUP INSTRUCTIONS:

# 1. Install required packages:
#    pip install streamlit openai pandas python-dotenv

# 2. Create a .env file in the same directory:
#    OPENAI_API_KEY=your-actual-api-key-here

# 3. Save this code as 'book_classifier_app.py'

# 4. Run the Streamlit app:
#    streamlit run book_classifier_app.py

# 5. Open your browser to the URL shown (usually http://localhost:8501)

# 6. Use the app:
#    - The app will automatically load your API key from .env
#    - Upload your CSV file (requires: Title, Author, Category columns)
#    - Configure settings (model, batch size)
#    - Click "Start Classification"
#    - View results and download the classified dataset

# ENVIRONMENT FILE SETUP:
# Create a file named '.env' in your project directory with:
# OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# CSV FILE REQUIREMENTS:
# - Required columns: Title, Author, Category
# - Optional columns: Summary (helps with classification accuracy)
# - Status column will be ignored if present

# FEATURES:
# - Automatic API key loading from .env file
# - Real-time progress tracking
# - Batch processing with customizable batch sizes
# - Cost estimation
# - Interactive data visualization
# - Category filtering and analysis
# - Download options for results
# - Responsive design with custom styling
# - Secure API key handling
# """