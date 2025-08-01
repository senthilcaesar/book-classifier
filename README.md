# 📚 Book Classification & Summary Tool

A powerful AI-powered web application built with Streamlit that automatically classifies books and generates intelligent summaries using OpenAI's GPT models. Features a clean, Google AI Studio-inspired interface for an intuitive user experience.

## ✨ Features

- **🤖 AI-Powered Classification**: Automatically categorizes books into 25+ predefined categories
- **📖 Smart Summary Generation**: Creates concise, informative summaries for books without existing descriptions
- **⚡ Batch Processing**: Efficiently processes multiple books simultaneously with customizable batch sizes
- **📊 Interactive Analytics**: Real-time dashboard with metrics, charts, and category distribution
- **💰 Cost Estimation**: Transparent pricing estimates before processing
- **🎨 Modern UI**: Clean, Google AI Studio-inspired design with responsive layout
- **📥 Export Options**: Download complete datasets or only newly processed books
- **🔒 Secure**: Environment-based API key management

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd readinglist_analyzer_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501` to access the application.

## 📋 CSV File Requirements

### Required Columns
- **Title**: Book title
- **Author**: Book author
- **Category**: Book category (can be empty for classification)

### Optional Columns
- **Summary**: Book summary (will be AI-generated if missing)

### Example CSV Format
```csv
Title,Author,Category,Summary
"The Great Gatsby","F. Scott Fitzgerald","",""
"Sapiens","Yuval Noah Harari","History","A brief history of humankind..."
"Clean Code","Robert C. Martin","Technology",""
```

## 🎯 Supported Categories

The tool classifies books into these categories:

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

## 💡 How It Works

1. **Upload**: Upload your CSV file with book data
2. **Analyze**: The tool analyzes your dataset and shows metrics
3. **Process**: AI generates summaries for books without them, then classifies all books
4. **Review**: Explore results with interactive charts and filters
5. **Download**: Export your enhanced dataset with classifications and summaries

## 🔧 Configuration Options

### Model Selection
- **GPT-4**: Higher accuracy, more expensive (~$0.03-0.04 per book)
- **GPT-3.5 Turbo**: Good accuracy, cost-effective (~$0.004-0.006 per book)

### Batch Processing
- Adjustable batch size (1-15 books per API call)
- Automatic rate limiting to respect API limits
- Progress tracking with real-time updates

## 📊 Analytics Dashboard

The application provides comprehensive analytics:

- **Dataset Overview**: Total books, categorized vs. uncategorized counts
- **Category Distribution**: Visual charts showing book distribution across categories
- **Processing Metrics**: Real-time progress tracking and cost estimation
- **Filtering**: Browse books by category with search and filter options

## 🛡️ Security & Privacy

- **API Key Protection**: Keys stored in environment variables, never in code
- **Local Processing**: All data processing happens locally on your machine
- **No Data Storage**: The application doesn't store your book data permanently

## 🎨 UI Design

The application features a modern, clean interface inspired by Google AI Studio:

- **Responsive Design**: Works on desktop and mobile devices
- **Google Sans Typography**: Clean, readable fonts
- **Material Design Elements**: Cards, shadows, and smooth transitions
- **Intuitive Navigation**: Tabbed interface for easy exploration
- **Accessibility**: Proper contrast ratios and keyboard navigation

## 📁 Project Structure

```
readinglist_analyzer_app/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🔍 Troubleshooting

### Common Issues

**"No API key found"**
- Ensure your `.env` file exists and contains `OPENAI_API_KEY=your-key`
- Check that the `.env` file is in the same directory as `app.py`

**"Missing required columns"**
- Verify your CSV has `Title`, `Author`, and `Category` columns
- Check for typos in column names (case-sensitive)

**"Rate limit exceeded"**
- The app includes automatic rate limiting
- If you hit limits, wait a few minutes before retrying
- Consider using smaller batch sizes

**Download not working**
- Try right-clicking the download button and selecting "Save link as..."
- Check your browser's download settings

## 💰 Cost Estimation

### GPT-4 Pricing (Approximate)
- Classification: $0.03-0.04 per book
- Summary Generation: $0.02-0.03 per book
- **Total**: ~$0.05-0.07 per book

### GPT-3.5 Turbo Pricing (Approximate)
- Classification: $0.004-0.006 per book
- Summary Generation: $0.003-0.005 per book
- **Total**: ~$0.007-0.011 per book

*Actual costs may vary based on book title/author length and API pricing changes.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the [OpenAI API documentation](https://platform.openai.com/docs)
3. Open an issue in this repository

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI GPT models](https://openai.com/)
- UI inspired by [Google AI Studio](https://aistudio.google.com/)
- Icons from various emoji sets

---

**Happy Reading! 📚✨**