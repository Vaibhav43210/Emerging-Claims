# My Streamlit App

This project is a Streamlit application designed for analyzing claims data using Natural Language Processing (NLP). The application allows users to upload claims data, perform NLP classification, visualize results, and gain insights into emerging risks.

## Project Structure

```
my-streamlit-app
├── app.py                  # Main Streamlit application code
├── requirements.txt        # Python dependencies
├── .streamlit
│   └── config.toml        # Streamlit configuration settings
├── Dockerfile              # Docker image configuration
├── Procfile                # Command to run the application on platforms like Heroku
├── .github
│   └── workflows
│       └── deploy.yml      # GitHub Actions workflow for deployment
├── .gitignore              # Files and directories to ignore by Git
└── README.md               # Project documentation
```

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/my-streamlit-app.git
   cd my-streamlit-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To start the Streamlit application, run the following command:
```
streamlit run app.py
```
This will open the application in your default web browser.

## Deployment

The application can be deployed using Docker or on platforms like Heroku. 

### Docker Deployment

To build and run the Docker container, use the following commands:
```
docker build -t my-streamlit-app .
docker run -p 8501:8501 my-streamlit-app
```
Access the application at `http://localhost:8501`.

### Heroku Deployment

To deploy on Heroku, ensure you have the Heroku CLI installed and run:
```
heroku create
git push heroku main
```

## Usage

1. Upload your claims Excel file in the "Data Load" tab.
2. Click on "Run NLP Classification on First 50 Claims" to classify the claims.
3. Explore the results in the subsequent tabs for aggregation, visualization, and insights.

## License

This project is licensed under the MIT License. See the LICENSE file for details.