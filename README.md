# Disaster Responder and Climate Impact Predictor (DRCIP)

## Project Description
The Disaster Responder and Climate Impact Predictor (DRCIP) is an advanced application designed to predict the impacts of climate change and respond to natural disasters using satellite imagery and machine learning. The application leverages NASA Earth data, historical disaster data, and advanced neural networks to provide valuable insights and real-time data to help mitigate the effects of natural disasters and climate change.

## Features
- **Climate Change Impact Prediction:** Uses historical and real-time data to predict the impacts of climate change on specific regions.
- **Natural Disaster Detection and Response:** Analyzes satellite images to detect and classify natural disasters such as floods and wildfires.
- **Interactive Visualizations:** Provides an intuitive web interface with interactive maps to visualize disaster impacts and climate predictions.
- **API for External Requests:** Exposes functionalities via a REST API to allow external users and systems to make requests.

## Technologies Used
### Data Processing and Storage
- **NASA Earth Data** (MODIS, Landsat)
- **AWS S3 or Google Cloud Storage** for storing large datasets
- **PostgreSQL with PostGIS** for geospatial data storage and querying

### Machine Learning and Analysis
- **Python** for data processing and machine learning
- **TensorFlow/Keras or PyTorch** for developing and training neural networks
- **scikit-learn** for additional machine learning models and data preprocessing
- **OpenCV** for image processing and computer vision tasks
- **GDAL** for handling geospatial data
- **Hugging Face Transformers** for NLP tasks

### Model Deployment
- **Docker** for containerizing the application
- **Kubernetes** for orchestrating and managing containerized applications
- **Flask or FastAPI** for building the REST API

### Web Development
- **React or Angular** for building the user interface
- **Mapbox or Leaflet** for interactive map integration

### Infrastructure and DevOps
- **CI/CD Tools** like **GitHub Actions or Jenkins**
- **Cloud Platforms** like **AWS, Google Cloud, or Azure**
- **Cloud Functions or AWS Lambda** for serverless functions

### Monitoring and Logging
- **Prometheus and Grafana** for monitoring and visualizing metrics
- **ELK Stack (Elasticsearch, Logstash, Kibana)** for logging

### Security
- **JWT (JSON Web Tokens)** for securing API endpoints
- **OAuth 2.0** for user authentication and authorization

## Project Workflow
1. **Data Collection and Preprocessing**
   - Gather and preprocess satellite imagery and climate data.

2. **Model Development**
   - Train neural networks to predict climate change impacts.
   - Develop computer vision models to detect and classify natural disasters.

3. **API Development**
   - Build REST API to expose prediction and analysis functionalities.

4. **Frontend Development**
   - Create an interactive web interface with map visualizations.

5. **Deployment**
   - Containerize the application using Docker.
   - Deploy on a cloud platform using Kubernetes.

6. **Monitoring and Maintenance**
   - Set up monitoring and logging.
   - Regularly update models with new data.

## Getting Started
### Prerequisites
- Python 3.8+
- Docker
- Node.js (for frontend development)
- PostgreSQL with PostGIS

### Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/WillianVMR/disaster-responder-satellite-images.git
   cd DRCIP
   ```
2. **Backend Setup:**
   * Create a virtual enviroment:
     ```sh
     python -m venv venv
     source venv/bin/activate   # On Windows use `venv\Scripts\activate`
     ```
   * Install the required Python packages:
     ```sh
     pip install -r requirements.txt
     ```
3. **Frontend setup:**
   * Navigate to the frontend directory and install dependencies:
     ```sh
     cd frontend
     npm install
     ```
4. **Database setup:**
   * Set up PostgreSQL with PostGIS and configure the connection settings in the `.env file.
5. **Run the Application:**
   * Start the backend server:
     ```sh
     python app.py
     ```
   * Start the frontend development server:
     ```sh
     npm start
     ```

### Usage
* Access the web application at `http://localhost:3000`.
* Use the REST API endpoints to interact with the backend services.

### Contributing
We welcome contributions! Please read our CONTRIBUTING.md for guidelines on how to contribute to this project.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Contact
For any questions or inquiries, please contact [wmourawillianribeiro@gmail.com].
