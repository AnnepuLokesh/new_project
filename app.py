
from flask import Flask, render_template, request, session,redirect,url_for
from flask_session import Session
import pickle 
import requests
import secrets

from langchain_fireworks import Fireworks
from PyPDF2 import PdfReader
import re
import genai
app = Flask(__name__)

app.config['SESSION_TYPE'] = 'filesystem'  # or 'redis' if you want to use Redis
app.config['SESSION_PERMANENT'] = False
app.secret_key = secrets.token_hex(16)
Session(app) 
import google.generativeai as genai

api_key = 'AIzaSyDVJ60np72_az-MeJyWBiHjk5zTsq9o47c'
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config
)

# API key
api_key = "fw_3ZJnFqsx85o1zJTNDZQuuMnb"

# Initialize a Fireworks model using the provided API key
llm = Fireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    base_url="https://api.fireworks.ai/inference/v1/completions",
    max_tokens=1500,
    fireworks_api_key=api_key
)
# Load models with raw string paths
rf_classifier_categorization = pickle.load(open(r"D:\resume\models\rf_classifier_categorization.pkl", "rb"))
tfidf_vectorizer_categorization = pickle.load(open(r"D:\resume\models\tfidf_vectorizer_categorization.pkl", "rb"))
rf_classifier_job_recommendation = pickle.load(open(r"models/rf_classifier_job_recommendation.pkl", "rb"))
tfidf_vectorizer_job_recommendation = pickle.load(open(r"models/tfidf_vectorizer_job_recommendation.pkl", "rb"))

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_contact_number_from_resume(text):
    contact_number = None
    pattern = r"(?:Phone|Contact)?\s?(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{3,4}"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()
    return contact_number

def extract_email_from_resume(text):
    email = None
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()
    return email

def extract_skills_from_resume(text):
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau',
        'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib',
        'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees',
        'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN',
        'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL',
        'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker',
        'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption',
        'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration',
        'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite',
        'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research',
        'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
        'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)',
        'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting',
        'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing',
        'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing',
        'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation',
        'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento',
        'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement',
        'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL',
        'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics',
        'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL',
        'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL',
        'Socket.io', 'WebSockets'
                     'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#',
        'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development',
        'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI',
        'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda',
        'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration',
        'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment',
        'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation',
        'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling',
        'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya',
        'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing',
        'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development',
        'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)',
        'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection',
        'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection',
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)',
        'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)',
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)']

    skills = []
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)
    return skills

def extract_projects_from_resume(text):
    projects = []
    project_keywords = [
        "project", "developed", "created", "designed", "implemented", "built", "managed", "collaborated", "led", 
        "research", "thesis"
    ]
    lines = text.split('\n')
    for line in lines:
        for keyword in project_keywords:
            if keyword in line.lower():
                projects.append(line.strip())
                break
    return projects

def extract_education_from_resume(text):
    education_keywords = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering', 'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science', 'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management', 'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design', 'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development'
        'Library Science', 'Information Science', 'Computer Engineering', 'Software Development', 'Cybersecurity', 'Information Security',
        'Network Engineering', 'Data Science', 'Data Analytics', 'Business Analytics', 'Operations Research', 'Decision Sciences',
        'Human-Computer Interaction', 'User Experience Design', 'User Interface Design', 'Digital Marketing', 'Content Strategy',
        'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media', 'Web Development',
        'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality', 'Blockchain Technology', 'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management', 'Fire Science',
        'Environmental Science', 'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing', 'Geoinformatics',
        'Cartography', 'GIS (Geographic Information Systems)', 'Environmental Management', 'Sustainability Studies', 'Renewable Energy',
        'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology']
    education = []
    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())
    return education

def extract_name_from_resume(text):
    name = None
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    if match:
        name = match.group()
    return name

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job


def send_to_google_bard(prompt, extracted_skills, extracted_projects):
    prompt = f"""
Analyze the provided resume details and generate a minimum of 30 to 40 high-quality interview questions tailored to assess the candidate's technical expertise and project-based knowledge.

**Extracted Skills**: {', '.join(extracted_skills)}
**Extracted Projects**: {', '.join(extracted_projects)}

### Guidelines for Question Generation:
1. For each **skill** listed in the "Extracted Skills" section, generate **5 questions**:
   - **2 Easy Questions**: These should test fundamental theoretical knowledge.
   - **2 Medium Questions**: These should test understanding of more advanced theoretical concepts and problem-solving abilities.
   - **1 Hard Question**: This should challenge the candidate's expertise by exploring advanced theoretical or real-world application concepts.

2. For each **project** listed in the "Extracted Projects" section, generate **5 project-specific questions**:
   - Focus on the candidate's role and contributions, design choices, and challenges faced during the project.
   - Assess their ability to apply their theoretical knowledge in real-world scenarios.

3. Exclude any coding-related questions entirely. Ensure that all questions are theoretical or concept-based.

### Formatting Instructions:
1. Number each question sequentially (e.g., 1), 2), 3), etc.).
2. For each skill or project, clearly categorize and present the questions under the corresponding section.
3. **Do not include answers** for the project-specific questions.
4. Provide concise and clear **answers** for the questions generated for the skills. Ensure answers are theoretical and avoid code snippets.


### Example Input:
**Extracted Skills**: Python, SQL, Flask  
**Extracted Projects**: Resume Parser Application, Machine Learning Model for Crop Recommendation

### Example Output:

#### Questions:
Skill: Python
1) What is the difference between a list and a tuple in Python?  
2) Explain the concept of Python's Global Interpreter Lock (GIL).  
3) What are the different data types in Python, and how are they classified?  
4) What is the significance of Python being an interpreted language?  
5) Explain how Python manages memory.

Skill: SQL
1) What is a primary key in SQL, and why is it important?  
2) Describe the ACID properties of a database transaction.  
3) What are the different types of SQL joins, and how are they used?  
4) What are SQL indexes, and how do they improve query performance?  
5) How would you optimize a SQL query that is performing poorly?

Skill: Flask
1) What is the purpose of Flask in web development?  
2) Explain how Flask handles HTTP requests and responses.  
3) What is the difference between Flask and other web frameworks?  
4) How do you manage sessions in Flask?  
5) What are Flask blueprints, and how are they used?

Project: Resume Parser Application
1) What role did you play in the development of the Resume Parser Application, and what were your main contributions?  
2) What challenges did you face while designing the parser for different resume formats?  
3) Explain the choice of algorithms used to extract relevant information from resumes.  
4) How did you handle the variability in resume layouts, and how did you ensure data accuracy?  
5) What optimizations did you implement to improve the performance of the Resume Parser?

Project: Machine Learning Model for Crop Recommendation
1) What was the primary objective of the Crop Recommendation system, and what data was used for training?  
2) How did you preprocess the data before feeding it into the model, and why was this preprocessing necessary?  
3) What machine learning algorithms did you consider for the model, and why did you choose the final algorithm?  
4) Explain how you evaluated the model's performance. What metrics did you use?  
5) What real-world challenges did you encounter while deploying the Crop Recommendation system?

#### Answers:
Skill: Python
1) Lists are mutable, meaning their elements can be modified, whereas tuples are immutable and cannot be changed after creation.
2) The Global Interpreter Lock (GIL) in Python allows only one thread to execute at a time, even on multi-core processors.
3) Python data types include numeric (int, float), sequence (list, tuple, string), mapping (dictionary), and more, categorized by their mutability and purpose.
4) Python being an interpreted language means code is executed line by line without prior compilation.
5) Python uses a private heap to manage memory and employs garbage collection to automatically reclaim unused memory.

Skill: SQL
1) A primary key uniquely identifies each record in a table and ensures data integrity.
2) ACID properties (Atomicity, Consistency, Isolation, Durability) ensure reliable transactions in a database.
3) SQL joins (INNER, LEFT, RIGHT, FULL) allow combining rows from two or more tables based on a related column.

Skill: Flask
1) Flask is a microframework used for building web applications with simplicity and flexibility.
2) Flask handles HTTP requests by routing them to appropriate functions and generating responses back to the client.
3) Flask is lightweight and modular, making it different from full-stack frameworks like Django.

---
The output now ensures no coding-related questions and provides theoretical answers for the skills only.
"""

    try:
        response = model.generate_content(prompt)
        generated_text = response.text
        print("Generated Text:", generated_text)
        
        # Split questions and answers
        questions, answers = "", ""
        if "#### Questions:" in generated_text and "#### Answers:" in generated_text:
            parts = generated_text.split("#### Answers:")
            questions = parts[0].replace("#### Questions:", "").strip()
            answers = parts[1].strip()

        # Process questions and answers into a dictionary format
        question_list = [line.split(")", 1)[-1].strip() for line in questions.split("\n") if ")" in line]
        answer_list = [line.split(")", 1)[-1].strip() for line in answers.split("\n") if ")" in line]
        qa_pairs = [
            {"question": question.strip(), "answer": answer.strip()}
            for question, answer in zip(question_list, answer_list)
            if question and answer
        ]

        return {"qa_pairs": qa_pairs,"questions":questions}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"qa_pairs": []}
skills_list=[]
@app.route("/pred", methods=['POST'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename

        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template('resume.html', pred="Invalid File Format")

        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)
        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)
        extracted_skills = extract_skills_from_resume(text)
        skills_list=extracted_skills
        extracted_education = extract_education_from_resume(text)
        name = extract_name_from_resume(text)
        extracted_projects = extract_projects_from_resume(text)
        prompt = f"""
Analyze the following resume details and generate high-quality interview questions tailored to assess the candidate's technical and project-based expertise.

Skills: {extracted_skills}

Projects: {extracted_projects}

Question Scope:
- Conceptual questions to test understanding of technologies used.
- Problem-solving and coding tasks based on key skills like Python, Java, and SQL.
- System design and database schema questions related to projects.
- Case studies or scenarios where candidates explain decision-making in their projects.
- Debugging and optimization challenges.

Output Requirements:
- Categorize questions by skill or project.
- Provide a mix of theoretical and practical questions.
- Include sample solutions or evaluation points for each question.
- Focus on creating a diverse and challenging set of questions that assess the candidate's depth of knowledge, problem-solving ability, and project implementation experience.
"""
        bard_response = send_to_google_bard(prompt, extracted_skills, extracted_projects)
        print("Bard Response:", bard_response)  # Debug the full response

        # Extract questions from qa_pairs
        qa_pairs = bard_response.get("qa_pairs", [])
        questions = [qa.get("question", "").strip() for qa in qa_pairs if "question" in qa]
        print("Extracted Questions:", questions)

# Store qa_pairs in session (optional, if needed later)
        session['qa_pairs'] = qa_pairs

        
        qa_pairs = session.get('qa_pairs', [])
        
        # Store qa_pairs in the session
        session['qa_pairs'] = bard_response.get("qa_pairs", [])
        # session['questions'] = generated_text.get("questions",[])

        return render_template('resume.html', predicted_category=predicted_category,
                               recommended_job=recommended_job, phone=phone,
                               name=name, email=email, extracted_skills=extracted_skills,
                               extracted_education=extracted_education,
                               extracted_projects=extracted_projects, 
                               qa_pairs=session['qa_pairs'],questions = questions)

    else:
        return render_template("resume.html", message="No resume file uploaded.")

@app.route('/answers')
def answers():
    # Retrieve qa_pairs from the session
    qa_pairs = session.get('qa_pairs', [])

    if qa_pairs:
        return render_template('answers.html', qa_pairs=qa_pairs)
    else:
        return render_template('answers.html', message="No questions and answers available.")



@app.route('/')
def resume():
    
    return render_template('resume.html')


@app.route('/write_answers')
def write_answers():
    # Retrieve qa_pairs from the session
    qa_pairs = session.get('qa_pairs', [])

    # Extract only the questions from qa_pairs
    questions = [qa.get('question', '').strip() for qa in qa_pairs if 'question' in qa]
    
    print("questions:", questions)

    return render_template('write_answers.html', questions=questions)

@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    submitted_answers = request.form.getlist('answers[]')
    qa_pairs = session.get('qa_pairs', [])
    expected_answers = {qa['question']: qa['answer'] for qa in qa_pairs}
    question_answer_mapping = [
        {
            "question": qa.get('question', '').strip(),
            "answer": submitted_answers[i].strip() if i < len(submitted_answers) and submitted_answers[i].strip() else "Not answered"
        }
        for i, qa in enumerate(qa_pairs)
    ]
    
    feedback = send_to_fireworks_for_feedback(expected_answers, question_answer_mapping,skills_list)
    print("feedback:", feedback)  # Debug the full response
    session['submitted_answers'] = question_answer_mapping
    session['feedback'] = feedback

    return render_template('answers_summary.html', question_answer_mapping=question_answer_mapping)

import json  # Add this import at the top of your script





def send_to_fireworks_for_feedback(expected_answers, submitted_answers, extracted_skills):
    # Calculate overall accuracy
    total_questions = len(expected_answers)
    correct_answers = sum(
        1 for qa in submitted_answers if expected_answers.get(qa['question'], '').strip().lower() == qa['answer'].lower()
    )
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0

    # Ensure at least some answers are provided
    if not submitted_answers:
        return "No answers submitted for evaluation."

    # Create a prompt for feedback generation
    prompt = f"""
    Analyze the following set of questions and answers submitted by a student for a technical interview preparation exercise. The questions were generated based on the skills extracted from the student's resume. Provide a comprehensive evaluation focused on overall performance by skill area. The evaluation should include:

    1. **Overall Performance**: Summarize the student’s performance across all skills, highlighting their readiness for a technical interview.
    2. **Accuracy**: The overall accuracy of the submitted answers is {accuracy:.2f}%.
    3. **Strengths**: Identify specific skills where the student demonstrated strong proficiency based on their answers.
    4. **Areas for Improvement**: Highlight specific skills where the student needs improvement, providing actionable recommendations for further study and practice.
    5. **Improvement Suggestions**: Offer specific guidance on how the student can enhance their understanding and skills in the areas identified for improvement.
    6. **Resources for Learning**: Suggest resources such as articles, tutorials, and courses that can help the student improve in weaker areas.

    ### Input Data:
    - **Extracted Skills from Resume**: {extracted_skills}
    - **Generated Questions and Expected Answers**: {expected_answers}
    - **Student's Submitted Answers**: {submitted_answers}
    - **Question Topics**: Each skill area covered by the questions.

    ### Desired Output:
    - **Overall Performance**: Summary of the student's overall performance across the extracted skills.
    - **Strengths**: List of skills where the student performed well.
    - **Areas for Improvement**: List of skills where the student needs improvement, with specific suggestions.
    - **Resources for Learning**: Recommended resources for further study.
    **Do not include any programming code or technical jargon in the response. Focus purely on performance evaluation, strengths, and improvement areas.**

    """

    try:
        response = llm.invoke(prompt)

        # Validate and handle response
        if isinstance(response, str) and "error" not in response.lower():
            return response

        if hasattr(response, 'text') and "error" not in response.text.lower():
            return response.text

        # Handle unexpected formats or errors
        return "An unexpected response was received from the LLM."

    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred while generating feedback."


import json


@app.route('/feedback')
def feedback():
    feedback_data = session.get('feedback', {})
    
    if isinstance(feedback_data, str):
        try:
            feedback_data = json.loads(feedback_data)
        except json.JSONDecodeError:
            feedback_data = {
                "overall_performance": feedback_data,
                "strengths": "",
                "areas_for_improvement": "",
                "resources_for_learning": ""
            }

    feedback_sections = {
        "overall_performance": feedback_data.get("overall_performance", ""),
        "strengths": feedback_data.get("strengths", ""),
        "areas_for_improvement": feedback_data.get("areas_for_improvement", ""),
        "resources_for_learning": feedback_data.get("resources_for_learning", "")
    }

    # Print feedback_sections for debugging
    print("Feedback Sections:", feedback_sections)

    # Check for missing or empty sections
    for section, value in feedback_sections.items():
        if not value:
            print(f"Missing or empty feedback section: {section}")
    
    # Format feedback as HTML
    resources_html = "<ul>" + "".join(
        f"<li>{resource.strip()}</li>" 
        for resource in feedback_sections['resources_for_learning'].split('\n') 
        if resource.strip()
    ) + "</ul>"

    feedback_html = f"""
    <div>
        <h2>Overall Performance</h2>
        <p>{feedback_sections['overall_performance']}</p>
        
        <h2>Strengths</h2>
        <p>{feedback_sections['strengths']}</p>
        
        <h2>Areas for Improvement</h2>
        <p>{feedback_sections['areas_for_improvement']}</p>
        
        <h2>Resources for Learning</h2>
        <div class="resources">
            {resources_html}
        </div>
    </div>
    """

    return render_template('feedback.html', feedback=feedback_html)

    
if __name__ == "__main__":
    app.run(debug=True)