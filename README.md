AI-Powered Resume Screener (NLP)

This project develops an intelligent resume screening pipeline using Natural Language Processing (NLP) techniques and semantic embeddings. The primary objective is to build a system capable of ranking candidate resumes against a given job description, going beyond simple keyword matching to capture the true semantic meaning of candidate qualifications. By addressing the challenges of unstructured resume formats and recruiter efficiency, this project demonstrates both strong technical depth and practical applicability.

The pipeline begins with text preprocessing, where resumes in multiple formats (PDF, DOCX, TXT) are parsed into clean, structured text. Standard NLP methods such as tokenization, lemmatization, and stopword removal are applied to normalize the input and remove noise. Both resumes and job descriptions are then encoded into dense vector embeddings using pretrained Sentence Transformer models (e.g., all-MiniLM-L6-v2), which capture contextual meaning at the sentence and document level.

To evaluate candidate-job fit, the system calculates semantic similarity between embeddings using cosine similarity. Since many organizations also rely on explicit keyword requirements, the final score is computed as a hybrid measure that combines semantic similarity with keyword coverage, ensuring both conceptual alignment and requirement satisfaction.

Multiple embedding models (such as MiniLM and MPNet variants) are compared to assess the trade-offs between speed, interpretability, and predictive quality. The system is deployed through a web interface (Gradio), enabling users to paste a job description, upload multiple resumes, and instantly receive a ranked list of candidates. Results can also be exported as CSV files for integration into recruitment workflows.

The project is designed as an end-to-end solution, incorporating resume ingestion, preprocessing, embedding, similarity computation, ranking, and deployment. Its modular design ensures that each step—from text cleaning to model selection—can be easily reproduced, extended, or integrated with existing Applicant Tracking Systems (ATS).
