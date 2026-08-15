Lets first know, what is data pre-processing and why do we need to learn about it in the initial stages of learning machine learning. Also include what all will we learn about this concept and its depth with codes and all. 

## Data Preprocessing
It's a process in which we care only about data, see if data is good to use, is there any modification needed before passing it forward, do we need to add anything, all these thing get covered in this concept.

### Why learning now?
As we know machine learning is all about getting a model trained on data, to make it useful in real world. So to get a good model to work with, we need data to be clean and useful mainly, which we why we learn this concept here initially. We need to have basic understanding about data to work with model later in time.

### What all concepts do we cover in this!
1) Missing data handling:
- Types: MCAR, MAR, MNAR (only practical understanding)
- When to drop vs impute vs model-based handling
- Advanced imputation:
    - KNN imputation
    - Iterative/MICE
    - Model-based (LightGBM/XGBoost handling missing)
- Handling missing in time-series & sequential data
- Feature engineering with missingness (missing as signal)
- Real-world pitfalls

2) Data Cleaning (Beyond basics):
- Handling inconsistent formats (dates, categories, text)
- Outlier detection (IQR, Z-score, Isolation Forest)
- De-duplication strategies
- Data leakage issues (very important in real systems)

3) Feature Engineering (Core Skill)
- Encoding (target encoding, embeddings—not just one-hot)
- Scaling (when it matters, when it doesn’t)
- Feature interactions
- Domain-driven feature creation
- Handling high-cardinality features
4) Data Transformation Pipelines
- sklearn pipelines & column transformers
- Reproducible preprocessing (production mindset)
- Handling train/test mismatch
- Online vs offline preprocessing
5) Handling Imbalanced Data
- SMOTE, ADASYN
- Class weights vs resampling
- When NOT to balance data
6) Text Preprocessing (for NLP → LLM path)
- Tokenization (classical vs modern)
- Cleaning vs over-cleaning (important nuance)
- Embeddings vs traditional preprocessing
- Preparing data for transformers
7) Large-Scale / Real-World Data Handling
- Batch processing vs streaming
- Memory optimization (very practical)
- Using tools like pandas vs polars vs spark (when to use what)
8) Data Validation & Quality Checks
- Schema validation
- Drift detection (train vs production)
- Tools (Great Expectations – brief intro)

We'll cover all these with if there are anything missing. 

