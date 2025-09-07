# 🎬 MovieVec - A Movie Semantic Search Engine

A high-performance semantic search engine for movies using state-of-the-art embedding models and FAISS indexing. Search through hundreds of thousands of movies using natural language queries with advanced cross-encoder re-ranking for superior accuracy.

## ✨ Features

- **Semantic Search**: Natural language queries like "crime movie with Al Pacino and Robert De Niro"
- **Cross-Encoder Re-ranking**: Advanced two-stage retrieval for maximum accuracy
- **FAISS Integration**: Fast similarity search with GPU acceleration
- **Comprehensive Movie Data**: Search across titles, plots, cast, crew, and genres
- **Optimized Performance**: Memory-efficient processing for large-scale datasets

## 🏗️ Architecture

The system uses a two-stage retrieval architecture:

1. **Stage 1 - Bi-Encoder + FAISS**: Fast initial retrieval of candidates
2. **Stage 2 - Cross-Encoder**: Precise re-ranking based on query-document interactions

```
Query → Bi-Encoder → FAISS Search → Top-K Candidates → Cross-Encoder → Final Results
```

## 📁 Project Structure

```
movie-semantic-search/
├── data/
│   ├── scraping.ipynb          # Data collection notebook
│   ├── movie_details.csv       # Main dataset (download required)
│   └── TMDB_movie_dataset_v11  # TMDB dataset (Kaggle)
├── src/
│   ├── config.py               # Configuration settings
│   ├── dataloader.py           # Data loading and preprocessing
│   ├── main.py                 # Main application entry point
│   ├── searcher.py             # Core search functionality
│   ├── test.py                 # Testing utilities
│   └── train.py                # Model training and index building
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 5GB+ free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pkyria/MovieVec-Semantic-Search.git
   cd MovieVec-Semantic-Search
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   For GPU acceleration (recommended):
   ```bash
   # Replace faiss-cpu with faiss-gpu if you have CUDA
   pip install faiss-gpu-cu12
   ```

3. **Download datasets**
   
   **Option A: Movie Details CSV (Recommended)**
   ```bash
   # Download from Google Drive link
   # Place in data/movie_details.csv
   ```
   
   **Option B: TMDB Dataset from Kaggle**
   ```bash
   # Download TMDB_movie_dataset_v11 from Kaggle
   # https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
   # Place in data/TMDB_movie_dataset_v11/
   ```

### Usage

1. **Run Search queries and build the search index** (one-time initialization of embeddings and faiss index)
   ```bash
   cd src
   python main.py
   ```


2. **Interactive search example** (Search queries once engine is running. Type 'quit' to terminate the process.)
   ```
   Search: crime movie with robert de niro and al pacino
   ```

## ⚙️ Configuration

Key settings in `src/config.py`:

```python
# Model Configuration
MODEL = 'all-mpnet-base-v2'  # Primary embedding model
CROSS_ENCODER = "cross-encoder/ms-marco-electra-base"  # Re-ranking model

# Performance Settings
DEVICE = 'cuda'  # or 'cpu'
BATCH_SIZE = 128

# FAISS Index Settings
INDEX_METRIC = 'IndexFlatIP'  # Options: IndexFlatL2, IndexFlatIP, IndexIVFFlat
IVFF_NLIST = 100  # For IndexIVFFlat clustering
```

## 🔍 Search Examples

The system excels at complex semantic queries:

```python
# Cast-based search
searcher.improved_search("crime movie with Al Pacino and Robert De Niro")

# Plot-based search
searcher.improved_search("movie about time travel with romantic subplot")

# Genre and setting
searcher.improved_search("sci-fi thriller taking place on a spaceship")

# Director and style
searcher.improved_search("Christopher Nolan mind-bending movie")
```

## 🧠 Model Details

### Primary Models

- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
  - 768 dimensions
  - Optimized for English semantic similarity - Movie detail oriented (might contain foregin titles but the details are in English)
  - Excellent performance on movie/entertainment content

- **Cross-Encoder**: `cross-encoder/ms-marco-electra-base`
  - Advanced re-ranking for query-document pairs
  - Significant accuracy improvement over bi-encoder alone

### Performance Metrics

- **Index Building**: ~2-3 hours for 800K movies (with GPU)
- **Search Latency**: 
  - Total: ~2-4 seconds per query
- **Memory Usage**: ~4GB (index + models)

## 📊 Dataset Information

### Movie Details CSV
- **Size**: 800K+ movies
- **Fields**: Title, Overview, Cast, Crew, Genres, Release Date, Popularity
- **Languages**: Foreign titles with English descriptions
- **Details**: Created from TMDB Dataset and API requests for Cast details on the most popular movies.
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1t0NoADn3tP03vYNUrKa2EZMqBb5DwhI0/view?usp=sharing)

### TMDB Dataset
- **Source**: [Kaggle TMDB Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
- **Size**: ~1,000,000 movies
- **Format**: Single CSV files
- **Processing**: Use `data/scraping.ipynb` for preprocessing

## 🛠️ Advanced Usage

### Custom Model Configuration

```python
# In config.py, try different models:

# For multilingual support
MODEL = 'paraphrase-multilingual-mpnet-base-v2'

# For faster inference
MODEL = 'all-MiniLM-L12-v2'

# For question-answering style queries
MODEL = 'multi-qa-mpnet-base-dot-v1'
```

### Index Optimization

```python
# For large datasets, use approximate search
INDEX_METRIC = 'IndexIVFFlat'
IVFF_NLIST = 4096  # Adjust based on dataset size

# For maximum accuracy, use exact search
INDEX_METRIC = 'IndexFlatIP'
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size in config.py
   BATCH_SIZE = 32
   
   # Or use CPU
   DEVICE = 'cpu'
   ```

2. **FAISS GPU Transfer Failed**
   ```python
   # The system automatically falls back to CPU search
   # Check GPU memory usage
   ```

3. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

### Performance Optimization

- **GPU Memory**: Reserve 4-6GB for optimal performance
- **RAM**: 8GB+ recommended for large datasets
- **Storage**: SSD recommended for faster index loading

# 🎬 Movie Search Examples

## 🔍 Interactive Search Examples

### Example 1: Cast-Based Search
```python
...

results = searcher.improved_search(query)

...
```

**Output:**
```
Search: crime movie with robert de niro and al pacino
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.26it/s]

🔎 Query: crime movie with robert de niro and al pacino
Indices: [   556 644267   6949 250785 561888  66532   2515  11360 441820 766337]
Distances: [0.66332644 0.65211445 0.6418909  0.63347995 0.63048655 0.62983775
 0.62495655 0.6146907  0.61428285 0.61099875]
Re-ranking 300 candidates with cross-encoder...
Cross-encoder re-ranking completed. Top score: 0.8473

Search Results:

1) Title: Heat, Overview: Obsessive master thief Neil McCauley leads a top-notch crew on various daring heists throughout Los Angeles while determined detective Vincent Hanna pursues him without rest. Each man recognizes and respects the ability and the dedication of the other even though they are aware their cat-and-mouse game may end in violence.

"Genres: Action, Crime, Drama

Cast & Crew: Al Pacino, Robert De Niro, Val Kilmer, Jon Voight, Dov Hoenig

Popularity: 41.522
-----

2) Title: Analyze This, Overview: Countless wiseguy films are spoofed in this film that centers on the neuroses and angst of a powerful Mafia racketeer who suffers from panic attacks. When Paul Vitti needs help dealing with his role in the "family," unlucky shrink Dr. Ben Sobel is given just days to resolve Vitti's emotional crisis and turn him into a happy, well-adjusted gangster.

"Genres: Comedy, Crime

Cast & Crew: Robert De Niro, Billy Crystal, Lisa Kudrow, Chazz Palminteri, Harold Ramis

Popularity: 21.966
-----

3) Title: The Family, Overview: The Manzoni family, a notorious mafia clan, is relocated to Normandy, France under the witness protection program, where fitting in soon becomes challenging as their old habits die hard.

"Genres: Crime, Comedy, Action

Cast & Crew: Robert De Niro, Michelle Pfeiffer, Tommy Lee Jones, Dianna Agron, Luc Besson

Popularity: 19.087
-----

4) Title: Insomnia, Overview: Two Los Angeles homicide detectives are dispatched to a northern town where the sun doesn't set to investigate the methodical murder of a local teen.

"Genres: Thriller, Crime

Cast & Crew: Al Pacino, Robin Williams, Hilary Swank, Martin Donovan, Hillary Seitz

Popularity: 17.626
-----

5) Title: Gang Related, Overview: Two corrupt cops have a successful, seemingly perfect money making scheme- they sell drugs that they seize from dealers, kill the dealers, and blame the crimes on street gangs. Their scheme is going along smoothly until they kill an undercover DEA agent posing as a dealer, and then try to cover-up their crime.

"Genres: Action, Crime, Thriller

Cast & Crew: Jim Belushi, Tupac Shakur, Lela Rochon, Dennis Quaid, Jim Kouf

Popularity: 10.093
-----

6) Title: Q & A, Overview: A young district attorney seeking to prove a case against a corrupt police detective encounters a former lover and her new protector, a crime boss who refuses to help him.

"Genres: Action, Thriller, Crime

Cast & Crew: Nick Nolte, Timothy Hutton, Armand Assante, Patrick O'Neal, Sidney Lumet

Popularity: 9.972
-----

7) Title: Perez., Overview: Demetrio Perez is a tough prosecutor torn between the corruption inherent in his job and the desire to do right by his family. But when opportunity presents itself and his daughter Thea falls in love with a Mafioso’s son, Perez has to cut through the morality of his law-abiding roots and become as dirty as the dangerous criminals he represents.

"Genres: Drama

Cast & Crew: Luca Zingaretti, Marco D'Amore, Simona Tabasco, Giampaolo Fabrizio, Edoardo De Angelis

Popularity: 7.43
-----

8) Title: Rob the Mob, Overview: The true-life story of a crazy-in-love Queens couple who robbed a series of mafia social clubs and got away with it… for a while… until they stumble upon a score bigger than they ever planned and become targets of both the mob and the FBI.

"Genres: Crime, Drama

Cast & Crew: Michael Pitt, Nina Arianda, Andy García, Ray Romano, Max Greene

Popularity: 10.823
-----

9) Title: ...And Justice for All, Overview: An ethical Baltimore defense lawyer disgusted with rampant legal corruption is forced to defend a judge he despises in a rape trial under the threat of being disbarred.

"Genres: Drama

Cast & Crew: Al Pacino, Jack Warden, John Forsythe, Lee Strasberg, Barry Levinson

Popularity: 15.131
-----

10) Title: San Quentin, Overview: An ex-con sets up a program to straighten out hard-core prisoners. Things don't go as planned.

"Genres: Crime

Cast & Crew:  

Popularity: 1.836
-----

Found 10 results in 3.891s
```

---

### Example 2: Plot-Based Search
```python
...

results = searcher.improved_search(query)

...

```


**Output:**
```
Search: movie about superheroes with romantic subplot
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.26it/s]

🔎 Query: movie about superheroes with romantic subplot
Indices: [244832 509100 151256 497264 743062 639867 654207 576235 441562 539667]
Distances: [0.7180302  0.70946664 0.69882476 0.68501115 0.67325056 0.66844416
 0.6626754  0.661201   0.66018116 0.6600257 ]
Re-ranking 300 candidates with cross-encoder...
Cross-encoder re-ranking completed. Top score: 0.8643

Search Results:

1) Title: Hancock, Overview: Hancock is a down-and-out superhero who's forced to employ a PR expert to help repair his image when the public grows weary of all the damage he's inflicted during his lifesaving heroics. The agent's idea of imprisoning the antihero to make the world miss him proves successful, but will Hancock stick to his new sense of purpose or slip back into old habits?

"Genres: Fantasy, Action

Cast & Crew: Will Smith, Charlize Theron, Jason Bateman, Jae Head, Will Smith

Popularity: 29.808
-----

2) Title: SuperBob, Overview: After six years without a date, Robert, the world's only superhero, is looking for love. The only barrier to his plan is that he regularly has to save the world, so he needs to get some time off if he is to meet his superwoman.

"Genres: Action, Comedy, Romance

Cast & Crew: Brett Goldstein, Catherine Tate, Natalia Tena, Laura Haddock, Rupert Christie

Popularity: 7.21
-----

3) Title: Superheroes, Overview: The story of a loving couple who struggles to keep its relationship alive against the inescapable passing of time, told in a nonlinear way over the course of ten years in their lives.

"Genres: Romance, Comedy, Drama

Cast & Crew: Alessandro Borghi, Jasmine Trinca, Greta Scarano, Vinicio Marchioni, Barbara Giordani

Popularity: 7.629
-----

4) Title: Superhero, Overview: A young working class man dreams of fighting crime as costumed hero Nightrider. An enigmatic young woman just might help him on his quest.

"Genres: Drama, Fantasy

Cast & Crew:  

Popularity: 0.6
-----

5) Title: Silver Man, Overview: A street performer fights a pathological man for his girlfriend's love.

"Genres: Drama, Romance

Cast & Crew:  

Popularity: 0.6
-----

6) Title: The Incredible Adventures of Fusion Man, Overview: It's a comedy, inspired superhero films. Dan, aka Fusion Man, is about to spend the evening with his boyfriend Marc, but he has to leave to rescue an innocent person. He discovers that Raphael, a young gay is about to commit suicide, handled by Waco, a super villain. The duel between Waco and Fusion Man will lead to a discussion that will allow everyone to take stock of the situation.

"Genres: Comedy

Cast & Crew:  

Popularity: 1.212
-----

7) Title: Open Letter to All the Terrorists in the World, Overview: Short superhero film.

"Genres:  

Cast & Crew:  

Popularity: 0.6
-----

8) Title: Storytime, Overview: A young boy believes his father is a genuine superhero, and when confronted with a difficult situation - decides to follow in his footsteps.

"Genres: Drama, Action

Cast & Crew:  

Popularity: 0.6
-----

9) Title: The Trouble Couple, Overview: plot is unknown

"Genres:  

Cast & Crew:  

Popularity: 0.6
-----

10) Title: Alter Egos, Overview: At a time when superheroes have lost government funding and public support, a superhero meets a girl who can help him overcome his own emotional crisis.

"Genres: Comedy

Cast & Crew:  

Popularity: 2.817
-----

Found 10 results in 2.856s
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for the embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [TMDB](https://www.themoviedb.org/) for movie data
- The open-source community for making this project possible

## Contact

For questions, suggestions, or collaborations:

- **Issues**: [GitHub Issues](https://github.com/pkyria/MovieVec-Semantic-Search/issues)
- **Email**: pankyriakidis@outlook.com

---

⭐ **Star this repository if you find it cool and you love movies!**

