# mnemosyne
Memory module for LLMs

## 🛠️ Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional and recommended)
- Required Python packages (see `requirements.txt`)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/kurcontko/mnemosyne.git
cd mnemosyne
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the required services using Docker Compose:
```bash
docker-compose up -d
```

5. Start
```bash
streamlit run app.py
```