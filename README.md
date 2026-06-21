# FINM-37000-Project
## Setup

### Prerequisites
- [uv](https://docs.astral.sh/uv/) installed
- A [Databento](https://databento.com) account and API key

### 1. Fork and clone the repo
Each team member should fork this repository to their own GitHub account, then clone their fork:
```bash
git clone <your-fork-url>
cd finm_37000_project
```

Add the main repo as an upstream remote so you can pull in updates:
```bash
git remote add upstream https://github.com/marijajov65/finm_37000_project.git
git fetch upstream
git merge upstream/main
```

### 2. Install dependencies
```bash
uv sync
```
This creates a `.venv` and installs all dependencies pinned in `uv.lock`, using the Python version specified in `.python-version`.

### 3. Set up environment variables
Copy the example file and fill in your own Databento API key:
```bash
cp .env.example .env
```
Then edit `.env`:
DATABENTO_API_KEY=your_key_here

### 4. Run the project
```bash
uv run src/main.py
```

### 5. Run tests
```bash
uv run pytest
```

## Contributing
1. Create a branch on your fork for your changes.
2. Commit and push to your fork.
3. Open a pull request from your fork into `marijajov65/finm_37000_project` (`main` branch).
4. Tag teammates for review before merging.

## Status
🚧 Early setup phase — repo structure and tooling are in place; analysis scope and data pipeline are in progress.