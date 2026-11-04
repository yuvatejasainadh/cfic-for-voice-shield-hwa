# CFIC × VoiceShield HackVersion — Contribution Guidelines

**Team SAIL for HackWithAditya**  
**Repository Lead & Maintainer:** Sainadh

---

## 1. Collaborative Git & PR Architecture

To ensure genuine, traceable collaboration across all four team members, our repository uses a GitHub PR-and-Merge workflow.

- **Sainadh (Project Lead):** Owns and maintains the authoritative GitHub repository, establishes the baseline scaffold on `main`, reviews pull requests, and merges approved contributions.
- **Team Members (Varun, Varshu, Pallavi):** Work independently using their own personal GitHub accounts, clone/fork the repository, implement their assigned modules on dedicated feature branches, and submit Pull Requests targeting `main`.

No artificial branch commits or simulated contributor activity should ever be pushed under the lead's account.

---

## 2. Step-by-Step Contributor Workflow

### Step 1: Clone the Official Repository
Each team member clones the repository to their local workstation:
```bash
git clone <GITHUB_REPOSITORY_URL>
cd CFIC-VoiceShield-HackVersion
```

### Step 2: Set Up Local Development Environment
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install core and development dependencies
pip install -r requirements-dev.txt

# Copy environment template
copy .env.example .env   # Windows
cp .env.example .env     # Linux/macOS
```

### Step 3: Create a Dedicated Feature Branch
Always branch off the latest `main`:
```bash
git checkout main
git pull origin main
git checkout -b <your-feature-branch>
```

#### Recommended Branch Naming:
- **Varun:** `feature/varun-pellav2-model-preparation`
- **Varshu:** `feature/varshu-pellav2-docker-service`
- **Pallavi:** `feature/pallavi-transcription-service`
- **Hotfixes / Lead:** `feature/project-lead-<feature-name>`

### Step 4: Implement Assigned Tasks in Isolated Directories
Work strictly within your assigned workspace to minimize merge conflicts:
- **Varun:** `models/`, `docs/models/`, `docs/benchmarks/`
- **Varshu:** `services/pella/`
- **Pallavi:** `services/transcription/`, `docs/benchmarks/`
- **Sainadh:** `app/`, `services/rias/`, core orchestration, root configuration

### Step 5: Run Tests Locally Before Committing
Run the automated test suite and ensure all checks pass:
```bash
python -m pytest -v
```

### Step 6: Commit and Push Using Your GitHub Account
Commit clean, focused changes with informative commit messages:
```bash
git add services/pella/
git commit -m "feat(pella): integrate validated PellaV2 ONNX inference runtime"
git push -u origin <your-feature-branch>
```

### Step 7: Open a Pull Request Targeting `main`
On GitHub, open a PR from your feature branch to `main`. Fill in the PR description template:
- Summary of changes implemented
- Test results and verification steps
- Artifact manifests and benchmark tables attached (if applicable)
- Confirmation that no secrets, credentials, or private audio files are included

### Step 8: Code Review and Integration
Sainadh reviews the PR, ensures service contract compliance, verifies the mandatory TCED safety rule is intact, and merges the PR into `main`.

---

## 3. Pull Request Acceptance Criteria

Every PR must satisfy the following checklist before merge approval:

- [ ] **Tests Pass:** All automated unit and contract tests in `tests/` pass with zero failures.
- [ ] **No Secrets:** No API keys, tokens, credentials, or sensitive URLs are present in code or committed files.
- [ ] **No Binary Weights:** Large model weights (>50 MB) must NOT be committed to Git. Store artifacts in cloud storage or document reproducible download/quantization scripts.
- [ ] **No Unverified Claims:** Do not report speculative benchmark accuracy or support for unverified languages without empirical test logs.
- [ ] **Contract Compliance:** Any change to API endpoints or response schemas must update `docs/API_CONTRACTS.md`.
- [ ] **TCED Safety Preserved:** The mandatory rule `RIAS HIGH -> SUSPICIOUS` must never be altered or bypassed.

---

## 4. Repository Hygiene & Safety Rules

- Never use `git push --force` on `main`.
- Never commit private audio recordings or transcripts.
- Ensure `.env` is listed in `.gitignore` and never committed.
