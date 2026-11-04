# VoiceShield RIAS Semantic-Risk Evaluation Rubric

## 1. Subsystem Role & Boundaries

The **Risk Identification and Assessment Subsystem (RIAS)** performs semantic-risk analysis on transcribed audio chunks.
- **Input:** Complete ASR transcript of a conversation chunk.
- **Output:** Exactly one categorical verdict: `LOW`, `MEDIUM`, or `HIGH`.
- **System Guardrail:** Any failure, malformed response, or invalid category produces `INCONCLUSIVE` (handled by the caller / orchestrator).
- **Architectural Boundary:** The LLM evaluates conversational semantics and fraud indicators. It does NOT inspect audio authenticity, acoustic features, or make final session decisions. Session decision fusion is strictly governed by the Temporal Correlation & Decision Engine (TCED).

---

## 2. Classification Categories

### `LOW` — Benign / Safe Communication
- **Definition:** Normal personal, social, commercial, or informational conversation with zero indicators of fraud, coercion, urgent financial demands, or credential phishing.
- **Examples:**
  - Daily chitchat, weather, travel plans, recipes, family news.
  - Legitimate commercial inquiries (e.g. asking store hours, scheduling a hair appointment, discussing product specifications).
  - Normal customer service discussions where no credentials, payment info, or coercive tactics are present.
  - Mentions of sensitive terms in purely benign, factual contexts (e.g. "I went to the police station to report a lost bicycle", "My bank app sent an OTP to my phone for my grocery order").

### `MEDIUM` — Ambiguous / Suspicious / Pre-Scam Indicators
- **Definition:** Ambiguous, unsolicited, mildly pushy, or unverified claims requesting callbacks, personal details, or unusual actions, without immediate severe extortion, legal threats, or direct credential theft.
- **Examples:**
  - Unsolicited calls offering pre-approved loans, credit cards, or lottery tickets asking the recipient to confirm identity or call a specific unverified number.
  - Vague delivery notices asking the recipient to click an unverified SMS link or verify residential address.
  - Cold callers aggressively pitching investment schemes without immediate demands for passwords or banking PINs.
  - Incomplete or fragmented dialogues where intent is evasive or suspiciously urgent but lacks explicit demands.

### `HIGH` — Critical Fraud / Severe Social Engineering
- **Definition:** Explicit social engineering, credential theft, financial extortion, law enforcement impersonation, digital arrest threats, urgent fund transfers, or fraudulent instructions.
- **Mandatory TCED Rule:** Any `HIGH` classification triggers an immediate `SUSPICIOUS` session outcome (`HIGH_RIAS_OVERRIDE`).
- **Examples:**
  - **Credential Phishing:** Direct solicitation of One-Time Passwords (OTP), UPI PINs, ATM PINs, CVV, NetBanking passwords, or Aadhaar numbers.
  - **Digital Arrest / Impersonation:** Caller claims to be CBI, Police, Cyber Crime Branch, Customs, or TRAI, alleging a parcel with contraband/drugs was intercepted, and threatening arrest unless money is transferred to a "RBI verification account".
  - **Urgent Financial Demands:** Demands to immediately wire money via UPI/RTGS/gift cards to avoid utility service disconnection (electricity, SIM card block, bank KYC expiry).
  - **Remote Access Exploitation:** Instructions to install remote-desktop software (AnyDesk, TeamViewer, RustDesk) or APK files to "resolve an account issue".
  - **Kidnapping / Emergency Impersonation:** AI voice clone or caller claiming a child/family member was arrested or injured, demanding immediate ransom.

---

## 3. Evaluation Rules & Guidelines

1. **Context Over Keywords:**
   - Do NOT classify as HIGH solely because words like "police", "OTP", "bank", or "password" appear. If the speaker says "I had to enter my OTP to buy the ticket", that is LOW.
   - Look for *direction of solicitation* and *coercive intent*.
2. **Language Neutrality:**
   - Transcripts may be in English, Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Odia, or code-switched (e.g., Hinglish, Tenglish).
   - Language identity or script must NEVER influence the risk classification.
3. **Faithfulness to Transcript:**
   - Classify strictly based on the provided text. Do not invent details not present in the transcript.
4. **Deterministic Formatting:**
   - The LLM must output valid JSON containing `"risk_level"` (`LOW`, `MEDIUM`, or `HIGH`) and a concise `"reason"`.
