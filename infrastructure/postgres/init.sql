-- CFIC Database Initialization Script (PostgreSQL 16)
-- Schema scaffolding for API client credentials, rate tiers, analysis jobs, and forensic logs

CREATE TABLE IF NOT EXISTS api_clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_name VARCHAR(128) NOT NULL,
    api_key_hash VARCHAR(256) NOT NULL UNIQUE,
    rate_limit_per_minute INT NOT NULL DEFAULT 30,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_jobs (
    job_id UUID PRIMARY KEY,
    client_id UUID REFERENCES api_clients(id),
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    audio_filename VARCHAR(256),
    audio_duration_seconds NUMERIC(6, 3),
    final_outcome VARCHAR(32),
    reason_code VARCHAR(64),
    pella_spoof_score NUMERIC(5, 4),
    rias_risk_level VARCHAR(16),
    detected_language VARCHAR(8),
    transcript_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON analysis_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON analysis_jobs(status);
