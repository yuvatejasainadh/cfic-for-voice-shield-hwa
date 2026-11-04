# Alembic Database Migrations Scaffolding

This directory is designated for database migration scripts (PostgreSQL) tracking schema evolutions for:
- API Client Keys and Rate-Limit Tiers
- Analysis Jobs & State Management
- Forensic Evidence Records

To initialize in a subsequent stage:
```bash
alembic init .
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```
