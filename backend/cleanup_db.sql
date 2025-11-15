-- Cleanup script to reset database and fix vector dimension mismatch
-- This script will:
-- 1. Drop document_embeddings table (to fix vector dimension issue)
-- 2. Truncate all data tables
-- 3. Reset sequences

-- Drop document_embeddings table to fix vector dimension mismatch
DROP TABLE IF EXISTS document_embeddings CASCADE;

-- Truncate all data tables
TRUNCATE TABLE documents CASCADE;
TRUNCATE TABLE distributions CASCADE;
TRUNCATE TABLE capital_calls CASCADE;
TRUNCATE TABLE adjustments CASCADE;
TRUNCATE TABLE funds CASCADE;

-- Reset sequences
ALTER SEQUENCE IF EXISTS funds_id_seq RESTART WITH 1;
ALTER SEQUENCE IF EXISTS documents_id_seq RESTART WITH 1;
ALTER SEQUENCE IF EXISTS capital_calls_id_seq RESTART WITH 1;
ALTER SEQUENCE IF EXISTS distributions_id_seq RESTART WITH 1;
ALTER SEQUENCE IF EXISTS adjustments_id_seq RESTART WITH 1;

-- Verify cleanup
SELECT 'Cleanup completed successfully!' AS status;
