CREATE INDEX queries_lower_text_created_at ON queries (LOWER(text), created_at);
