DROP TABLE IF EXISTS fact_sales CASCADE;
DROP TABLE IF EXISTS dim_customer CASCADE;
DROP TABLE IF EXISTS dim_book CASCADE;
DROP TABLE IF EXISTS dim_category CASCADE;
DROP TABLE IF EXISTS dim_location CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS dim_profit_range CASCADE;
DROP TABLE IF EXISTS etl_metadata CASCADE;

-- ============================================================================
-- DIMENSION TABLES
-- ============================================================================

-- Dimension: Customer
CREATE TABLE dim_customer (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_name VARCHAR(255),
    customer_type VARCHAR(50),  -- Individual, Corporate, Student
    registration_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE dim_customer IS 'Customer dimension table';
COMMENT ON COLUMN dim_customer.customer_id IS 'Unique customer identifier from OLTP';
COMMENT ON COLUMN dim_customer.customer_type IS 'Type of customer: Individual, Corporate, Student';

-- Dimension: Book
CREATE TABLE dim_book (
    book_id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    author VARCHAR(255),
    publisher VARCHAR(255),
    publish_year INTEGER,
    isbn VARCHAR(20),
    category VARCHAR(100),
    language VARCHAR(50),
    page_count INTEGER,
    edition_count INTEGER,
    has_api_data BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(title, author)  -- Prevent duplicate books
);

COMMENT ON TABLE dim_book IS 'Book dimension table with enriched API data';
COMMENT ON COLUMN dim_book.has_api_data IS 'Whether book data was enriched from Open Library API';

-- Dimension: Category
CREATE TABLE dim_category (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE NOT NULL,
    category_description TEXT,
    parent_category VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE dim_category IS 'Book category dimension';

-- Insert default categories
INSERT INTO dim_category (category_name, category_description) VALUES
('college', 'College and university textbooks'),
('competition', 'Competitive exam preparation books'),
('education', 'General education books'),
('engineering', 'Engineering and technical books');

-- Dimension: Location (City/State)
CREATE TABLE dim_location (
    location_id SERIAL PRIMARY KEY,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    region VARCHAR(50),  -- North, South, East, West, Central
    country VARCHAR(50) DEFAULT 'India',
    postal_code VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(city, state)
);

COMMENT ON TABLE dim_location IS 'Shipping location dimension';

-- Dimension: Date (Time dimension)
CREATE TABLE dim_date (
    date_id SERIAL PRIMARY KEY,
    full_date DATE UNIQUE NOT NULL,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name VARCHAR(20),
    day INTEGER NOT NULL CHECK (day BETWEEN 1 AND 31),
    day_of_week INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    day_of_week_name VARCHAR(20),
    week_of_year INTEGER,
    is_weekend BOOLEAN NOT NULL,
    is_holiday BOOLEAN DEFAULT FALSE,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE dim_date IS 'Date dimension for time-based analysis';
COMMENT ON COLUMN dim_date.day_of_week IS '0=Monday, 6=Sunday';

-- Dimension: Profit Range (for classification)
CREATE TABLE dim_profit_range (
    profit_range_id SERIAL PRIMARY KEY,
    range_name VARCHAR(20) UNIQUE NOT NULL,  -- Low, Medium, High
    min_profit DECIMAL(10,2) NOT NULL,
    max_profit DECIMAL(10,2) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE dim_profit_range IS 'Profit range classification dimension';

-- Insert profit ranges
INSERT INTO dim_profit_range (range_name, min_profit, max_profit, description) VALUES
('Low', 0.00, 50.00, 'Profit below 50 INR'),
('Medium', 50.01, 100.00, 'Profit between 50-100 INR'),
('High', 100.01, 999999.99, 'Profit above 100 INR');

-- ============================================================================
-- FACT TABLE
-- ============================================================================

CREATE TABLE fact_sales (
    sale_id SERIAL PRIMARY KEY,
    
    -- Foreign Keys to Dimensions
    customer_id VARCHAR(50) REFERENCES dim_customer(customer_id),
    book_id INTEGER REFERENCES dim_book(book_id),
    category_id INTEGER REFERENCES dim_category(category_id),
    location_id INTEGER REFERENCES dim_location(location_id),
    date_id INTEGER REFERENCES dim_date(date_id),
    profit_range_id INTEGER REFERENCES dim_profit_range(profit_range_id),
    
    -- Transaction Details (Facts/Measures)
    purchase_date DATE NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    item_price DECIMAL(10,2) NOT NULL CHECK (item_price >= 0),
    total_amount DECIMAL(10,2) NOT NULL CHECK (total_amount >= 0),
    profit DECIMAL(10,2) NOT NULL,
    profit_margin DECIMAL(5,2),  -- Percentage
    
    -- Metadata
    source_system VARCHAR(50) DEFAULT 'OLTP',
    load_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (total_amount = quantity * item_price)
);

COMMENT ON TABLE fact_sales IS 'Fact table for book sales transactions';
COMMENT ON COLUMN fact_sales.profit_margin IS 'Profit margin as percentage';
COMMENT ON COLUMN fact_sales.load_date IS 'When this record was loaded to DWH';

-- ============================================================================
-- ETL METADATA TABLE
-- ============================================================================

CREATE TABLE etl_metadata (
    etl_id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100) NOT NULL,
    run_date TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL,  -- success, failed, running
    rows_extracted INTEGER,
    rows_transformed INTEGER,
    rows_loaded INTEGER,
    error_message TEXT,
    execution_time_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE etl_metadata IS 'ETL pipeline execution metadata';

-- ============================================================================
-- INDEXES FOR QUERY PERFORMANCE
-- ============================================================================

-- Fact table indexes
CREATE INDEX idx_fact_sales_date ON fact_sales(date_id);
CREATE INDEX idx_fact_sales_book ON fact_sales(book_id);
CREATE INDEX idx_fact_sales_customer ON fact_sales(customer_id);
CREATE INDEX idx_fact_sales_category ON fact_sales(category_id);
CREATE INDEX idx_fact_sales_location ON fact_sales(location_id);
CREATE INDEX idx_fact_sales_purchase_date ON fact_sales(purchase_date);
CREATE INDEX idx_fact_sales_profit_range ON fact_sales(profit_range_id);

-- Composite indexes for common queries
CREATE INDEX idx_fact_sales_date_category ON fact_sales(date_id, category_id);
CREATE INDEX idx_fact_sales_date_location ON fact_sales(date_id, location_id);

-- Dimension table indexes
CREATE INDEX idx_dim_book_category ON dim_book(category);
CREATE INDEX idx_dim_book_publish_year ON dim_book(publish_year);
CREATE INDEX idx_dim_location_state ON dim_location(state);
CREATE INDEX idx_dim_date_year_month ON dim_date(year, month);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Sales Summary by Category
CREATE OR REPLACE VIEW v_sales_by_category AS
SELECT 
    c.category_name,
    COUNT(f.sale_id) as total_transactions,
    SUM(f.quantity) as total_books_sold,
    SUM(f.total_amount) as total_revenue,
    SUM(f.profit) as total_profit,
    AVG(f.profit_margin) as avg_profit_margin,
    AVG(f.item_price) as avg_price
FROM fact_sales f
JOIN dim_category c ON f.category_id = c.category_id
GROUP BY c.category_name
ORDER BY total_revenue DESC;

-- View: Monthly Sales Trend
CREATE OR REPLACE VIEW v_monthly_sales_trend AS
SELECT 
    d.year,
    d.month,
    d.month_name,
    COUNT(f.sale_id) as total_transactions,
    SUM(f.total_amount) as total_revenue,
    SUM(f.profit) as total_profit
FROM fact_sales f
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;

-- View: Top Books by Revenue
CREATE OR REPLACE VIEW v_top_books AS
SELECT 
    b.title,
    b.author,
    b.publisher,
    COUNT(f.sale_id) as times_sold,
    SUM(f.quantity) as total_quantity,
    SUM(f.total_amount) as total_revenue,
    SUM(f.profit) as total_profit
FROM fact_sales f
JOIN dim_book b ON f.book_id = b.book_id
GROUP BY b.title, b.author, b.publisher
ORDER BY total_revenue DESC
LIMIT 50;

-- View: Sales by Location
CREATE OR REPLACE VIEW v_sales_by_location AS
SELECT 
    l.state,
    l.city,
    COUNT(f.sale_id) as total_transactions,
    SUM(f.total_amount) as total_revenue,
    SUM(f.profit) as total_profit,
    AVG(f.item_price) as avg_price
FROM fact_sales f
JOIN dim_location l ON f.location_id = l.location_id
GROUP BY l.state, l.city
ORDER BY total_revenue DESC;

-- ============================================================================
-- STORED PROCEDURES
-- ============================================================================

-- Procedure: Update Dimension Slowly Changing Dimension (SCD Type 2)
CREATE OR REPLACE FUNCTION update_dim_book_scd2(
    p_title VARCHAR(500),
    p_author VARCHAR(255),
    p_new_publisher VARCHAR(255),
    p_new_publish_year INTEGER
)
RETURNS INTEGER AS $$
DECLARE
    v_book_id INTEGER;
BEGIN
    -- Check if book exists
    SELECT book_id INTO v_book_id
    FROM dim_book
    WHERE title = p_title AND author = p_author;
    
    IF v_book_id IS NOT NULL THEN
        -- Update existing record
        UPDATE dim_book
        SET 
            publisher = p_new_publisher,
            publish_year = p_new_publish_year,
            updated_at = CURRENT_TIMESTAMP
        WHERE book_id = v_book_id;
        
        RETURN v_book_id;
    ELSE
        -- Insert new record
        INSERT INTO dim_book (title, author, publisher, publish_year)
        VALUES (p_title, p_author, p_new_publisher, p_new_publish_year)
        RETURNING book_id INTO v_book_id;
        
        RETURN v_book_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate profit margin
CREATE OR REPLACE FUNCTION calculate_profit_margin(
    p_profit DECIMAL(10,2),
    p_total_amount DECIMAL(10,2)
)
RETURNS DECIMAL(5,2) AS $$
BEGIN
    IF p_total_amount = 0 THEN
        RETURN 0;
    END IF;
    RETURN ROUND((p_profit / p_total_amount) * 100, 2);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger: Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_fact_sales_updated_at
    BEFORE UPDATE ON fact_sales
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_dim_book_updated_at
    BEFORE UPDATE ON dim_book
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Grant permissions to application user
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO pustaka_admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO pustaka_admin;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO pustaka_admin;

-- ============================================================================
-- SAMPLE QUERIES FOR TESTING
-- ============================================================================

-- Query 1: Total sales by category
-- SELECT * FROM v_sales_by_category;

-- Query 2: Monthly trend
-- SELECT * FROM v_monthly_sales_trend ORDER BY year, month;

-- Query 3: Top 10 books
-- SELECT * FROM v_top_books LIMIT 10;

-- Query 4: Sales by state
-- SELECT state, SUM(total_revenue) as revenue
-- FROM v_sales_by_location
-- GROUP BY state
-- ORDER BY revenue DESC;

-- ============================================================================
-- END OF SCHEMA DEFINITION
-- ============================================================================