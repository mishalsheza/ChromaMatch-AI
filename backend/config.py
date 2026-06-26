import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-fallback-secret-key')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit for uploads
    
    # Parse allowed origins from env or default to localhost for dev
    raw_origins = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173')
    ALLOWED_ORIGINS = [origin.strip() for origin in raw_origins.split(',') if origin.strip()]

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    
    # Strip debug info in production
    STRIP_DEBUG = True

# Dictionary to map environment name to config class
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
