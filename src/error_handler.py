#!/usr/bin/env python3
"""
Error Handler - Wave 4: Production Infrastructure
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



Comprehensive error handling, recovery strategies, and error tracking.
"""

import json
import time
import traceback
from pathlib import Path
from typing import Optional, Any, Dict, List, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
import logging

T = TypeVar('T')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    INDEX = "index"
    RETRIEVAL = "retrieval"
    FILTER = "filter"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    CONFIG = "config"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Error context for debugging"""
    error_id: str
    timestamp: str
    category: ErrorCategory
    severity: ErrorSeverity
    exception_type: str
    exception_message: str
    traceback_str: str
    component: str
    operation: str
    query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'category': self.category.value,
            'severity': self.severity.value,
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'traceback': self.traceback_str,
            'component': self.component,
            'operation': self.operation,
            'query': self.query,
            'metadata': self.metadata,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful
        }


class ErrorHandler:
    """Central error handling system"""
    
    def __init__(self, error_log_dir: Optional[Path] = None, max_retries: int = 3):
        self.error_log_dir = error_log_dir or (Path.home() / '.aria_errors')
        self.error_log_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.errors: List[ErrorContext] = []
        self.error_counts: Dict[ErrorCategory, int] = {}
    
    def classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify exception"""
        msg = str(exception).lower()
        if 'index' in msg or 'file not found' in msg:
            return ErrorCategory.INDEX
        elif 'retrieval' in msg or 'search' in msg:
            return ErrorCategory.RETRIEVAL
        elif 'filter' in msg:
            return ErrorCategory.FILTER
        elif 'timeout' in msg:
            return ErrorCategory.TIMEOUT
        elif 'memory' in msg or 'resource' in msg:
            return ErrorCategory.RESOURCE
        elif 'config' in msg:
            return ErrorCategory.CONFIG
        return ErrorCategory.UNKNOWN
    
    def assess_severity(self, category: ErrorCategory) -> ErrorSeverity:
        """Assess severity"""
        if category in [ErrorCategory.INDEX, ErrorCategory.CONFIG]:
            return ErrorSeverity.CRITICAL
        elif category in [ErrorCategory.RESOURCE, ErrorCategory.RETRIEVAL]:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.FILTER, ErrorCategory.TIMEOUT]:
            return ErrorSeverity.MEDIUM
        return ErrorSeverity.LOW
    
    def handle_error(
        self,
        exception: Exception,
        component: str,
        operation: str,
        query: Optional[str] = None
    ) -> ErrorContext:
        """Handle error with full context"""
        import hashlib
        
        category = self.classify_error(exception)
        severity = self.assess_severity(category)
        
        error_ctx = ErrorContext(
            error_id=hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:12],
            timestamp=datetime.now().isoformat(),
            category=category,
            severity=severity,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            traceback_str=traceback.format_exc(),
            component=component,
            operation=operation,
            query=query
        )
        
        self.errors.append(error_ctx)
        self.error_counts[category] = self.error_counts.get(category, 0) + 1
        
        # Log
        error_file = self.error_log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(error_file, 'a') as f:
            f.write(json.dumps(error_ctx.to_dict()) + '\n')
        
        logger.error(f"[{severity.value}] {component}.{operation}: {exception}")
        
        return error_ctx
    
    def retry_with_backoff(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Retry with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    sleep_time = 2 ** attempt
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries + 1} in {sleep_time}s")
                    time.sleep(sleep_time)
        
        if last_exception:
            raise last_exception
        raise Exception("Retry failed")


# Global handler
_global_handler: Optional[ErrorHandler] = None

def get_global_handler() -> ErrorHandler:
    global _global_handler
    if _global_handler is None:
        _global_handler = ErrorHandler()
    return _global_handler


def handle_errors(component: str, operation: str):
    """Decorator for error handling"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_global_handler()
                query = kwargs.get('query') or (args[0] if args and isinstance(args[0], str) else None)
                handler.handle_error(e, component, operation, query)
                raise
        return wrapper
    return decorator


if __name__ == '__main__':
    handler = ErrorHandler()
    print("Error handler initialized")
