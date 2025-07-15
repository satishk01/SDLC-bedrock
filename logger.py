import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import os

class LLMLogger:
    """Global logger for LLM responses and operations"""
    
    def __init__(self, log_file: str = None, log_level: int = logging.INFO):
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate timestamp-based log file name if not provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"llm_operations_{timestamp}.log"
            
        # If no directory specified, put in logs folder
        if not os.path.dirname(log_file):
            log_file = os.path.join(log_dir, log_file)
            
        self.log_file = log_file
        self.logger = logging.getLogger('LLMLogger')
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """Setup file and console handlers for logging"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.'
        if log_dir != '.' and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_llm_request(self, 
                       prompt: str, 
                       request_type: str = "general",
                       model_id: str = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Log LLM request details"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "LLM_REQUEST",
            "request_type": request_type,
            "model_id": model_id,
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "metadata": metadata or {}
        }
        
        self.logger.info(f"LLM Request: {json.dumps(log_data, indent=2)}")
    
    def log_llm_response(self, 
                        response: str, 
                        request_type: str = "general",
                        model_id: str = None,
                        processing_time: Optional[float] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """Log LLM response details"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "LLM_RESPONSE",
            "request_type": request_type,
            "model_id": model_id,
            "response_length": len(response),
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "processing_time_seconds": processing_time,
            "metadata": metadata or {}
        }
        
        self.logger.info(f"LLM Response: {json.dumps(log_data, indent=2)}")
    
    def log_llm_error(self, 
                     error: str, 
                     request_type: str = "general",
                     model_id: str = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Log LLM error details"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "LLM_ERROR",
            "request_type": request_type,
            "model_id": model_id,
            "error": str(error),
            "metadata": metadata or {}
        }
        
        self.logger.error(f"LLM Error: {json.dumps(log_data, indent=2)}")
    
    def log_workflow_step(self, 
                         step_name: str, 
                         status: str, 
                         details: Optional[Dict[str, Any]] = None):
        """Log workflow step information"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "WORKFLOW_STEP",
            "step_name": step_name,
            "status": status,
            "details": details or {}
        }
        
        self.logger.info(f"Workflow Step: {json.dumps(log_data, indent=2)}")
    
    def log_info(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log general information"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "INFO",
            "message": message,
            "metadata": metadata or {}
        }
        
        self.logger.info(f"Info: {json.dumps(log_data, indent=2)}")
    
    def log_error(self, message: str, error: Optional[Exception] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log general error"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "ERROR",
            "message": message,
            "error": str(error) if error else None,
            "metadata": metadata or {}
        }
        
        self.logger.error(f"Error: {json.dumps(log_data, indent=2)}")

# Global logger instance with timestamp-based naming
llm_logger = LLMLogger()

# Convenience functions for easy import
def log_llm_request(prompt: str, request_type: str = "general", model_id: str = None, metadata: Optional[Dict[str, Any]] = None):
    llm_logger.log_llm_request(prompt, request_type, model_id, metadata)

def log_llm_response(response: str, request_type: str = "general", model_id: str = None, processing_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
    llm_logger.log_llm_response(response, request_type, model_id, processing_time, metadata)

def log_llm_error(error: str, request_type: str = "general", model_id: str = None, metadata: Optional[Dict[str, Any]] = None):
    llm_logger.log_llm_error(error, request_type, model_id, metadata)

def log_workflow_step(step_name: str, status: str, details: Optional[Dict[str, Any]] = None):
    llm_logger.log_workflow_step(step_name, status, details)

def log_info(message: str, metadata: Optional[Dict[str, Any]] = None):
    llm_logger.log_info(message, metadata)

def log_error(message: str, error: Optional[Exception] = None, metadata: Optional[Dict[str, Any]] = None):
    llm_logger.log_error(message, error, metadata)