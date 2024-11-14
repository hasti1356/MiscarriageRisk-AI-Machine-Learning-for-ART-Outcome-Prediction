# src/monitoring.py

import psutil
import time
import threading
from typing import Dict, Any
import logging

class PerformanceMonitor:
    """Monitor system resources and model performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': []
        }

    def start_monitoring(self) -> None:
        """Start monitoring system resources."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop monitoring system resources."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()

    def _monitor_resources(self) -> None:
        """Monitor system resources periodically."""
        while self.monitoring:
            try:
                self.metrics['cpu_usage'].append(psutil.cpu_percent())
                self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
                self.metrics['disk_usage'].append(psutil.disk_usage('/').percent)
                
                if any(usage > 90 for usage in [self.metrics['cpu_usage'][-1], 
                                              self.metrics['memory_usage'][-1],
                                              self.metrics['disk_usage'][-1]]):
                    self.logger.warning("High resource usage detected")
                    
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def get_metrics(self) -> Dict[str, list]:
        """Get current monitoring metrics."""
        return self.metrics

    def clear_metrics(self) -> None:
        """Clear stored metrics."""
        self.metrics = {key: [] for key in self.metrics}
