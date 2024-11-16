# deployment/deploy.py

import argparse
import os
import subprocess
from typing import Optional
import yaml
import logging

class Deployer:
    """Handle deployment operations for the risk prediction system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.env = os.getenv('DEPLOYMENT_ENV', 'development')
        
    def deploy(self, 
               target: str,
               version: Optional[str] = None,
               dry_run: bool = False) -> None:
        """
        Deploy the application to the specified target environment.
        
        Args:
            target: Target environment (development/staging/production)
            version: Specific version to deploy
            dry_run: If True, only show what would be deployed
        """
        try:
            self._validate_deployment_config(target)
            
            if dry_run:
                self.logger.info(f"Dry run deployment to {target}")
                return
                
            if target == 'kubernetes':
                self._deploy_to_kubernetes(version)
            elif target == 'docker':
                self._deploy_to_docker(version)
            else:
                raise ValueError(f"Unsupported deployment target: {target}")
                
        except Exception as e:
            self.logger.error(f"Deployment failed: {str(e)}")
            raise

    def _deploy_to_kubernetes(self, version: Optional[str]) -> None:
        """Deploy to Kubernetes cluster."""
        try:
            # Apply configurations
            kubectl_cmd = ["kubectl", "apply", "-f", "deployment/kubernetes/"]
            subprocess.run(kubectl_cmd, check=True)
            
            # Update image if version specified
            if version:
                update_cmd = [
                    "kubectl", "set", "image",
                    "deployment/risk-prediction-api",
                    f"api=risk-prediction:{version}"
                ]
                subprocess.run(update_cmd, check=True)
                
            self.logger.info("Kubernetes deployment completed successfully")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Kubernetes deployment failed: {str(e)}")

    def _deploy_to_docker(self, version: Optional[str]) -> None:
        """Deploy using Docker Compose."""
        try:
            cmd = ["docker-compose", "-f", "deployment/docker-compose.yml", "up", "-d"]
            if version:
                cmd.extend(["--build"])
                os.environ['IMAGE_VERSION'] = version
                
            subprocess.run(cmd, check=True)
            self.logger.info("Docker deployment completed successfully")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Docker deployment failed: {str(e)}")

    def _validate_deployment_config(self, target: str) -> None:
        """Validate deployment configuration."""
        config_path = f"deployment/config/{target}.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        required_keys = ['database_url', 'redis_url', 'api_key']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

def main():
    parser = argparse.ArgumentParser(description='Deploy risk prediction system')
    parser.add_argument('target', choices=['kubernetes', 'docker'],
                       help='Deployment target')
    parser.add_argument('--version', help='Version to deploy')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform a dry run')
    
    args = parser.parse_args()
    
    deployer = Deployer()
    deployer.deploy(args.target, args.version, args.dry_run)

if __name__ == '__main__':
    main()
