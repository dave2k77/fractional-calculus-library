# Development Tools

This folder contains shell scripts and tools for development workflow:

## Scripts

- **`activate_env.sh`**: Activates the conda environment (supports both `fracnn` and `hpfracc-env`)
- **`startup_env.sh`**: Startup script that initializes the development environment
- **`setup_auto_activation.sh`**: Sets up automatic environment activation

## Usage

```bash
# Activate the development environment
source tools/activate_env.sh

# Or use the startup script
source tools/startup_env.sh

# Set up automatic activation
./tools/setup_auto_activation.sh
```

## Note

These scripts are Linux/Unix specific. The corresponding Windows batch and PowerShell scripts have been removed since this is a Linux environment.





