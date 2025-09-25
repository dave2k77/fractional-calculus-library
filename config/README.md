# Configuration Files

This folder contains configuration files for the fractional calculus library:

## Files

- **`environment.yml`**: Conda environment specification for the `fracnn` environment
- **`.flake8`**: Python linting configuration
- **`.envrc`**: Directory environment configuration (direnv)

## Usage

- To create the conda environment: `conda env create -f config/environment.yml`
- Linting configuration is automatically detected by flake8
- Environment variables are set when entering the directory (if using direnv)

**Note**: The `.readthedocs.yml` file is kept in the root directory as required by Read the Docs.
