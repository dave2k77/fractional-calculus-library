# TestPyPI Deployment Failure - Troubleshooting Guide

## Issue

The TestPyPI deployment is failing because the GitHub environment `testpypi` is not properly configured.

## Why This Happened

After renaming the repository from `fractional-calculus-library` to `hpfracc`, the GitHub environment configuration may need to be updated or recreated.

## Solution Options

### Option 1: Configure TestPyPI Environment (Recommended if you want TestPyPI)

1. **Create GitHub Environment:**
   - Go to: https://github.com/dave2k77/hpfracc/settings/environments
   - Click **New environment**
   - Name it: `testpypi`
   - Click **Configure environment**

2. **Set up TestPyPI Trusted Publishing:**
   - Go to: https://test.pypi.org/manage/account/publishing/
   - Add a new pending publisher:
     - **TestPyPI Project Name:** `hpfracc`
     - **Owner:** `dave2k77`
     - **Repository name:** `hpfracc`
     - **Workflow name:** `pypi-publish.yml`
     - **Environment name:** `testpypi`
   - Click **Add**

3. **Test the deployment:**
   - Go to: https://github.com/dave2k77/hpfracc/actions
   - Select "Publish to PyPI" workflow
   - Click **Run workflow**
   - Select branch: `main`
   - Click **Run workflow**

### Option 2: Disable TestPyPI Job (Simpler - if you don't need TestPyPI)

If you don't need TestPyPI testing, you can remove or disable the TestPyPI job. The workflow will work fine without it.

## Current Workflow Behavior

- **Regular Releases:** Only publish to production PyPI (testpypi job skipped)
- **Manual Trigger:** Attempts to publish to TestPyPI (fails if environment not configured)

The TestPyPI job only runs when manually triggered via `workflow_dispatch`, not during regular releases.

## Recommendation

If you don't actively use TestPyPI, you can either:
1. Ignore the failure (it only affects manual test deployments)
2. Remove the `publish-to-testpypi` job from the workflow
3. Configure it properly if you want TestPyPI testing

The production PyPI publishing (via releases) should work fine as long as the `pypi` environment is configured correctly.

