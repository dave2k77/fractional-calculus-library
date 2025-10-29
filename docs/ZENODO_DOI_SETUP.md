# How to Get a DOI for HPFRACC Library

This guide explains how to obtain a Digital Object Identifier (DOI) for the HPFRACC library using Zenodo, the recommended free service for academic software.

## Why Zenodo?

Zenodo is the **recommended** choice for GitHub repositories because:

- ✅ **Free** - No cost for DOI assignment
- ✅ **Automatic** - Integrates with GitHub releases via webhooks
- ✅ **Academic standard** - Widely accepted in research publications
- ✅ **Version tracking** - Each release gets its own DOI
- ✅ **Persistent archives** - Long-term preservation of software versions
- ✅ **Citation exports** - Automatic BibTeX and other citation formats

## Step-by-Step Setup

### Step 1: Create Zenodo Account

1. Go to [Zenodo.org](https://zenodo.org/)
2. Click "Sign up" (top right)
3. Sign in with your GitHub account (recommended) or create a new account
4. Complete profile with:
   - Full name: Davian R. Chin
   - Affiliation: University of Reading, Department of Biomedical Engineering
   - Email: d.r.chin@pgr.reading.ac.uk

### Step 2: Link GitHub Repository

1. Log into Zenodo
2. Go to **Settings** → **GitHub** (or visit directly: https://zenodo.org/account/settings/github/)
3. Click "Connect GitHub"
4. Authorize Zenodo to access your repositories
5. Find `fractional-calculus-library` in the list
6. Toggle the switch to **ON** for automatic DOI assignment on releases

### Step 3: Create a GitHub Release Tag

When you create a GitHub release with a tag, Zenodo will automatically:

1. Detect the new release
2. Create a DOI (takes ~5-10 minutes)
3. Archive the repository snapshot
4. Generate citation information

**Creating a Release:**

```bash
# Tag the current version
git tag -a v3.0.0 -m "HPFRACC v3.0.0: Neural Fractional SDE Solvers with Intelligent Backend Selection"
git push origin v3.0.0

# Or create release via GitHub web interface:
# 1. Go to: https://github.com/dave2k77/fractional-calculus-library/releases/new
# 2. Select tag: v3.0.0 (or create new tag)
# 3. Title: "HPFRACC v3.0.0"
# 4. Description: [Add release notes]
# 5. Click "Publish release"
```

### Step 4: Customize Zenodo Metadata (Optional but Recommended)

After Zenodo creates the DOI (usually within 10 minutes), you can customize:

1. Go to [Your Zenodo uploads](https://zenodo.org/deposit)
2. Find the auto-generated upload for your release
3. Click "Edit" to add:
   - **Title**: "HPFRACC: High-Performance Fractional Calculus Library with Neural Fractional SDE Solvers"
   - **Authors**: 
     - Davian R. Chin (University of Reading, Department of Biomedical Engineering)
   - **Description**: [Full description of library capabilities]
   - **Keywords**: fractional calculus, neural networks, stochastic differential equations, GPU acceleration, machine learning
   - **License**: MIT
   - **Version**: 3.0.0
   - **Programming Language**: Python
   - **Related Publications**: [Link to paper if published]

### Step 5: Get Your DOI

Once processed (typically 5-10 minutes after release):

1. Visit your Zenodo uploads page
2. Find your release
3. The DOI will be displayed, e.g.: `10.5281/zenodo.17476041` (example from HPFRACC v3.0.0)
4. Click "Get citation" for formatted citations

## Citation Format

After receiving your DOI, update citations to:

**BibTeX:**
```bibtex
@software{hpfracc2025,
  title={HPFRACC: High-Performance Fractional Calculus Library with Neural Fractional SDE Solvers},
  author={Chin, Davian R.},
  year={2025},
  version={3.0.0},
  doi={10.5281/zenodo.17476041},
  url={https://github.com/dave2k77/fractional-calculus-library},
  publisher={Zenodo},
  note={Department of Biomedical Engineering, University of Reading}
}
```

**APA Style:**
```
Chin, D. R. (2025). HPFRACC: High-Performance Fractional Calculus Library 
with Neural Fractional SDE Solvers (Version 3.0.0) [Computer software]. 
Zenodo. https://doi.org/10.5281/zenodo.17476041
```

## Manual Upload (Alternative)

If you prefer manual control:

1. Go to [Zenodo Upload](https://zenodo.org/deposit/new)
2. Upload the repository as a ZIP file (or use GitHub integration but publish manually)
3. Fill in metadata manually
4. Click "Publish" to get DOI immediately

## Best Practices

1. **Version-based DOIs**: Each major version should get its own DOI
   - v3.0.0 → DOI A
   - v3.1.0 → DOI B (new DOI)
   - v3.1.1 (patch) → Can use same DOI or new one

2. **Release Notes**: Include comprehensive release notes in GitHub releases for better metadata

3. **Badge**: Add a DOI badge to your README:
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17476041.svg)](https://doi.org/10.5281/zenodo.17476041)
   ```
   (Example from HPFRACC v3.0.0)

4. **CITATION File**: Create `CITATION.cff` in repository root:
   ```yaml
   cff-version: 1.2.0
   message: "If you use this software, please cite it as below."
   title: "HPFRACC: High-Performance Fractional Calculus Library with Neural Fractional SDE Solvers"
   authors:
     - given-names: Davian R.
       family-names: Chin
       affiliation: University of Reading, Department of Biomedical Engineering
       email: d.r.chin@pgr.reading.ac.uk
version: 3.0.0
date-released: 2025-01-28
doi: 10.5281/zenodo.17476041
   license: MIT
   repository-code: https://github.com/dave2k77/fractional-calculus-library
   ```

## Alternative DOI Services

While Zenodo is recommended, other options include:

1. **Figshare** - Similar to Zenodo, also free
2. **Software Heritage** - For long-term archival (assigns SWHIDs)
3. **Institutional Repository** - If University of Reading provides DOI services
4. **Journal Supplementary Materials** - If publishing associated research paper

## Next Steps

1. ✅ Set up Zenodo account
2. ✅ Connect GitHub repository
3. ✅ Create GitHub release v3.0.0
4. ✅ Customize Zenodo metadata
5. ✅ Update documentation with DOI
6. ✅ Add DOI badge to README.md
7. ✅ Create CITATION.cff file

## Resources

- [Zenodo Documentation](https://help.zenodo.org/)
- [GitHub Releases Guide](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [Citation File Format](https://citation-file-format.github.io/)
- [CITATION.cff Generator](https://citation-file-format.github.io/cff-initializer-javascript/)

## Notes

- DOI assignment is **permanent** - once assigned, it cannot be changed
- Each version/release can have its own DOI
- Zenodo provides free storage and DOI assignment
- The DOI format is: `10.5281/zenodo.XXXXXXXXX` (example: `10.5281/zenodo.17476041` for HPFRACC v3.0.0)
- DOI becomes available within 5-10 minutes after GitHub release

