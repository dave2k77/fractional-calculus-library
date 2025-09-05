#!/usr/bin/env python3
"""
Verification script to check that all citation keys in the LaTeX paper
are present in the references.bib file.
"""

import re
import sys

def extract_citations_from_tex(tex_file):
    """Extract all citation keys from the LaTeX file."""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Find all \cite{...} patterns
    citations = re.findall(r'\\cite\{([^}]+)\}', content)
    
    # Split multiple citations (e.g., \cite{key1,key2,key3})
    all_citations = []
    for citation in citations:
        keys = [key.strip() for key in citation.split(',')]
        all_citations.extend(keys)
    
    return set(all_citations)

def extract_keys_from_bib(bib_file):
    """Extract all citation keys from the BibTeX file."""
    with open(bib_file, 'r') as f:
        content = f.read()
    
    # Find all @article{key, patterns
    keys = re.findall(r'@\w+\{([^,]+),', content)
    return set(keys)

def main():
    tex_file = 'hpfracc_paper.tex'
    bib_file = 'references.bib'
    
    print("Verifying citation keys...")
    print("=" * 50)
    
    # Extract citations from LaTeX file
    tex_citations = extract_citations_from_tex(tex_file)
    print(f"Citations found in LaTeX file: {len(tex_citations)}")
    print(f"Keys: {sorted(tex_citations)}")
    print()
    
    # Extract keys from BibTeX file
    bib_keys = extract_keys_from_bib(bib_file)
    print(f"Keys found in BibTeX file: {len(bib_keys)}")
    print(f"Keys: {sorted(bib_keys)}")
    print()
    
    # Check for missing keys
    missing_in_bib = tex_citations - bib_keys
    missing_in_tex = bib_keys - tex_citations
    
    if missing_in_bib:
        print("❌ MISSING IN BIBTEX FILE:")
        for key in sorted(missing_in_bib):
            print(f"  - {key}")
        print()
    else:
        print("✅ All citations from LaTeX file are present in BibTeX file")
        print()
    
    if missing_in_tex:
        print("ℹ️  KEYS IN BIBTEX BUT NOT USED IN LATEX:")
        for key in sorted(missing_in_tex):
            print(f"  - {key}")
        print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY:")
    print(f"  LaTeX citations: {len(tex_citations)}")
    print(f"  BibTeX keys: {len(bib_keys)}")
    print(f"  Missing in BibTeX: {len(missing_in_bib)}")
    print(f"  Unused in LaTeX: {len(missing_in_tex)}")
    
    if missing_in_bib:
        print("\n❌ VERIFICATION FAILED: Some citations are missing from BibTeX file")
        return 1
    else:
        print("\n✅ VERIFICATION PASSED: All citations are present")
        return 0

if __name__ == "__main__":
    sys.exit(main())
