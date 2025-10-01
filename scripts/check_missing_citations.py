#!/usr/bin/env python3
"""
Check for missing citations in the manuscript.
"""

import re
from pathlib import Path

def extract_citations_from_file(filepath):
    """Extract all citations from a LaTeX file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find all \citep{...} patterns
        citations = re.findall(r'\\citep\{([^}]+)\}', content)
        
        # Split multiple citations (e.g., \citep{key1,key2,key3})
        all_citations = []
        for citation_group in citations:
            individual_citations = [c.strip() for c in citation_group.split(',')]
            all_citations.extend(individual_citations)
        
        return all_citations
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

def extract_references_from_bib(bib_file):
    """Extract all reference keys from the .bib file."""
    try:
        with open(bib_file, 'r') as f:
            content = f.read()
        
        # Find all @article{key, @book{key, etc.
        references = re.findall(r'@\w+\{([^,\s]+)', content)
        return references
    except Exception as e:
        print(f"Error reading {bib_file}: {e}")
        return []

def main():
    """Check for missing citations."""
    print("🔍 CHECKING FOR MISSING CITATIONS...")
    print()
    
    manuscript_dir = Path("/home/davianc/fractional-calculus-library/manuscript")
    
    # Extract all citations from manuscript files
    all_citations = []
    
    # Check all .tex files
    for tex_file in manuscript_dir.glob("**/*.tex"):
        if tex_file.name != "hpfracc_paper.tex":  # Skip main file
            citations = extract_citations_from_file(tex_file)
            if citations:
                print(f"📄 {tex_file.name}: {len(citations)} citations found")
                all_citations.extend(citations)
    
    # Remove duplicates
    unique_citations = list(set(all_citations))
    print(f"\n📊 Total unique citations found: {len(unique_citations)}")
    
    # Extract references from .bib file
    bib_file = manuscript_dir / "references.bib"
    references = extract_references_from_bib(bib_file)
    print(f"📚 Total references in .bib file: {len(references)}")
    
    # Find missing citations
    missing_citations = []
    for citation in unique_citations:
        if citation not in references:
            missing_citations.append(citation)
    
    print()
    if missing_citations:
        print("❌ MISSING CITATIONS FOUND:")
        for citation in missing_citations:
            print(f"  • {citation}")
        print()
        print("🔧 These citations need to be added to references.bib")
    else:
        print("✅ ALL CITATIONS FOUND in references.bib")
        print("🎯 No missing citations detected")
    
    print()
    print("📋 All citations used:")
    for citation in sorted(unique_citations):
        status = "✅" if citation in references else "❌"
        print(f"  {status} {citation}")
    
    return len(missing_citations) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)














