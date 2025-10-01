#!/usr/bin/env python3
"""
Find missing citations specifically in the introduction.
"""

import re

def check_introduction_citations():
    """Check all citations in the introduction against the references.bib file."""
    print("🔍 CHECKING INTRODUCTION CITATIONS")
    print("="*50)
    print()
    
    # Extract citations from introduction
    with open('/home/davianc/fractional-calculus-library/manuscript/sections/01_introduction.tex', 'r') as f:
        intro_content = f.read()
    
    # Find all citations
    citation_groups = re.findall(r'\\citep\{([^}]+)\}', intro_content)
    
    all_citations = []
    for group in citation_groups:
        individual_citations = [c.strip() for c in group.split(',')]
        all_citations.extend(individual_citations)
    
    unique_citations = list(set(all_citations))
    print(f"📚 Citations found in introduction: {len(unique_citations)}")
    for citation in sorted(unique_citations):
        print(f"  • {citation}")
    
    # Check references.bib
    with open('/home/davianc/fractional-calculus-library/manuscript/references.bib', 'r') as f:
        bib_content = f.read()
    
    # Find all reference keys
    references = re.findall(r'@\w+\{([^,\s]+)', bib_content)
    
    print(f"\n📖 References in .bib file: {len(references)}")
    
    # Check for missing citations
    missing_citations = []
    for citation in unique_citations:
        if citation not in references:
            missing_citations.append(citation)
    
    print(f"\n🔍 MISSING CITATIONS ANALYSIS:")
    if missing_citations:
        print("❌ MISSING CITATIONS FOUND:")
        for citation in missing_citations:
            print(f"  • {citation}")
        return missing_citations
    else:
        print("✅ ALL CITATIONS FOUND in references.bib")
        return []

def add_missing_citations(missing_citations):
    """Add missing citations to references.bib."""
    if not missing_citations:
        return
    
    print(f"\n🔧 ADDING {len(missing_citations)} MISSING CITATIONS...")
    
    # Standard citation templates for common missing citations
    citation_templates = {
        'baydin2018automatic': '''@article{baydin2018automatic,
  title={Automatic differentiation in machine learning: a survey},
  author={Baydin, Atilim Gunes and Pearlmutter, Barak A and Radul, Alexey Andreyevich and Siskind, Jeffrey Mark},
  journal={Journal of Machine Learning Research},
  volume={18},
  number={1},
  pages={5595--5637},
  year={2017}
}''',
        'podlubny1999fractional': '''@book{podlubny1999fractional,
  title={Fractional differential equations: an introduction to fractional derivatives, fractional differential equations, to methods of their solution and some of their applications},
  author={Podlubny, Igor},
  year={1999},
  publisher={Academic press}
}''',
        'kilbas2006theory': '''@book{kilbas2006theory,
  title={Theory and applications of fractional differential equations},
  author={Kilbas, Anatoly A and Srivastava, Hari M and Trujillo, Juan J},
  volume={204},
  year={2006},
  publisher={Elsevier}
}''',
        'chen2018neural': '''@article{chen2018neural,
  title={Neural ordinary differential equations},
  author={Chen, Tian Qi and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David K},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}''',
        'raissi2019physics': '''@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational physics},
  volume={378},
  pages={686--707},
  year={2019}
}''',
    }
    
    # Read current references.bib
    with open('/home/davianc/fractional-calculus-library/manuscript/references.bib', 'r') as f:
        bib_content = f.read()
    
    # Add missing citations
    for citation in missing_citations:
        if citation in citation_templates:
            print(f"  ✅ Adding {citation}")
            bib_content += f"\n\n{citation_templates[citation]}"
        else:
            print(f"  ⚠️  Template not available for {citation}")
    
    # Write back
    with open('/home/davianc/fractional-calculus-library/manuscript/references.bib', 'w') as f:
        f.write(bib_content)
    
    print(f"✅ Added {len([c for c in missing_citations if c in citation_templates])} citations to references.bib")

def main():
    """Main function to check and fix introduction citations."""
    missing_citations = check_introduction_citations()
    
    if missing_citations:
        add_missing_citations(missing_citations)
        print("\n🔄 Citations added - manuscript needs recompilation")
        return True
    else:
        print("\n✅ No missing citations - manuscript is ready")
        return False

if __name__ == "__main__":
    needs_recompilation = main()
    exit(0)














