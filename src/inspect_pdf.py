import pymupdf

doc = pymupdf.open('ViT-XAI-Lyo_Final_Paper.pdf')
print("Total pages:", len(doc))
for i, page in enumerate(doc):
    lines = page.get_text().splitlines()
    first = lines[0] if lines else "EMPTY"
    last = lines[-1] if lines else "EMPTY"
    print(f"Page {i+1}: ({len(lines)} lines)")
    print(f"   First: {first[:60]}")
    print(f"   Last:  {last[:60]}")
