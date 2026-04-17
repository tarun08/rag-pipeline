import os

docs_dir = "docs"
files = os.listdir(docs_dir)

chunk_size = 800
chunk_overlap = 50

print(f"{'Filename':<20} | {'Size (KB)':<10} | {'Chars':<10} | {'Est. Chunks':<12}")
print("-" * 60)

for file in files:
    path = os.path.join(docs_dir, file)
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            chars = len(content)
            size_kb = chars / 1024
            # Simplified chunk estimation: (chars - overlap) / (chunk_size - overlap)
            est_chunks = max(1, (chars - chunk_overlap) // (chunk_size - chunk_overlap))
            print(f"{file:<20} | {size_kb:<10.2f} | {chars:<10} | {est_chunks:<12}")
