#!/usr/bin/env bash

# Retrieve first 2,000 e-books from PG
mkdir "training_data"
cd "training_data"
i=1
while [ $i -le 5000 ]; do 
    wget "https://gutenberg.org/cache/epub/$i/pg$i.txt"
    i=$((i+1))
done
# Filter to English-language books
grep -L "Language: English" * | xargs rm


# Loop through all files and swap characters
find "." -type f -print0 | while IFS= read -r -d '' file; do
    echo "Processing: $file"

    # Use a temp file to avoid in-place encoding issues
    tmpfile=$(mktemp)

    sed \
        -e "s/[“”]/\"/g" \
        -e "s/[‘’]/'/g" \
        -e "s/—/-/g" \
        -e "s/–/-/g" \
        -e "s/…/.../g" \
        -e "s/ / /g" \
        "$file" > "$tmpfile"

    mv "$tmpfile" "$file"
done

cd ..

if [ -f "bin/activate" ]; then
    source bin/activate
else
    python -m venv .
    source bin/activate
    pip install wikipedia-api
    pip install python-dotenv
fi

python3 gather_data.py
