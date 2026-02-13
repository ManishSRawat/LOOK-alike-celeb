import re

# Path to the keras-vggface models.py file
file_path = r'C:\Users\Acer\anaconda3\Lib\site-packages\keras_vggface\models.py'

print(f"Reading {file_path}...")
# Read from backup if it exists, otherwise from original
backup_path = file_path + '.backup'
try:
    with open(backup_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print("Using backup file as source")
except FileNotFoundError:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Create backup
    print(f"Creating backup at {backup_path}...")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)

# Replace ALL occurrences of '/' with '_' in any name= parameter
# This catches all cases: name='conv1/7x7_s2', name='conv1/7x7_s2/bn', name=var + "/bn", etc.
pattern = r"name=(['\"])([^'\"]*)/([^'\"]*)\1"
matches = re.findall(pattern, content)
count = 0

# Keep replacing until no more matches found
while True:
    new_content, replacements = re.subn(pattern, lambda m: f"name={m.group(1)}{m.group(2)}_{m.group(3)}{m.group(1)}", content)
    if replacements == 0:
        break
    count += replacements
    content = new_content

# Also handle concatenated strings like: name=var + "/bn"
pattern2 = r'\+ "(/[^"]*)"'
matches2 = re.findall(pattern2, content)
count2 = len(matches2)
content = re.sub(pattern2, lambda m: f'+ "{m.group(1).replace("/", "_")}"', content)

print(f"Replaced {count} layer names with '/' in quoted strings")
print(f"Replaced {count2} layer names with '/' in concatenated strings")

# Write the patched content
print(f"Writing patched content to {file_path}...")
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Successfully patched keras-vggface models.py!")
print(f"Total replacements: {count + count2}")
print("The app should now work with Keras 3.x")
