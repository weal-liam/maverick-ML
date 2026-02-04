import chardet

def detect_encoding(file_path):
    # Read a portion of the file to detect encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read(1000)

    # Detect the encoding using chardet
    encoding = chardet.detect(raw_data)['encoding']

    return encoding

def to_utf8(file_path, original_encoding):
    #open the file with the original encoding and read its content
    with open(file_path, 'r', encoding=original_encoding) as f:
        content = f.read()
        
    #write the content back to the file with utf-8 encoding
    with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(content)

def handle_file_encoding(file_path):
    encoding = detect_encoding(file_path)

    if encoding != 'utf-8':
        to_utf8(file_path, encoding)