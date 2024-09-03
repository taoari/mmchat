from datetime import datetime
from pypdf import PdfReader

def _get_pdf_metadata(filename):
    
    # Open the PDF file
    pdf_reader = PdfReader(filename)
    
    # Extract metadata
    metadata = pdf_reader.metadata
    
    # Create a metadata dictionary
    metadata_dict = {
        'Title': metadata.get('/Title', 'N/A'),
        'Author': metadata.get('/Author', 'N/A'),
        'Subject': metadata.get('/Subject', 'N/A'),
        'Keywords': metadata.get('/Keywords', 'N/A'),
        'Creator': metadata.get('/Creator', 'N/A'),
        'Producer': metadata.get('/Producer', 'N/A'),
        'CreationDate': metadata.get('/CreationDate', 'N/A'),
        'ModDate': metadata.get('/ModDate', 'N/A')
    }
    
    return metadata_dict

def _convert_pdf_datetime(pdf_datetime_str):
    # Remove the leading 'D:'
    if pdf_datetime_str.startswith('D:'):
        pdf_datetime_str = pdf_datetime_str[2:16]
    
    # Define the format string based on the PDF DateTime format
    format_str = '%Y%m%d%H%M%S'
    
    try:
        # Parse the string to a datetime object
        dt = datetime.strptime(pdf_datetime_str, format_str)
        
        # Return the datetime object
        return dt
    except ValueError as e:
        print(f"Error parsing date: {e}")
        return None