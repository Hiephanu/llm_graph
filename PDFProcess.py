import PyPDF2

def pdf_to_text(pdf_path, output_txt):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Thêm dấu xuống dòng sau dấu chấm (.)
                page_text = page_text.replace('. ', '.\n')
                text += page_text

    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

if __name__ == "__main__":
    pdf_path = 'dataset/source.pdf'
    output_txt = 'dataset/source.txt'

    pdf_to_text(pdf_path, output_txt)

    print("PDF converted to text successfully!")
