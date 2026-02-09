For the Sci-Fi Bill of Rights demo, see sci_fi_bill_of_rights_inputs/README.txt.
It requires downloading some public TXT files.

If you want to use this on PDFs, particularly ones that need OCR, then you
need to install some additional Python libraries.  At this time we haven't
required those baked into URSA.

On a mac you need:

```
brew update
brew install ocrmypdf tesseract
# NOTE: Feb 1, 2026 - gettext did not install on my Mac so had to
#       build from source. This is a LENGTHY but reliable process:
#       brew install --build-from-source gettext
#       once gettext is installed, you can go back to
#       brew install ocrmypdf
pip install pypdf # you need this too in your Python env.
```

Once these are installed, you should see something like this, if OCR is needed:

```
[READING]: your_doc.pdf
[OCR]: mode=skip (441 chars, 22 pages) -> your_doc.pdf.ocr.skip.pdf
[OCR]: still low after skip-text; retrying with force-ocr -> your_doc.pdf.ocr.force.pdf
```

Note that the first `[OCR]` line will only show up if the PDF reading fails and there
are no text layers discovered (this `skips` some complex / lengthy OCR techniques
and tries a quick and dirty one.).

Note that the second `[OCR]` line will show up only if the `skip` version
still produced no good data to read. This is called the `force` version.

Once a doc has been OCRed (either version) the reader will automatically
remember this for the future (i.e. it will run this only the first time it
needs to).
