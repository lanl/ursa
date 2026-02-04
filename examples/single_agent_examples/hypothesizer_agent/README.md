For the Sci-Fi Bill of Rights demo, see sci_fi_bill_of_rights_inputs/README.txt.
It requires downloading some public TXT files.

If you want to use this on PDFs, particularly ones that need OCR, then you
need to install some additional Python libraries.  At this time we haven't
required those baked into URSA.

On a mac you need:

```
brew update
brew install ocrmypdf tesseract
# NOTE: Feb 1, 2026 - gettext did not install on my mac so had to
#       build from source, this is LENGTHY process, but 100%
#       works:
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

Note that the second `[OCR]` line will only show up if the `skip` version
still produced no good data to read, this is called the `force` version.

Once a doc has been OCRed (either version) the reader will pick this up automatically
in the future (ie it will only run this the first time it needs to).
