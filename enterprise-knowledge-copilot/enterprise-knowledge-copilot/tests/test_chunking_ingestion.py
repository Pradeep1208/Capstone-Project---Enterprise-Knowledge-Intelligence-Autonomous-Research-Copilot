import pytest

from copilot.chunking import chunk_document, fixed_split, recursive_split
from copilot.ingestion import Document, Page, load_document


def make_doc(text, page=7):
    return Document("d1", "Doc", "d1.pdf", [Page(text, page=page, section="Item 1")])


def test_recursive_respects_size_and_keeps_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    chunks = recursive_split(text, 500, 80)
    assert all(len(c) <= 500 + 80 for c in chunks)
    assert "Paragraph 9." in " ".join(chunks)


def test_fixed_split_overlaps():
    text = "alpha " * 400
    chunks = fixed_split(text, 300, 60)
    assert len(chunks) > 3 and all(len(c) <= 300 for c in chunks)


def test_chunks_keep_page_metadata():
    chunks = chunk_document(make_doc("sentence one. " * 200), "recursive", 400, 50)
    assert len(chunks) > 1 and {c.page for c in chunks} == {7} and chunks[0].section == "Item 1"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_document(make_doc("x" * 500), "fixed", 100, 100)


def test_loaders():
    assert load_document(b"# Title\nhello world body text here for the test", "a.md").pages[0].section == "Title"
    csv = load_document(b"name,city\nAnn,Berlin\nBo,Potsdam\n", "p.csv")
    assert "city: Berlin" in csv.pages[0].text
    js = load_document(b'[{"k": "v1"}, {"k": "v2"}]', "p.json")
    assert "k: v2" in js.pages[0].text
    html = load_document(b"<html><script>x=1</script><body><p>Item 1A. Risk Factors</p><p>Some risk text is here.</p></body></html>", "f.html")
    assert "x=1" not in html.pages[0].text


def test_rejects_bad_files():
    with pytest.raises(ValueError):
        load_document(b"MZ...", "malware.exe")
    with pytest.raises(ValueError):
        load_document(b"x" * 2_000_000, "big.txt", max_mb=1)
    with pytest.raises(ValueError):
        load_document(b"not a pdf", "broken.pdf")
