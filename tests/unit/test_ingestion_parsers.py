"""Offline parser tests on small hand-written fixtures that mimic each API's documented structure."""

from datetime import date

import pytest

from cliniver.ingestion.clinicaltrials import parse_study
from cliniver.ingestion.europepmc import parse_jats
from cliniver.ingestion.http import redact
from cliniver.ingestion.openfda import parse_label
from cliniver.ingestion.pubmed import parse_pubmed_xml
from cliniver.ingestion.pubtator import parse_bioc
from cliniver.ingestion.who_gho import textualize

PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
<Article><Journal><JournalIssue><PubDate><Year>2019</Year><Month>Nov</Month></PubDate></JournalIssue>
<Title>Test Journal</Title></Journal>
<ArticleTitle>Metformin and <i>lactic</i> acidosis</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Metformin is first-line.</AbstractText>
<AbstractText Label="RESULTS">Risk was not increased (HR 1.0) in eGFR &gt;= 30.</AbstractText></Abstract>
<Language>eng</Language>
<PublicationTypeList><PublicationType>Journal Article</PublicationType>
<PublicationType>Retracted Publication</PublicationType></PublicationTypeList></Article>
<MeshHeadingList><MeshHeading><DescriptorName>Metformin</DescriptorName><QualifierName>adverse effects</QualifierName>
</MeshHeading></MeshHeadingList></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId><ArticleId IdType="pmc">PMC9</ArticleId>
</ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""


def test_pubmed_parse_keeps_qualifiers_and_flags():
    [d] = parse_pubmed_xml(PUBMED_XML, snapshot="2026-11-19", condition="t2d")
    assert d.doc_id == "pubmed:123"
    assert d.title == "Metformin and lactic acidosis"  # inline markup flattened
    assert [s.name for s in d.sections] == ["BACKGROUND", "RESULTS"]
    assert ">= 30" in d.text and "not increased" in d.text  # numbers and negation survive
    assert d.pub_date == date(2019, 11, 1) and d.date_precision == "month"
    assert d.extra["is_retracted"] is True
    assert d.mesh_terms == ["Metformin/adverse effects"]
    assert d.identifiers == {"doi": "10.1/x", "pmcid": "PMC9", "pmid": "123"}


def test_pubmed_other_abstract_fallback():
    # Regression (EXP-01-001): English abstract only in <OtherAbstract Language="eng">.
    xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>9</PMID><Article>
    <Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>T</ArticleTitle></Article>
    <OtherAbstract Type="Publisher" Language="eng"><AbstractText>Apixaban reduced stroke.</AbstractText></OtherAbstract>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    [d] = parse_pubmed_xml(xml, snapshot="s")
    assert d.text == "Apixaban reduced stroke." and d.extra["abstract_source"] == "OtherAbstract"


def test_pubmed_non_english_other_abstract_ignored():
    xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>8</PMID><Article>
    <Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>T</ArticleTitle></Article>
    <OtherAbstract Language="spa"><AbstractText>Texto.</AbstractText></OtherAbstract>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    [d] = parse_pubmed_xml(xml, snapshot="s")
    assert d.text == "" and d.extra["abstract_source"] is None


def test_rxclass_combination_filter_and_fallback(tmp_path):
    from cliniver.config.data_config import RxNavCfg
    from cliniver.ingestion.http import RawStore
    from cliniver.ingestion.normalization import RxNavClient

    def item(min_rxcui, name, cls_id):
        return {"minConcept": {"rxcui": min_rxcui, "name": name},
                "rxclassMinConceptItem": {"classId": cls_id, "className": cls_id, "classType": "ATC1-4"}, "rela": ""}

    def make(responses):
        c = RxNavClient(RxNavCfg(base_url="x", min_interval_s=1, class_sources=["ATC"]), RawStore(tmp_path, "rx", "s"))
        c._get = lambda name, path, params=None: responses[path.split("/")[-1] if "rxcui/" not in path else "related.json"]
        return c

    base = {"rxcui.json": {"idGroup": {"rxnormId": ["1"]}}, "related.json": {"relatedGroup": {"conceptGroup": []}}}
    mixed = make({**base, "byRxcui.json": {"rxclassDrugInfoList": {"rxclassDrugInfo": [
        item("1", "metformin", "A10BA"), item("99", "metformin / sitagliptin", "A10BD")]}}})
    classes = [x for x in mixed.concepts_for("metformin", snapshot="s") if x.kind == "drug_class"]
    assert [x.extra["class_type"] for x in classes] and {x.concept_id for x in classes} == {"rxclass:ATC1-4:A10BA"}
    assert not classes[0].extra["via_combination"]

    combo_only = make({**base, "byRxcui.json": {"rxclassDrugInfoList": {"rxclassDrugInfo": [
        item("99", "sacubitril / valsartan", "C09DX")]}}})
    classes = [x for x in combo_only.concepts_for("sacubitril", snapshot="s") if x.kind == "drug_class"]
    assert classes[0].extra["via_combination"] and classes[0].extra["combination_member"] == "sacubitril / valsartan"


def test_openfda_label_parse():
    label = {"id": "abc", "set_id": "set1", "effective_time": "20240115",
             "contraindications": ["Severe renal impairment (eGFR below 30)."],
             "openfda": {"brand_name": ["Glucophage"], "generic_name": ["METFORMIN HYDROCHLORIDE"], "rxcui": ["1"]}}
    d = parse_label(label, sections=["contraindications", "boxed_warning"], snapshot="s", query_drug="metformin",
                    condition="t2d")
    assert d.pub_date == date(2024, 1, 15)
    assert [s.name for s in d.sections] == ["contraindications"]  # absent sections skipped
    assert "eGFR below 30" in d.text and d.license.startswith("CC0")


def test_clinicaltrials_parse():
    study = {"hasResults": True, "protocolSection": {
        "identificationModule": {"nctId": "NCT0001", "briefTitle": "T"},
        "descriptionModule": {"briefSummary": "A  trial."},
        "statusModule": {"overallStatus": "COMPLETED", "completionDateStruct": {"date": "2020-05"}},
        "designModule": {"phases": ["PHASE3"], "studyType": "INTERVENTIONAL"}}}
    d = parse_study(study, snapshot="s", condition="hypertension")
    assert d.doc_id == "clinicaltrials:NCT0001" and d.text == "A trial."
    assert d.pub_date == date(2020, 5, 1) and d.date_precision == "month"
    assert d.extra["has_results"] is True


def test_pubtator_parse():
    doc = {"id": "77", "passages": [{"infons": {"type": "abstract"}, "annotations": [
        {"text": "metformin", "infons": {"type": "Chemical", "identifier": "MESH:D008687"},
         "locations": [{"offset": 5, "length": 9}]}]}],
        "relations": [{"infons": {"type": "Treat", "score": "0.9", "role1": {"identifier": "MESH:D008687"},
                                  "role2": {"identifier": "MESH:D003924"}}}]}
    ents, rels = parse_bioc(doc)
    assert ents[0].identifier == "MESH:D008687" and ents[0].offset == 5
    assert rels[0].relation_type == "Treat" and rels[0].score == pytest.approx(0.9)


def test_jats_parse_sections():
    xml = b"""<article><front><abstract><p>Short abstract.</p></abstract></front><body>
    <sec><title>Methods</title><p>We enrolled 100 adults.</p><table-wrap/></sec></body></article>"""
    d = parse_jats(xml, {"pmcid": "PMC1", "title": "X", "firstPublicationDate": "2021-02-03", "license": "cc by"},
                   snapshot="s", condition="t2d")
    assert [s.name for s in d.sections] == ["ABSTRACT", "Methods"]
    assert d.extra["n_tables"] == 1 and d.license == "cc by"


def test_who_textualize():
    row = {"Id": 5, "IndicatorCode": "BP_04", "SpatialDim": "GLOBAL", "SpatialDimType": "GLOBAL",
           "TimeDim": 2015, "Dim1": "SEX_BTSX", "NumericValue": 22.1, "Low": 19.9, "High": 24.5}
    d = textualize(row, "Raised blood pressure", snapshot="s")
    assert "22.1" in d.text and "both sexes" in d.text and d.role == "statistics"


def test_redact_hides_credentials():
    assert redact({"email": "a@b.c", "api_key": "k", "term": "x"}) == {
        "email": "<redacted>", "api_key": "<redacted>", "term": "x"}


def test_openfda_unstructured_label_fallback():
    """Regression (EXP-01-003): labels with only spl_unclassified_section were empty documents."""
    base = {"id": "u1", "set_id": "s", "openfda": {"generic_name": ["METFORMIN HYDROCHLORIDE"]},
            "spl_unclassified_section": ["Metformin is contraindicated in severe renal impairment."]}
    d = parse_label(base, sections=["contraindications"], snapshot="s", query_drug="metformin", condition="t2d")
    assert "severe renal impairment" in d.text and d.extra["unstructured_label"] is True
    # A structured label never uses the fallback, even if it also carries an unclassified section
    d2 = parse_label({**base, "contraindications": ["eGFR below 30."]}, sections=["contraindications"],
                     snapshot="s", query_drug="metformin", condition="t2d")
    assert [s.name for s in d2.sections] == ["contraindications"] and d2.extra["unstructured_label"] is False


def test_jats_paragraphs_directly_under_body():
    """Regression (EXP-01-003): letters/editorials with <p> directly under <body> parsed to empty text."""
    xml = b"""<article><body><p>Frailty alone should not preclude anticoagulation.</p><p>Second point.</p>
    <sec><title>Ref</title><p>Sectioned text.</p></sec></body></article>"""
    d = parse_jats(xml, {"pmcid": "PMC2", "title": "L"}, snapshot="s", condition="af_anticoagulation")
    assert [s.name for s in d.sections] == ["BODY", "Ref"] and "Frailty alone" in d.text
